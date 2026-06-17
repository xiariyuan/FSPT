#!/usr/bin/env python3
"""
Local Motion Prior Recovery Audit (Stage-0)

Tests whether neighbor motion priors can predict re-entry positions well enough
to create an oracle gap when combined with local DINO cosine search in a crop.

Four prior centers:
  1. base_center       — current tracker position at reentry
  2. velocity_center   — constant-velocity extrapolation from last visible frames
  3. translation_center — mean displacement of escort neighbors
  4. affine_center     — weighted similarity transform from escort neighbors

For each center, crops of radius {32, 64, 96} are extracted and local DINO
cosine search produces top-k=5 candidates. Oracle gap is measured.

Usage:
  python scripts/audit_local_motion_prior_recovery.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --output-dir outputs/local_motion_prior_audit_stage0 \
    --max-batches 30 --escort-k 16 --topk 5
"""

from __future__ import annotations
import argparse, json, sys, math, os, signal
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Model forward timed out")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config loader (same as audit scripts)
# ---------------------------------------------------------------------------
def load_config(config_path: str):
    from omegaconf import OmegaConf
    import yaml
    def _load_and_resolve(path, _seen=None):
        if _seen is None: _seen = set()
        path = str(Path(path).resolve())
        if path in _seen: return OmegaConf.create({})
        _seen.add(path)
        with open(path) as f: raw = yaml.safe_load(f)
        if raw is None: return OmegaConf.create({})
        defaults = raw.pop("defaults", []) or []
        base = OmegaConf.create({})
        for entry in defaults:
            if isinstance(entry, str) and entry != "_self_":
                bp = Path(path).parent / f"{entry}.yaml"
                if not bp.exists(): bp = Path(path).parent / entry
                if bp.exists(): base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
            elif isinstance(entry, dict):
                for k, v in entry.items():
                    if k != "_self_":
                        bp = Path(path).parent / f"{v}.yaml"
                        if not bp.exists(): bp = Path(path).parent / v
                        if bp.exists(): base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
        current = OmegaConf.create(raw)
        return OmegaConf.merge(base, current)
    return _load_and_resolve(config_path)


# ---------------------------------------------------------------------------
# t_last_visible extraction
# ---------------------------------------------------------------------------
def find_t_last_visible(occ_mask_n: np.ndarray, t_reentry: int) -> int:
    """Find the last visible frame before the occlusion run ending at t_reentry-1.

    Scans backward from t_reentry-1 to find the first visible frame.
    """
    for t in range(t_reentry - 1, -1, -1):
        if not occ_mask_n[t]:
            return t
    return 0


# ---------------------------------------------------------------------------
# Escort point selection
# ---------------------------------------------------------------------------
def select_escort_points(
    tracks_b: np.ndarray,      # (N, T, 2) normalized
    occ_b: np.ndarray,         # (N, T) bool
    query_idx: int,
    t_last_vis: int,
    t_reentry: int,
    K: int = 16,
    jitter_sigma: float = 50.0,
) -> dict:
    """Select K escort points visible at both t_last_vis and t_reentry."""
    N = tracks_b.shape[0]
    q_pos = tracks_b[query_idx, t_last_vis]  # (2,)

    candidates = []
    for i in range(N):
        if i == query_idx:
            continue
        # Must be visible at both frames
        if occ_b[i, t_last_vis] or occ_b[i, t_reentry]:
            continue
        pos_last = tracks_b[i, t_last_vis]
        pos_re = tracks_b[i, t_reentry]
        # Skip if any position is (0,0) — likely invalid
        if (np.linalg.norm(pos_last) < 1e-6) or (np.linalg.norm(pos_re) < 1e-6):
            continue

        # Compute jitter: velocity variance over the occlusion window
        # Use frames around t_last_vis for stability
        velocities = []
        for tt in range(max(1, t_last_vis - 5), t_last_vis + 1):
            if not occ_b[i, tt] and not occ_b[i, tt - 1]:
                v = tracks_b[i, tt] - tracks_b[i, tt - 1]
                velocities.append(v)
        if len(velocities) < 2:
            jitter = 1.0
        else:
            velocities = np.array(velocities)
            jitter = float(np.mean(np.var(velocities, axis=0)))

        dist = float(np.linalg.norm(pos_last - q_pos))
        candidates.append({
            "idx": i,
            "pos_last": pos_last,
            "pos_reentry": pos_re,
            "dist_to_query": dist,
            "jitter": jitter,
        })

    if not K:
        K = 16

    # Sort by distance, take top K
    candidates.sort(key=lambda c: c["dist_to_query"])
    selected = candidates[:K]

    # Compute weights
    for c in selected:
        spatial_w = math.exp(-c["dist_to_query"] ** 2 / (2 * 0.15 ** 2))  # sigma_d ~ 0.15 normalized
        stability_w = math.exp(-c["jitter"] / (2 * 1e-4 ** 2))  # sigma_j ~ 1e-4
        c["weight"] = spatial_w * stability_w

    return {
        "n_valid": len(selected),
        "n_total_candidates": len(candidates),
        "escorts": selected,
    }


# ---------------------------------------------------------------------------
# Motion prior centers
# ---------------------------------------------------------------------------
def compute_velocity_center(
    tracks_b: np.ndarray,  # (N, T, 2)
    occ_b: np.ndarray,     # (N, T) bool
    n_idx: int,
    t_last_vis: int,
    t_reentry: int,
    k_lookback: int = 3,
) -> np.ndarray:
    """Constant-velocity extrapolation from last k visible frames."""
    # Collect velocities from visible frame pairs before occlusion
    velocities = []
    for t in range(t_last_vis, max(0, t_last_vis - k_lookback), -1):
        if t > 0 and not occ_b[n_idx, t] and not occ_b[n_idx, t - 1]:
            v = tracks_b[n_idx, t] - tracks_b[n_idx, t - 1]
            velocities.append(v)
            if len(velocities) >= k_lookback:
                break

    if not velocities:
        # Fallback: no velocity info, return last visible position
        return tracks_b[n_idx, t_last_vis].copy()

    mean_v = np.mean(velocities, axis=0)
    dt = t_reentry - t_last_vis
    return tracks_b[n_idx, t_last_vis] + mean_v * dt


def compute_translation_center(
    tracks_b: np.ndarray,
    escort_data: dict,
    n_idx: int,
    t_last_vis: int,
    t_reentry: int,
) -> tuple:
    """Mean displacement of escort points applied to query position.

    Returns (center_norm, valid).
    """
    escorts = escort_data["escorts"]
    if len(escorts) < 2:
        return tracks_b[n_idx, t_last_vis].copy(), False

    displacements = []
    weights = []
    for e in escorts:
        d = e["pos_reentry"] - e["pos_last"]
        displacements.append(d)
        weights.append(e["weight"])

    displacements = np.array(displacements)  # (K, 2)
    weights = np.array(weights)
    wsum = weights.sum()
    if wsum < 1e-8:
        weights = np.ones(len(weights))
    else:
        weights = weights / wsum

    mean_d = np.average(displacements, weights=weights, axis=0)
    return tracks_b[n_idx, t_last_vis] + mean_d, True


def compute_affine_center(
    tracks_b: np.ndarray,
    escort_data: dict,
    n_idx: int,
    t_last_vis: int,
    t_reentry: int,
    residual_threshold: float = 0.05,
) -> tuple:
    """Weighted similarity transform from escort displacements.

    Fits: p_re = s * R * p_last + t  (similarity: rotation + uniform scale + translation)

    Returns (center_norm, valid, fallback_to_translation, residual).
    """
    escorts = escort_data["escorts"]
    if len(escorts) < 4:
        # Not enough for similarity fit, fallback
        center, _ = compute_translation_center(tracks_b, escort_data, n_idx, t_last_vis, t_reentry)
        return center, len(escorts) >= 2, True, -1.0

    # Build weighted system for similarity transform:
    # [p_re_x]   [a  -b  tx] [p_last_x]
    # [p_re_y] = [b   a  ty] [p_last_y]
    # [1     ]              [1        ]
    # Unknowns: [a, b, tx, ty]
    src = np.array([e["pos_last"] for e in escorts])      # (K, 2)
    dst = np.array([e["pos_reentry"] for e in escorts])    # (K, 2)
    w = np.array([e["weight"] for e in escorts])
    wsum = w.sum()
    if wsum < 1e-8:
        w = np.ones(len(w))
    else:
        w = w / wsum

    # Build A @ x = b for weighted least squares
    # x = [a, b, tx, ty]
    # For each point: [px, -py, 1, 0] @ [a, b, tx, ty]^T = qx
    #                 [py,  px, 0, 1] @ [a, b, tx, ty]^T = qy
    K = len(escorts)
    A = np.zeros((2 * K, 4))
    b_vec = np.zeros(2 * K)
    W_diag = np.zeros(2 * K)

    for i, e in enumerate(escorts):
        px, py = e["pos_last"]
        qx, qy = e["pos_reentry"]
        A[2*i]     = [px, -py, 1, 0]
        A[2*i + 1] = [py,  px, 0, 1]
        b_vec[2*i]     = qx
        b_vec[2*i + 1] = qy
        W_diag[2*i]     = e["weight"]
        W_diag[2*i + 1] = e["weight"]

    # Weighted least squares: (A^T W A) x = A^T W b
    W = np.diag(W_diag)
    AtWA = A.T @ W @ A
    AtWb = A.T @ W @ b_vec

    try:
        x = np.linalg.solve(AtWA, AtWb)
    except np.linalg.LinAlgError:
        center, _ = compute_translation_center(tracks_b, escort_data, n_idx, t_last_vis, t_reentry)
        return center, True, True, -1.0

    a, b_val, tx, ty = x

    # Compute residual
    pred = A @ x
    residual = float(np.sqrt(np.average((pred - b_vec) ** 2, weights=W_diag)))

    # Apply to query point
    qx_query, qy_query = tracks_b[n_idx, t_last_vis]
    pred_x = a * qx_query - b_val * qy_query + tx
    pred_y = b_val * qx_query + a * qy_query + ty
    center = np.array([pred_x, pred_y])

    # Check if fit is reasonable (scale should be near 1)
    scale = math.sqrt(a ** 2 + b_val ** 2)
    fallback = (residual > residual_threshold) or (scale < 0.5) or (scale > 2.0)

    if fallback:
        center, valid = compute_translation_center(tracks_b, escort_data, n_idx, t_last_vis, t_reentry)
        return center, valid, True, residual

    return center, True, False, residual


# ---------------------------------------------------------------------------
# Local DINO cosine search (optimized: uses precomputed full-frame features)
# ---------------------------------------------------------------------------
def precompute_frame_dino_features(
    dino_extractor,
    frame_uint8: np.ndarray,  # (H, W, 3) uint8
    device: torch.device,
) -> torch.Tensor:
    """Precompute DINO features for a full frame. Returns (D, h, w) on CPU."""
    return dino_extractor.feature_map(
        torch.from_numpy(frame_uint8).to(device), device
    )


def local_dino_cosine_search_from_features(
    re_feat: torch.Tensor,         # (D, Hf, Wf) precomputed reentry features
    sf_feat: torch.Tensor,         # (D, Hf, Wf) precomputed support features
    query_pos_norm: np.ndarray,    # (2,) normalized at t_last_vis
    center_norm: np.ndarray,       # (2,) prior center normalized
    orig_w: int,
    orig_h: int,
    crop_radius_px: int,
    topk: int = 5,
) -> dict:
    """Local DINO cosine search using precomputed feature maps.

    Masks the score map to only consider features within the crop region.
    Returns dict with topk positions (normalized), scores, and best position.
    """
    _, Hf, Wf = re_feat.shape

    # Map crop center/radius to feature grid coords
    feat_scale_x = Wf / float(orig_w)
    feat_scale_y = Hf / float(orig_h)

    center_fx = center_norm[0] * Wf  # feature grid x
    center_fy = center_norm[1] * Hf  # feature grid y
    radius_fx = max(1, int(crop_radius_px * feat_scale_x))
    radius_fy = max(1, int(crop_radius_px * feat_scale_y))

    # Query template position in feature grid
    qfx = int(np.clip(round(query_pos_norm[0] * Wf), 0, Wf - 1))
    qfy = int(np.clip(round(query_pos_norm[1] * Hf), 0, Hf - 1))

    # Extract multi-scale templates from support features at query position
    templates = []
    for radius in [1, 2]:
        y0 = max(0, qfy - radius)
        y1 = min(Hf, qfy + radius + 1)
        x0 = max(0, qfx - radius)
        x1 = min(Wf, qfx + radius + 1)
        tmpl = sf_feat[:, y0:y1, x0:x1]
        pad_l = max(0, radius - qfx)
        pad_r = max(0, qfx + radius + 1 - Wf)
        pad_t = max(0, radius - qfy)
        pad_b = max(0, qfy + radius + 1 - Hf)
        if pad_l or pad_r or pad_t or pad_b:
            tmpl = F.pad(tmpl, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
        templates.append(tmpl)

    # Full score map via template matching
    score_maps = []
    for tmpl in templates:
        c, th, tw = tmpl.shape
        search = F.pad(re_feat.unsqueeze(0), (tw // 2, tw // 2, th // 2, th // 2), mode="replicate")
        patches = F.unfold(search, kernel_size=(th, tw))
        patches = patches.transpose(1, 2)
        patches = F.normalize(patches, dim=-1)
        template_vec = F.normalize(tmpl.reshape(1, -1), dim=-1)
        scores = torch.matmul(patches, template_vec.t()).reshape(Hf, Wf)
        score_maps.append(scores)

    sims = torch.stack(score_maps, dim=0).mean(dim=0)  # (Hf, Wf)

    # Mask to crop region
    mask = torch.zeros_like(sims, dtype=torch.bool)
    y0m = max(0, int(center_fy - radius_fy))
    y1m = min(Hf, int(center_fy + radius_fy + 1))
    x0m = max(0, int(center_fx - radius_fx))
    x1m = min(Wf, int(center_fx + radius_fx + 1))
    mask[y0m:y1m, x0m:x1m] = True
    sims_masked = sims.masked_fill(~mask, -1e9)

    # Top-k within masked region
    k = min(topk, sims_masked.numel())
    top_vals, top_idx = torch.topk(sims_masked.reshape(-1), k=k)

    # Convert feature indices to global normalized coords
    results = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        fy = idx // Wf
        fx = idx % Wf
        global_norm = np.array([(fx + 0.5) / Wf, (fy + 0.5) / Hf], dtype=np.float32)
        results.append({
            "pos_norm": global_norm.tolist(),
            "score": float(val),
        })

    return {
        "topk": results,
        "best_norm": np.array(results[0]["pos_norm"]) if results else center_norm,
        "best_score": results[0]["score"] if results else 0.0,
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def safe_median(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) > 0 else float("nan")


def safe_mean(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) > 0 else float("nan")


def safe_p90(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 90)) if len(arr) > 0 else float("nan")


def compute_group_stats(samples, group_key, group_val, prior_name, radii):
    """Compute stats for one prior across one group."""
    filtered = [s for s in samples if s.get(group_key) == group_val and s.get(f"prior_valid_{prior_name}", False)]
    n = len(filtered)
    if n == 0:
        return {"n": 0, "valid_prior_frac": 0.0}

    center_errors = [s[f"{prior_name}_center_error_px"] for s in filtered]
    result = {
        "n": n,
        "valid_prior_frac": round(n / max(1, len([s for s in samples if s.get(group_key) == group_val])), 3),
        "center_error_median": round(safe_median(center_errors), 2),
        "center_error_mean": round(safe_mean(center_errors), 2),
        "center_error_p90": round(safe_p90(center_errors), 2),
    }

    for r in radii:
        gt_in = [s.get(f"gt_in_crop_{prior_name}_r{r}", False) for s in filtered]
        result[f"GT_in_crop@{r}"] = round(float(np.mean(gt_in)), 3) if gt_in else 0.0

        oracle_errs = [s.get(f"oracle_best_error_{prior_name}_r{r}", float("inf")) for s in filtered]
        valid_oracle = [e for e in oracle_errs if np.isfinite(e) and e < 999]
        result[f"oracle_best_error_median@{r}"] = round(safe_median(valid_oracle), 2) if valid_oracle else None

        base_errs = [s["base_error_px"] for s in filtered]
        better_2px = [1 for oe, be in zip(oracle_errs, base_errs) if np.isfinite(oe) and oe < be - 2]
        result[f"oracle_better_frac@2px@{r}"] = round(len(better_2px) / max(1, n), 3)

    if prior_name in ("translation", "affine"):
        fallback = [s.get(f"affine_fallback_to_translation", False) for s in filtered] if prior_name == "affine" else []
        result["fallback_rate"] = round(float(np.mean(fallback)), 3) if fallback else 0.0
        insuff = [s.get("insufficient_neighbors", False) for s in filtered]
        result["insufficient_neighbors_frac"] = round(float(np.mean(insuff)), 3) if insuff else 0.0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Local Motion Prior Recovery Audit (Stage-0)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--escort-k", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--radii", type=str, default="32,64,96")
    args = parser.parse_args()

    radii = [int(r) for r in args.radii.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # Load model
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model._export_candidates_debug = True
    model = model.to(device).eval()

    # Load DINO extractor
    from models.recovery_features import DINORecoveryExtractor
    dino_extractor = DINORecoveryExtractor()
    dino_extractor._ensure_loaded(device)

    # Val loader
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    dataloader = _build_val_loader_from_config(cfg)

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    print(f"Starting local motion prior audit: max_batches={args.max_batches}, escort_k={args.escort_k}, topk={args.topk}, radii={radii}")
    import time as _time

    for batch_idx, batch in enumerate(dataloader):
        batch_start = _time.time()
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video = batch.get("video")
        query_points = batch.get("query_points")
        target_points = batch.get("target_points", batch.get("tracks"))
        occluded = batch.get("occluded")
        video_name = batch.get("video_name", ["unknown"])
        if video is None or query_points is None or target_points is None:
            continue

        video_dev = video.to(device)
        query_dev = query_points.to(device)
        if video_dev.dim() == 4: video_dev = video_dev.unsqueeze(0)
        if query_dev.dim() == 2: query_dev = query_dev.unsqueeze(0)
        if target_points.dim() == 3: target_points = target_points.unsqueeze(0)
        occ_np = None
        if occluded is not None:
            if occluded.dim() == 2: occluded = occluded.unsqueeze(0)
            occ_np = occluded.cpu().numpy()

        B, T, C, H, W = video_dev.shape
        N_pts = query_dev.shape[1]
        meta = {"video_name": video_name, "base_tracks": batch.get("base_tracks"), "base_visibility": batch.get("base_visibility")}

        with torch.no_grad():
            try:
                t_model_start = _time.time()
                # Set 90-second timeout for model forward
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(90)
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                t_model = _time.time() - t_model_start
                print(f"  Batch {batch_idx}: model forward {t_model:.1f}s")
                sys.stdout.flush()
            except _TimeoutError:
                signal.alarm(0)
                print(f"  Batch {batch_idx}: model forward TIMEOUT (>90s), skipping")
                sys.stdout.flush()
                continue
            except Exception as e:
                signal.alarm(0)
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}
        relocal_mask = info.get("relocal_mask")
        if not isinstance(relocal_mask, torch.Tensor):
            continue

        base_tracks = info.get("base_tracks", batch.get("base_tracks"))
        if base_tracks is not None: base_tracks = base_tracks.cpu().numpy()
        else: base_tracks = outputs[0].cpu().numpy()
        gt_tracks = target_points.cpu().numpy()
        tracks_all = outputs[0].cpu().numpy()  # (B, N, T, 2)

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        # Convert video frames to uint8 numpy for DINO (per-frame lazy conversion)
        # video_dev is (B, T, C, H, W) float32 [0,1]
        video_np = (video_dev[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (T, C, H, W)
        video_np = video_np.transpose(0, 2, 3, 1)  # (T, H, W, C) uint8

        # Precompute DINO features for all frames in this batch (lazy, cached)
        dino_cache = {}  # t -> (D, Hf, Wf) tensor on CPU
        import time as _time

        def get_dino_features(t_idx):
            if t_idx not in dino_cache:
                t_start = _time.time()
                dino_cache[t_idx] = precompute_frame_dino_features(
                    dino_extractor, video_np[t_idx], device
                )
                t_elapsed = _time.time() - t_start
            return dino_cache[t_idx]

        for b in range(B):
            n_triggered_total = 0
            for n_idx in range(N_pts):
                rm = relocal_mask[b, n_idx].cpu().numpy()
                triggered = np.where(rm)[0]
                if len(triggered) == 0:
                    continue

                for t0 in triggered:
                    t0 = int(t0)
                    # Occlusion filter
                    if occ_np is not None and occ_np[b, n_idx, t0]:
                        continue

                    # Find t_last_visible
                    t_last_vis = find_t_last_visible(occ_np[b, n_idx], t0) if occ_np is not None else max(0, t0 - 1)
                    occ_length = t0 - t_last_vis - 1

                    # GT validity check
                    gt_norm = gt_tracks[b, n_idx, t0]
                    if np.linalg.norm(gt_norm) < 1e-6:
                        continue

                    base_norm = base_tracks[b, n_idx, t0]
                    base_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h])
                    gt_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h])
                    base_err = float(np.linalg.norm(base_px - gt_px))

                    # Escort selection
                    escort_data = select_escort_points(
                        tracks_all[b], occ_np[b] if occ_np is not None else np.zeros((N_pts, T), dtype=bool),
                        n_idx, t_last_vis, t0, K=args.escort_k,
                    )

                    # Prior centers
                    base_center = base_norm.copy()
                    velocity_center = compute_velocity_center(tracks_all[b], occ_np[b] if occ_np is not None else np.zeros((N_pts, T), dtype=bool), n_idx, t_last_vis, t0)
                    translation_center, translation_valid = compute_translation_center(tracks_all[b], escort_data, n_idx, t_last_vis, t0)
                    affine_center, affine_valid, affine_fallback, affine_residual = compute_affine_center(tracks_all[b], escort_data, n_idx, t_last_vis, t0)

                    insufficient = escort_data["n_valid"] < 4

                    # Build sample record
                    sample = {
                        "video_name": vid_name,
                        "point_idx": int(n_idx),
                        "t_query": int(t0),
                        "t_last_visible": int(t_last_vis),
                        "t_reentry": int(t0),
                        "occ_length": int(occ_length),
                        "base_error_px": round(base_err, 2),
                        "group_all": True,
                        "group_base16": bool(base_err > 16),
                        "group_base32": bool(base_err > 32),
                        "num_total_points": int(N_pts),
                        "num_valid_escorts": escort_data["n_valid"],
                        "escort_k_requested": args.escort_k,
                        "escort_k_used": min(args.escort_k, escort_data["n_valid"]),
                        "prior_valid_base": True,
                        "prior_valid_velocity": True,  # velocity always computable (may degrade to last-pos)
                        "prior_valid_translation": translation_valid,
                        "prior_valid_affine": affine_valid and not insufficient,
                        "affine_fallback_to_translation": affine_fallback,
                        "insufficient_neighbors": insufficient,
                    }

                    # Center errors
                    centers = {
                        "base": base_center,
                        "velocity": velocity_center,
                        "translation": translation_center,
                        "affine": affine_center,
                    }
                    for name, center in centers.items():
                        center_px = np.array([center[0] * orig_w, center[1] * orig_h])
                        sample[f"{name}_center_error_px"] = round(float(np.linalg.norm(center_px - gt_px)), 2)

                    # Get precomputed DINO features for reentry and support frames
                    re_feat = get_dino_features(t0)
                    sf_feat = get_dino_features(t_last_vis)
                    query_pos_at_last = tracks_all[b, n_idx, t_last_vis]  # normalized

                    # Local DINO search for each prior center and radius
                    for name, center in centers.items():
                        center_norm = center.copy()
                        # Clamp to valid range
                        center_norm[0] = np.clip(center_norm[0], 0.01, 0.99)
                        center_norm[1] = np.clip(center_norm[1], 0.01, 0.99)

                        for r in radii:
                            gt_in_crop = False
                            oracle_err = float("inf")

                            try:
                                # Check if GT is within crop (quick check before DINO)
                                center_px_crop = np.array([center_norm[0] * orig_w, center_norm[1] * orig_h])
                                gt_dist = np.linalg.norm(gt_px - center_px_crop)
                                gt_in_crop = gt_dist < r

                                if gt_in_crop:
                                    # Only do DINO search if GT is in crop (otherwise oracle can't improve)
                                    search_result = local_dino_cosine_search_from_features(
                                        re_feat, sf_feat,
                                        query_pos_at_last, center_norm,
                                        orig_w, orig_h, r, topk=args.topk,
                                    )
                                    # Oracle: best of topk
                                    for tk in search_result["topk"]:
                                        tk_norm = np.array(tk["pos_norm"])
                                        tk_px = np.array([tk_norm[0] * orig_w, tk_norm[1] * orig_h])
                                        err = float(np.linalg.norm(tk_px - gt_px))
                                        if err < oracle_err:
                                            oracle_err = err

                                    # Store top-1 for debugging
                                    if search_result["topk"]:
                                        sample[f"top1_{name}_r{r}_norm"] = search_result["topk"][0]["pos_norm"]
                                        sample[f"top1_{name}_r{r}_score"] = round(search_result["topk"][0]["score"], 4)
                                else:
                                    oracle_err = float("inf")

                            except Exception as e:
                                oracle_err = float("inf")

                            sample[f"gt_in_crop_{name}_r{r}"] = bool(gt_in_crop)
                            sample[f"oracle_best_error_{name}_r{r}"] = round(oracle_err, 2) if np.isfinite(oracle_err) else 999.0
                            sample[f"oracle_better_by_2px_{name}_r{r}"] = bool(np.isfinite(oracle_err) and oracle_err < base_err - 2)

                    samples.append(sample)
                    n_triggered_total += 1

        n_so_far = len(samples)
        batch_elapsed = _time.time() - batch_start
        print(f"  Batch {batch_idx+1}: {n_so_far} triggered+visible samples ({batch_elapsed:.1f}s, dino_cache={len(dino_cache)} frames)")
        sys.stdout.flush()

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    n = len(samples)
    print(f"\nTotal: {n} triggered+visible samples")
    if n == 0:
        print("No samples! Check relocal_mask and occlusion filter.")
        return

    # per_sample.jsonl
    per_sample_path = out_dir / "per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # per_sequence.json
    vid_names = sorted(set(s["video_name"] for s in samples))
    per_sequence = {}
    for vid in vid_names:
        vs = [s for s in samples if s["video_name"] == vid]
        per_sequence[vid] = {
            "n": len(vs),
            "n_base16": sum(1 for s in vs if s["group_base16"]),
            "n_base32": sum(1 for s in vs if s["group_base32"]),
            "base_error_median": round(safe_median([s["base_error_px"] for s in vs]), 2),
            "affine_valid_frac": round(float(np.mean([s["prior_valid_affine"] for s in vs])), 3),
            "escort_median": round(float(np.median([s["num_valid_escorts"] for s in vs])), 1),
        }
    with open(out_dir / "per_sequence.json", "w") as f:
        json.dump(per_sequence, f, indent=2)

    # summary.json
    priors = ["base", "velocity", "translation", "affine"]
    groups = [("group_all", True), ("group_base16", True), ("group_base32", True)]

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "max_batches": args.max_batches,
        "escort_k": args.escort_k,
        "topk": args.topk,
        "radii": radii,
        "n_triggered_visible": n,
    }

    for group_key, _ in groups:
        group_name = group_key.replace("group_", "")
        group_samples = [s for s in samples if s.get(group_key)]
        summary[group_name] = {"n": len(group_samples)}

        for prior in priors:
            stats = compute_group_stats(samples, group_key, True, prior, radii)
            summary[group_name][prior] = stats

        # Deltas: affine vs base
        base_stats = summary[group_name].get("base", {})
        for other in ["velocity", "translation", "affine"]:
            other_stats = summary[group_name].get(other, {})
            if base_stats.get("center_error_median") and other_stats.get("center_error_median"):
                delta = other_stats["center_error_median"] - base_stats["center_error_median"]
                pct = delta / base_stats["center_error_median"] * 100
                summary[group_name][f"{other}_vs_base_center_error_delta_pct"] = round(pct, 1)
            for r in radii:
                base_gt = base_stats.get(f"GT_in_crop@{r}", 0)
                other_gt = other_stats.get(f"GT_in_crop@{r}", 0)
                summary[group_name][f"{other}_vs_base_gt_in_crop{r}_delta"] = round(other_gt - base_gt, 3)
                base_ob = base_stats.get(f"oracle_better_frac@2px@{r}", 0)
                other_ob = other_stats.get(f"oracle_better_frac@2px@{r}", 0)
                summary[group_name][f"{other}_vs_base_oracle_better2px{r}_delta"] = round(other_ob - base_ob, 3)

    # Escort summary
    all_escorts = [s["num_valid_escorts"] for s in samples]
    summary["escort_stats"] = {
        "median": round(float(np.median(all_escorts)), 1),
        "mean": round(float(np.mean(all_escorts)), 1),
        "min": int(np.min(all_escorts)),
        "max": int(np.max(all_escorts)),
        "frac_with_lt_4": round(float(np.mean(np.array(all_escorts) < 4)), 3),
    }

    # Stage-0 pass/fail
    b16 = summary.get("base16", {}).get("affine", {})
    base_b16 = summary.get("base16", {}).get("base", {})
    criteria_met = 0
    criterion_details = []

    # Criterion 1: center_error_median improvement >= 20%
    if base_b16.get("center_error_median") and b16.get("center_error_median"):
        imp = (base_b16["center_error_median"] - b16["center_error_median"]) / base_b16["center_error_median"]
        c1 = imp >= 0.20
        criteria_met += int(c1)
        criterion_details.append(f"center_error improvement {imp:.1%} >= 20%: {'PASS' if c1 else 'FAIL'}")

    # Criterion 2: GT_in_crop@64 improvement >= +0.15
    base_gt64 = base_b16.get("GT_in_crop@64", 0)
    affine_gt64 = b16.get("GT_in_crop@64", 0)
    c2 = (affine_gt64 - base_gt64) >= 0.15
    criteria_met += int(c2)
    criterion_details.append(f"GT_in_crop@64 delta {affine_gt64 - base_gt64:.3f} >= 0.15: {'PASS' if c2 else 'FAIL'}")

    # Criterion 3: oracle_better_frac@2px@64 >= 0.25
    ob64 = b16.get("oracle_better_frac@2px@64", 0)
    c3 = ob64 >= 0.25
    criteria_met += int(c3)
    criterion_details.append(f"oracle_better@2px@64 {ob64:.3f} >= 0.25: {'PASS' if c3 else 'FAIL'}")

    # Criterion 4 (preferred): oracle_better_frac@2px@64 >= 0.35 on base>32
    b32 = summary.get("base32", {}).get("affine", {})
    ob64_32 = b32.get("oracle_better_frac@2px@64", 0)
    c4 = ob64_32 >= 0.35

    stage0_pass = criteria_met >= 2
    summary["stage0_verdict"] = "PASS" if stage0_pass else "FAIL"
    summary["stage0_criteria_met"] = criteria_met
    summary["stage0_criteria_details"] = criterion_details
    summary["stage0_criterion4_base32_oracle35"] = f"{'PASS' if c4 else 'FAIL'}: {ob64_32:.3f}"

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Local Motion Prior Recovery Audit — Stage-0 Results")
    print(f"{'='*70}")
    print(f"n={n} triggered+visible samples, escort_k={args.escort_k}")
    print(f"escort valid median: {summary['escort_stats']['median']}, frac<4: {summary['escort_stats']['frac_with_lt_4']:.3f}")
    print()

    for group_name in ["all", "base16", "base32"]:
        g = summary.get(group_name, {})
        if g.get("n", 0) == 0:
            print(f"  {group_name}: n=0")
            continue
        print(f"  {group_name} (n={g['n']}):")
        for prior in priors:
            p = g.get(prior, {})
            if p.get("n", 0) == 0:
                continue
            print(f"    {prior:12s} n={p['n']:>4d}: center_err_med={p.get('center_error_median','?'):>7}, "
                  f"GT@64={p.get('GT_in_crop@64', '?'):.3f}, "
                  f"oracle@2px@64={p.get('oracle_better_frac@2px@64', '?'):.3f}")

    print(f"\n  Stage-0 verdict: {summary['stage0_verdict']} ({criteria_met}/3 criteria met)")
    for d in criterion_details:
        print(f"    {d}")
    print(f"    (preferred) base>32 oracle@2px@64>=0.35: {summary['stage0_criterion4_base32_oracle35']}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
