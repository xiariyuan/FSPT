#!/usr/bin/env python3
"""
Base-Centered Local Recovery Audit

Tests whether base-centered local DINO cosine search produces candidates
that can directly improve recovery after long occlusion.

Reports 4 channels:
  1. base — current tracker position
  2. raw top1 — best DINO match in crop (unfiltered)
  3. oracle top5 — best of top-5 DINO matches
  4. selective top1 — top1 accepted/rejected by threshold rules

Usage:
  python scripts/audit_base_centered_local_recovery.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --output-dir outputs/base_centered_local_recovery_audit \
    --max-batches 30 --topk 5
"""

from __future__ import annotations
import argparse, json, sys, math, os, signal
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config loader
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
# Timeout helper
# ---------------------------------------------------------------------------
class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError("Model forward timed out")


# ---------------------------------------------------------------------------
# t_last_visible extraction
# ---------------------------------------------------------------------------
def find_t_last_visible(occ_mask_n: np.ndarray, t_reentry: int) -> int:
    for t in range(t_reentry - 1, -1, -1):
        if not occ_mask_n[t]:
            return t
    return 0


# ---------------------------------------------------------------------------
# DINO feature precompute
# ---------------------------------------------------------------------------
def precompute_frame_dino_features(dino_extractor, frame_uint8, device):
    return dino_extractor.feature_map(torch.from_numpy(frame_uint8).to(device), device)


# ---------------------------------------------------------------------------
# Local DINO search from precomputed features
# ---------------------------------------------------------------------------
def local_search(re_feat, sf_feat, query_pos_norm, center_norm, orig_w, orig_h, crop_radius_px, topk=5):
    _, Hf, Wf = re_feat.shape
    center_fx = center_norm[0] * Wf
    center_fy = center_norm[1] * Hf
    radius_fx = max(1, int(crop_radius_px * (Wf / float(orig_w))))
    radius_fy = max(1, int(crop_radius_px * (Hf / float(orig_h))))

    qfx = int(np.clip(round(query_pos_norm[0] * Wf), 0, Wf - 1))
    qfy = int(np.clip(round(query_pos_norm[1] * Hf), 0, Hf - 1))

    templates = []
    for radius in [1, 2]:
        y0, y1 = max(0, qfy - radius), min(Hf, qfy + radius + 1)
        x0, x1 = max(0, qfx - radius), min(Wf, qfx + radius + 1)
        tmpl = sf_feat[:, y0:y1, x0:x1]
        pl, pr = max(0, radius - qfx), max(0, qfx + radius + 1 - Wf)
        pt, pb = max(0, radius - qfy), max(0, qfy + radius + 1 - Hf)
        if pl or pr or pt or pb:
            tmpl = F.pad(tmpl, (pl, pr, pt, pb), mode="replicate")
        templates.append(tmpl)

    score_maps = []
    for tmpl in templates:
        c, th, tw = tmpl.shape
        search = F.pad(re_feat.unsqueeze(0), (tw//2, tw//2, th//2, th//2), mode="replicate")
        patches = F.unfold(search, kernel_size=(th, tw)).transpose(1, 2)
        patches = F.normalize(patches, dim=-1)
        template_vec = F.normalize(tmpl.reshape(1, -1), dim=-1)
        scores = torch.matmul(patches, template_vec.t()).reshape(Hf, Wf)
        score_maps.append(scores)

    sims = torch.stack(score_maps, dim=0).mean(dim=0)

    # Mask to crop region
    mask = torch.zeros_like(sims, dtype=torch.bool)
    y0m = max(0, int(center_fy - radius_fy))
    y1m = min(Hf, int(center_fy + radius_fy + 1))
    x0m = max(0, int(center_fx - radius_fx))
    x1m = min(Wf, int(center_fx + radius_fx + 1))
    mask[y0m:y1m, x0m:x1m] = True
    sims_masked = sims.masked_fill(~mask, -1e9)

    k = min(topk, sims_masked.numel())
    top_vals, top_idx = torch.topk(sims_masked.reshape(-1), k=k)

    results = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        fy, fx = divmod(idx, Wf)
        gn = np.array([(fx + 0.5) / Wf, (fy + 0.5) / Hf], dtype=np.float32)
        results.append({"pos_norm": gn.tolist(), "score": float(val)})

    return results


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def sweep_thresholds(samples, radii):
    """Find best operating points for selective top1."""
    results = {}
    for r in radii:
        r_results = {}
        score_key = f"top1_score_r{r}"
        margin_key = f"top1_margin_r{r}"
        shift_key = f"top1_shift_px_r{r}"
        err_key = f"top1_error_r{r}"

        scores = np.array([s.get(score_key, 0) for s in samples])
        margins = np.array([s.get(margin_key, 0) for s in samples])
        shifts = np.array([s.get(shift_key, 0) for s in samples])
        top1_errs = np.array([s.get(err_key, 999) for s in samples])
        base_errs = np.array([s["base_error_px"] for s in samples])

        base16 = np.array([s.get("group_base16", False) for s in samples], dtype=bool)
        base32 = np.array([s.get("group_base32", False) for s in samples], dtype=bool)

        configs = []

        # Score threshold sweep
        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            accept = scores >= thr
            configs.append({"name": f"score>={thr}", "mask": accept, "type": "score", "threshold": thr})

        # Margin threshold sweep
        for thr in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            accept = margins >= thr
            configs.append({"name": f"margin>={thr}", "mask": accept, "type": "margin", "threshold": thr})

        # Shift threshold sweep (accept if shift is large enough = top1 actually moves)
        for thr in [2, 4, 6, 8, 10, 16]:
            accept = shifts >= thr
            configs.append({"name": f"shift>={thr}", "mask": accept, "type": "shift", "threshold": thr})

        # Combined: score AND margin
        for st in [0.2, 0.3, 0.5]:
            for mt in [0.02, 0.05, 0.1]:
                accept = (scores >= st) & (margins >= mt)
                configs.append({"name": f"score>={st}_margin>={mt}", "mask": accept, "type": "combo", "score_thr": st, "margin_thr": mt})

        for cfg in configs:
            mask = cfg["mask"]
            n_accept = int(mask.sum())
            if n_accept == 0:
                continue

            # Overall
            accept_top1_errs = top1_errs[mask]
            accept_base_errs = base_errs[mask]
            better = (accept_top1_errs < accept_base_errs).sum()
            better_2px = (accept_top1_errs < accept_base_errs - 2).sum()

            # base>16
            accept_b16 = mask & base16
            n_b16 = int(accept_b16.sum())
            b16_better_2px = 0
            b16_median_improve = 0
            if n_b16 > 0:
                b16_top1 = top1_errs[accept_b16]
                b16_base = base_errs[accept_b16]
                b16_better_2px = int((b16_top1 < b16_base - 2).sum())
                b16_median_improve = float(np.median(b16_base) - np.median(b16_top1))

            entry = {
                "config": cfg["name"],
                "type": cfg["type"],
                "n_accept": n_accept,
                "coverage": round(n_accept / len(samples), 3),
                "overall_median_base": round(float(np.median(accept_base_errs)), 2),
                "overall_median_top1": round(float(np.median(accept_top1_errs)), 2),
                "overall_better_frac": round(float(better / n_accept), 3),
                "overall_better_2px_frac": round(float(better_2px / n_accept), 3),
                "base16_n_accept": n_b16,
                "base16_better_2px_frac": round(b16_better_2px / max(1, n_b16), 3),
                "base16_median_improve": round(b16_median_improve, 2),
            }
            r_results[cfg["name"]] = entry

        results[f"r{r}"] = r_results
    return results


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def safe_med(arr):
    a = np.array(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    return round(float(np.median(a)), 2) if len(a) else None

def safe_mean(arr):
    a = np.array(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    return round(float(np.mean(a)), 2) if len(a) else None

def safe_p90(arr):
    a = np.array(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    return round(float(np.percentile(a, 90)), 2) if len(a) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
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

    # DINO extractor
    from models.recovery_features import DINORecoveryExtractor
    dino_extractor = DINORecoveryExtractor()
    dino_extractor._ensure_loaded(device)

    # Val loader
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    dataloader = _build_val_loader_from_config(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import time as _time
    samples = []

    print(f"Base-centered local recovery audit: max_batches={args.max_batches}, topk={args.topk}, radii={radii}")

    for batch_idx, batch in enumerate(dataloader):
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
                t_m = _time.time()
                old_h = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(90)
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_h)
                print(f"  Batch {batch_idx}: model {_time.time()-t_m:.1f}s")
                sys.stdout.flush()
            except _TimeoutError:
                signal.alarm(0)
                print(f"  Batch {batch_idx}: TIMEOUT, skip")
                continue
            except Exception as e:
                signal.alarm(0)
                print(f"  Batch {batch_idx}: ERROR {e}")
                continue

        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}
        relocal_mask = info.get("relocal_mask")
        if not isinstance(relocal_mask, torch.Tensor):
            continue

        base_tracks = info.get("base_tracks", batch.get("base_tracks"))
        if base_tracks is not None: base_tracks = base_tracks.cpu().numpy()
        else: base_tracks = outputs[0].cpu().numpy()
        gt_tracks = target_points.cpu().numpy()

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        # Precompute DINO features
        video_np = (video_dev[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
        dino_cache = {}

        def get_dino(t):
            if t not in dino_cache:
                dino_cache[t] = precompute_frame_dino_features(dino_extractor, video_np[t], device)
            return dino_cache[t]

        for b in range(B):
            for n_idx in range(N_pts):
                rm = relocal_mask[b, n_idx].cpu().numpy()
                triggered = np.where(rm)[0]
                if len(triggered) == 0:
                    continue

                for t0 in triggered:
                    t0 = int(t0)
                    if occ_np is not None and occ_np[b, n_idx, t0]:
                        continue

                    t_last_vis = find_t_last_visible(occ_np[b, n_idx], t0) if occ_np is not None else max(0, t0 - 1)
                    occ_length = t0 - t_last_vis - 1

                    gt_norm = gt_tracks[b, n_idx, t0]
                    if np.linalg.norm(gt_norm) < 1e-6:
                        continue

                    base_norm = base_tracks[b, n_idx, t0]
                    base_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h])
                    gt_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h])
                    base_err = float(np.linalg.norm(base_px - gt_px))

                    re_feat = get_dino(t0)
                    sf_feat = get_dino(t_last_vis)
                    query_at_last = base_tracks[b, n_idx, t_last_vis]  # use base track at last visible

                    sample = {
                        "video_name": vid_name,
                        "point_idx": int(n_idx),
                        "t_reentry": int(t0),
                        "occ_length": int(occ_length),
                        "base_error_px": round(base_err, 2),
                        "group_all": True,
                        "group_base16": bool(base_err > 16),
                        "group_base32": bool(base_err > 32),
                    }

                    for r in radii:
                        center_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h])
                        gt_dist = float(np.linalg.norm(gt_px - center_px))
                        gt_in_crop = gt_dist < r
                        sample[f"gt_in_crop_r{r}"] = bool(gt_in_crop)

                        if not gt_in_crop:
                            for suffix in ["top1_error", "oracle_error", "top1_score", "top1_margin", "top1_shift_px", "top1_better_by_2px", "oracle_better_by_2px"]:
                                sample[f"{suffix}_r{r}"] = 999.0 if "error" in suffix else (False if "better" in suffix else 0.0)
                            continue

                        try:
                            results = local_search(re_feat, sf_feat, query_at_last, base_norm, orig_w, orig_h, r, topk=args.topk)

                            # Top1
                            top1_norm = np.array(results[0]["pos_norm"])
                            top1_px = np.array([top1_norm[0] * orig_w, top1_norm[1] * orig_h])
                            top1_err = float(np.linalg.norm(top1_px - gt_px))
                            top1_score = results[0]["score"]
                            top1_shift = float(np.linalg.norm(top1_px - base_px))

                            # Top1-top2 margin
                            top1_margin = top1_score - results[1]["score"] if len(results) > 1 else top1_score

                            # Oracle (best of topk)
                            oracle_err = top1_err
                            for tk in results[1:]:
                                tk_norm = np.array(tk["pos_norm"])
                                tk_px = np.array([tk_norm[0] * orig_w, tk_norm[1] * orig_h])
                                err = float(np.linalg.norm(tk_px - gt_px))
                                if err < oracle_err:
                                    oracle_err = err

                            sample[f"top1_error_r{r}"] = round(top1_err, 2)
                            sample[f"oracle_error_r{r}"] = round(oracle_err, 2)
                            sample[f"top1_score_r{r}"] = round(top1_score, 4)
                            sample[f"top1_margin_r{r}"] = round(top1_margin, 4)
                            sample[f"top1_shift_px_r{r}"] = round(top1_shift, 2)
                            sample[f"top1_better_by_2px_r{r}"] = bool(top1_err < base_err - 2)
                            sample[f"oracle_better_by_2px_r{r}"] = bool(oracle_err < base_err - 2)

                        except Exception:
                            for suffix in ["top1_error", "oracle_error", "top1_score", "top1_margin", "top1_shift_px", "top1_better_by_2px", "oracle_better_by_2px"]:
                                sample[f"{suffix}_r{r}"] = 999.0 if "error" in suffix else (False if "better" in suffix else 0.0)

                    samples.append(sample)

        n_so_far = len(samples)
        print(f"  Batch {batch_idx+1}: {n_so_far} samples, dino_cache={len(dino_cache)}")
        sys.stdout.flush()

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    n = len(samples)
    print(f"\nTotal: {n} samples")
    if n == 0:
        return

    # per_sample.jsonl
    with open(out_dir / "per_sample.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # per_sequence.json
    vid_names = sorted(set(s["video_name"] for s in samples))
    per_seq = {}
    for vid in vid_names:
        vs = [s for s in samples if s["video_name"] == vid]
        per_seq[vid] = {
            "n": len(vs),
            "n_base16": sum(1 for s in vs if s["group_base16"]),
            "base_error_median": safe_med([s["base_error_px"] for s in vs]),
        }
        for r in radii:
            per_seq[vid][f"top1_median_r{r}"] = safe_med([s.get(f"top1_error_r{r}", 999) for s in vs])
            per_seq[vid][f"oracle_median_r{r}"] = safe_med([s.get(f"oracle_error_r{r}", 999) for s in vs])
    with open(out_dir / "per_sequence.json", "w") as f:
        json.dump(per_seq, f, indent=2)

    # summary.json
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "max_batches": args.max_batches,
        "topk": args.topk,
        "radii": radii,
        "n_total": n,
    }

    for group_name, group_filter in [("all", lambda s: True), ("base16", lambda s: s["group_base16"]), ("base32", lambda s: s["group_base32"])]:
        gs = [s for s in samples if group_filter(s)]
        ng = len(gs)
        if ng == 0:
            summary[group_name] = {"n": 0}
            continue

        base_errs = [s["base_error_px"] for s in gs]
        g = {
            "n": ng,
            "base_median": safe_med(base_errs),
            "base_mean": safe_mean(base_errs),
            "base_p90": safe_p90(base_errs),
        }

        for r in radii:
            top1_errs = [s.get(f"top1_error_r{r}", 999) for s in gs]
            oracle_errs = [s.get(f"oracle_error_r{r}", 999) for s in gs]
            valid = [i for i, s in enumerate(gs) if s.get(f"gt_in_crop_r{r}", False)]
            gt_in = len(valid) / max(1, ng)

            top1_valid = [top1_errs[i] for i in valid if top1_errs[i] < 999]
            oracle_valid = [oracle_errs[i] for i in valid if oracle_errs[i] < 999]
            base_valid = [base_errs[i] for i in valid]

            top1_better = sum(1 for i in valid if top1_errs[i] < base_errs[i])
            top1_better_2px = sum(1 for i in valid if top1_errs[i] < base_errs[i] - 2)
            oracle_better_2px = sum(1 for i in valid if oracle_errs[i] < base_errs[i] - 2)

            shifts = [gs[i].get(f"top1_shift_px_r{r}", 0) for i in valid]
            margins = [gs[i].get(f"top1_margin_r{r}", 0) for i in valid]
            scores = [gs[i].get(f"top1_score_r{r}", 0) for i in valid]

            g[f"GT_in_crop_r{r}"] = round(gt_in, 3)
            g[f"top1_median_r{r}"] = safe_med(top1_valid)
            g[f"top1_mean_r{r}"] = safe_mean(top1_valid)
            g[f"oracle_median_r{r}"] = safe_med(oracle_valid)
            g[f"oracle_mean_r{r}"] = safe_mean(oracle_valid)
            g[f"top1_better_frac_r{r}"] = round(top1_better / max(1, len(valid)), 3)
            g[f"top1_better_2px_frac_r{r}"] = round(top1_better_2px / max(1, len(valid)), 3)
            g[f"oracle_better_2px_frac_r{r}"] = round(oracle_better_2px / max(1, len(valid)), 3)
            g[f"shift_median_r{r}"] = safe_med(shifts)
            g[f"margin_median_r{r}"] = safe_med(margins)
            g[f"score_median_r{r}"] = safe_med(scores)

        summary[group_name] = g

    # Threshold sweep
    summary["threshold_sweep"] = sweep_thresholds(samples, radii)

    # Stage-1 pass criteria
    b16 = summary.get("base16", {})
    criteria = []
    passed = 0

    # Criterion 1: raw top1 base>16 better_by_2px_frac >= 0.55
    for r in radii:
        val = b16.get(f"top1_better_2px_frac_r{r}", 0)
        if val >= 0.55:
            passed += 1
            criteria.append(f"raw top1 better_2px@r{r}={val:.3f} >= 0.55: PASS")
            break
    else:
        criteria.append(f"raw top1 better_2px (max over radii) < 0.55: FAIL")

    # Criterion 2: raw top1 base>16 median improvement >= 10%
    for r in radii:
        top1_med = b16.get(f"top1_median_r{r}")
        base_med = b16.get("base_median")
        if top1_med is not None and base_med is not None and base_med > 0:
            imp = (base_med - top1_med) / base_med
            if imp >= 0.10:
                passed += 1
                criteria.append(f"raw top1 median improvement@r{r}={imp:.1%} >= 10%: PASS")
                break
    else:
        criteria.append(f"raw top1 median improvement < 10%: FAIL")

    # Criterion 3: selective top1 exists with better_2px >= 0.60 and overall median not worse by > 0.5
    found_selective = False
    sweep = summary.get("threshold_sweep", {})
    for r in radii:
        r_sweep = sweep.get(f"r{r}", {})
        for name, entry in r_sweep.items():
            if entry.get("base16_better_2px_frac", 0) >= 0.60:
                overall_base_med = entry.get("overall_median_base", 0)
                overall_top1_med = entry.get("overall_median_top1", 0)
                if overall_top1_med <= overall_base_med + 0.5:
                    found_selective = True
                    passed += 1
                    criteria.append(f"selective '{name}'@r{r}: b16_better2px={entry['base16_better_2px_frac']:.3f}, overall_top1_med={overall_top1_med:.1f} vs base={overall_base_med:.1f}: PASS")
                    break
        if found_selective:
            break
    if not found_selective:
        criteria.append(f"selective top1: no operating point with b16_better_2px>=0.60 + overall stable: FAIL")

    # Criterion 4: base>32 raw or selective better_2px >= 0.60
    b32 = summary.get("base32", {})
    c4 = False
    for r in radii:
        if b32.get(f"top1_better_2px_frac_r{r}", 0) >= 0.60:
            c4 = True
            criteria.append(f"base>32 raw top1 better_2px@r{r}={b32[f'top1_better_2px_frac_r{r}']:.3f} >= 0.60: PASS")
            break
    if not c4:
        for r in radii:
            if b32.get(f"oracle_better_2px_frac_r{r}", 0) >= 0.60:
                criteria.append(f"base>32 oracle better_2px@r{r}={b32[f'oracle_better_2px_frac_r{r}']:.3f} >= 0.60 (oracle, not raw): NOTE")
                break

    stage1_pass = passed >= 2
    summary["stage1_verdict"] = "PASS" if stage1_pass else "FAIL"
    summary["stage1_criteria_passed"] = passed
    summary["stage1_criteria_details"] = criteria

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n{'='*70}")
    print(f"Base-Centered Local Recovery Audit")
    print(f"{'='*70}")
    for group_name in ["all", "base16", "base32"]:
        g = summary.get(group_name, {})
        if g.get("n", 0) == 0:
            continue
        print(f"\n  {group_name} (n={g['n']}):")
        print(f"    base:     median={g.get('base_median','?')}, mean={g.get('base_mean','?')}")
        for r in radii:
            print(f"    r={r}: GT_in_crop={g.get(f'GT_in_crop_r{r}',0):.3f}, "
                  f"top1_med={g.get(f'top1_median_r{r}','?')}, "
                  f"oracle_med={g.get(f'oracle_median_r{r}','?')}, "
                  f"top1_better2px={g.get(f'top1_better_2px_frac_r{r}',0):.3f}, "
                  f"oracle_better2px={g.get(f'oracle_better_2px_frac_r{r}',0):.3f}")

    print(f"\n  Stage-1: {summary['stage1_verdict']} ({passed} criteria passed)")
    for d in criteria:
        print(f"    {d}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
