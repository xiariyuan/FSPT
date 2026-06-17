#!/usr/bin/env python3
"""
Final recovery quality comparison: position error before/after retracking.

Runs baseline (no recovery), cotracker-recovery, and dino-recovery on TAP-Vid DAVIS.
Computes per-query position errors against GT tracks, focusing on re-entry frames.

Key metrics:
- Overall median pixel error (all queries, all frames)
- Re-entry frame error (first visible frame after occlusion)
- Error on retracked queries vs non-retracked
- Error change: baseline vs recovery

Usage:
  python scripts/eval_recovery_position_error.py \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth \
    --max-batches 20
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def build_model(cfg, checkpoint_path, device):
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
    return model.to(device).eval()


def build_val_loader(config):
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    return _build_val_loader_from_config(config)


def find_reentry_frames(gt_vis: np.ndarray, query_t: int) -> list:
    """Find frames where a point re-appears after occlusion starting from query_t."""
    vis = gt_vis.astype(bool)
    reentry_frames = []
    in_occlusion = False
    for t in range(query_t + 1, len(vis)):
        if not vis[t]:
            in_occlusion = True
        elif in_occlusion:
            reentry_frames.append(t)
            in_occlusion = False
    return reentry_frames


def compute_pixel_error(pred_xy: np.ndarray, gt_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    """Compute pixel error from normalized coords."""
    pred_px = pred_xy * np.array([W, H])
    gt_px = gt_xy * np.array([W, H])
    return np.linalg.norm(pred_px - gt_px, axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/recovery_position_error_audit.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = {
        "baseline": "configs/fspt_online_recovery_real_eval256.yaml",
        "dino_recovery": "configs/fspt_online_recovery_dino_real_eval256.yaml",
        "learned_head": "configs/fspt_online_recovery_learned_head_smoke.yaml",
    }

    all_results = {}

    for name, config_path in configs.items():
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        cfg = load_config(config_path)
        model = build_model(cfg, args.checkpoint, device)
        dataloader = build_val_loader(cfg)

        query_errors = []  # per-query errors at re-entry frames
        retracked_errors = []
        non_retracked_errors = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.max_batches: break
            if isinstance(batch, dict):
                video = batch.get("video", batch.get("frames"))
                query_points = batch.get("query_points", batch.get("queries"))
                gt_tracks = batch.get("target_points", batch.get("tracks", batch.get("gt_tracks")))
                gt_visibility = batch.get("occluded", batch.get("visibility", batch.get("gt_visibility")))
                meta = {k: v for k, v in batch.items()
                        if k not in ("video", "frames", "query_points", "queries",
                                     "target_points", "tracks", "gt_tracks",
                                     "occluded", "visibility", "gt_visibility",
                                     "video_name", "original_size")}
            else:
                continue
            if video is None or query_points is None: continue

            video = video.to(device)
            query_points = query_points.to(device)

            with torch.no_grad():
                try:
                    outputs = model(video, query_points, meta=meta, return_info=True)
                except Exception as e:
                    print(f"  Batch {batch_idx}: ERROR - {e}")
                    continue

            pred_tracks = outputs[0].cpu()  # (B, N, T, 2) normalized
            info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}
            relocal_mask = info.get("relocal_mask", None)
            retracking_mask = info.get("retracking_retrack_mask_bn", None)

            B, N, T, _ = pred_tracks.shape
            if video.dim() == 5:
                _, _, _, H, W = video.shape
            else:
                _, _, H, W = video.shape

            # Get GT tracks
            if isinstance(gt_tracks, torch.Tensor):
                gt = gt_tracks.cpu()
            else:
                continue

            # GT tracks are in pixel coords; pred_tracks are normalized [0,1]
            # Convert GT to normalized coords using original_size if available
            original_size = batch.get("original_size", None)
            if isinstance(original_size, torch.Tensor):
                orig_h, orig_w = int(original_size[0, 0].item()), int(original_size[0, 1].item())
            else:
                orig_h, orig_w = H, W
            # Normalize GT: divide by (orig_w, orig_h)
            gt_norm = gt.clone()
            gt_norm[..., 0] = gt[..., 0] / orig_w
            gt_norm[..., 1] = gt[..., 1] / orig_h

            # GT visibility: occluded=True means invisible, we want visible
            if isinstance(gt_visibility, torch.Tensor):
                gt_vis = (~gt_visibility).cpu()  # (B, N, T) True=visible

            for b in range(B):
                for n in range(N):
                    qt = int(query_points[b, n, 0].item())
                    pred = pred_tracks[b, n].numpy()  # (T, 2) normalized
                    gt_np = gt_norm[b, n].numpy()  # (T, 2) normalized

                    # Compute pixel error at each frame
                    errors = compute_pixel_error(pred, gt_np, H, W)

                    # Find GT re-entry frames (visible after occlusion)
                    gt_vis_np = gt_vis[b, n].numpy() if isinstance(gt_vis, torch.Tensor) else np.ones(T, dtype=bool)
                    reentry_frames = find_reentry_frames(gt_vis_np, qt)

                    is_retracked = False
                    if isinstance(retracking_mask, torch.Tensor):
                        is_retracked = bool(retracking_mask[b, n].item())

                    entry = {
                        "batch": batch_idx, "b": b, "n": n,
                        "retracked": is_retracked,
                        "overall_median_error": float(np.median(errors)),
                    }

                    # Error at re-entry frames (first 3)
                    if reentry_frames:
                        re_errors = [float(errors[t]) for t in reentry_frames[:3]]
                        entry["reentry_frame_errors"] = re_errors
                        entry["reentry_first_error"] = re_errors[0]
                        query_errors.append(re_errors[0])

                        if is_retracked:
                            retracked_errors.append(re_errors[0])
                        else:
                            non_retracked_errors.append(re_errors[0])

                    # Error on retracked vs non-retracked queries
                    if isinstance(relocal_mask, torch.Tensor):
                        rm = relocal_mask[b, n].cpu().numpy()
                        # Error at relocal frames
                        relocal_frames = np.where(rm)[0]
                        if len(relocal_frames) > 0:
                            entry["relocal_frame_errors"] = [float(errors[t]) for t in relocal_frames[:5]]

        # Summary
        all_errors = np.array(query_errors) if query_errors else np.array([0])
        retracked_arr = np.array(retracked_errors) if retracked_errors else np.array([])
        non_retracked_arr = np.array(non_retracked_errors) if non_retracked_errors else np.array([0])

        summary = {
            "n_queries_with_reentry": len(query_errors),
            "overall_reentry_median": float(np.median(all_errors)),
            "overall_reentry_mean": float(np.mean(all_errors)),
            "overall_reentry_lt4px": float(np.mean(all_errors < 4.0)),
            "n_retracked": len(retracked_errors),
            "n_non_retracked": len(non_retracked_errors),
        }
        if len(retracked_arr) > 0:
            summary["retracked_median"] = float(np.median(retracked_arr))
            summary["retracked_mean"] = float(np.mean(retracked_arr))
            summary["retracked_lt4px"] = float(np.mean(retracked_arr < 4.0))
        if len(non_retracked_arr) > 0:
            summary["non_retracked_median"] = float(np.median(non_retracked_arr))
            summary["non_retracked_mean"] = float(np.mean(non_retracked_arr))
            summary["non_retracked_lt4px"] = float(np.mean(non_retracked_arr < 4.0))

        all_results[name] = summary

        print(f"  Queries with re-entry: {summary['n_queries_with_reentry']}")
        print(f"  Retracked: {summary['n_retracked']}, Non-retracked: {summary['n_non_retracked']}")
        print(f"  Overall reentry median: {summary['overall_reentry_median']:.2f} px")
        if "retracked_median" in summary:
            print(f"  Retracked median: {summary['retracked_median']:.2f} px")
        if "non_retracked_median" in summary:
            print(f"  Non-retracked median: {summary['non_retracked_median']:.2f} px")

    # Cross-comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<35s} {'Baseline':>12s} {'DINO Recovery':>15s}")
    print(f"{'-'*60}")

    baseline = all_results.get("baseline", {})
    dino = all_results.get("dino_recovery", {})

    for key in ["n_queries_with_reentry", "overall_reentry_median", "overall_reentry_lt4px",
                 "n_retracked", "retracked_median", "retracked_lt4px",
                 "non_retracked_median", "non_retracked_lt4px"]:
        bv = baseline.get(key, "N/A")
        dv = dino.get(key, "N/A")
        if isinstance(bv, float):
            bv = f"{bv:.3f}"
        if isinstance(dv, float):
            dv = f"{dv:.3f}"
        print(f"  {key:<33s} {str(bv):>12s} {str(dv):>15s}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
