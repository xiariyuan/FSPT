#!/usr/bin/env python3
"""ReEntry-unified: minimum-segment-length visibility filter — model-agnostic, same for all trackers.

Motivation: Very short re-entry segments (< L_min frames) are more likely to be
flickering false positives than true re-detections. Filtering them improves
precision at a small recall cost, which should improve AJ_RD.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/gemini/code/FSPT")
REPO = ROOT / "external/tapnextpp/repo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official


def find_reentry_segments(vis: np.ndarray) -> list[tuple[int, int, int]]:
    """Find all visible segments that start after occlusion (re-entry events).

    Args:
        vis: (T,) bool, True=visible

    Returns:
        List of (start_t, end_t_exclusive, length) for each re-entry segment.
        Re-entry = segment starts at t where vis[t]=True AND (t==0 OR vis[t-1]=False).
    """
    T = len(vis)
    segments = []
    t = 0
    while t < T:
        if vis[t]:
            start = t
            while t < T and vis[t]:
                t += 1
            end = t
            length = end - start
            # Is this a re-entry? Either starts at frame 0, or previous frame was occluded
            if start == 0 or not vis[start - 1]:
                segments.append((start, end, length))
        else:
            t += 1
    return segments


def apply_reentry_min_duration(
    pred_vis: np.ndarray,
    L_min: int = 3,
) -> np.ndarray:
    """Apply minimum-duration ReEntry filter.

    Args:
        pred_vis: (Q, T) bool, predicted visibility
        L_min: minimum segment length to keep

    Returns:
        Corrected visibility (Q, T) bool
    """
    corrected = pred_vis.copy()
    for q in range(pred_vis.shape[0]):
        segments = find_reentry_segments(pred_vis[q])
        for start, end, length in segments:
            if length < L_min:
                corrected[q, start:end] = False
    return corrected


def apply_reentry_jump_filter(
    pred_tracks: np.ndarray,  # (Q, T, 2) [x,y] pixel
    pred_vis: np.ndarray,     # (Q, T) bool
    jump_threshold_px: float = 64.0,
) -> np.ndarray:
    """Filter re-entry segments where the track jumps too far from last known position.

    Args:
        pred_tracks: (Q, T, 2) pixel coords [x, y]
        pred_vis: (Q, T) bool
        jump_threshold_px: max allowed jump distance in pixels

    Returns:
        Corrected visibility (Q, T) bool
    """
    corrected = pred_vis.copy()
    Q, T = pred_vis.shape
    for q in range(Q):
        segments = find_reentry_segments(pred_vis[q])
        for start, end, length in segments:
            if start == 0:
                continue  # first frame, no previous position to compare
            # Find last known good position (last frame before occlusion)
            last_good = start - 1
            while last_good >= 0 and not pred_vis[q, last_good]:
                last_good -= 1
            if last_good < 0:
                continue
            # Check jump distance
            prev_pos = pred_tracks[q, last_good]
            reentry_pos = pred_tracks[q, start]
            dist = np.sqrt(np.sum((reentry_pos - prev_pos) ** 2))
            if dist > jump_threshold_px:
                corrected[q, start:end] = False
    return corrected


def apply_reentry_combined(
    pred_tracks: np.ndarray,
    pred_vis: np.ndarray,
    L_min: int = 3,
    jump_threshold_px: float = 64.0,
) -> np.ndarray:
    """Apply combined ReEntry: min-duration + jump filter."""
    corrected = apply_reentry_min_duration(pred_vis, L_min=L_min)
    corrected = apply_reentry_jump_filter(pred_tracks, corrected, jump_threshold_px=jump_threshold_px)
    return corrected


def eval_cache_with_reentry(
    cache_path: str,
    output_dir: Path,
    label: str,
    L_min: int = 3,
    jump_threshold_px: float = 64.0,
    PIX: float = 255.0,
) -> dict:
    """Evaluate a prediction cache with ReEntry applied."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = payload["records"]

    per_video = {}
    all_ajrd = []

    for r in records:
        vid = str(r["video_id"])
        pred_yx = np.asarray(r["pred_tracks"], dtype=np.float32)     # (N,T,2) [y,x] norm
        gt_yx = np.asarray(r["gt_tracks"], dtype=np.float32)         # (N,T,2) [y,x] norm
        pred_vis = np.asarray(r["pred_visibility"], dtype=bool)      # (N,T) True=visible
        gt_vis = np.asarray(r["gt_visibility"], dtype=bool)          # (N,T)
        qpts = np.asarray(r["query_points"], dtype=np.float32)       # (N,3) [t,y,x] norm

        N, T = pred_yx.shape[:2]

        # Convert to pixel [x,y]
        pred_xy_px = pred_yx[..., ::-1] * PIX    # (N,T,2) [x,y] px
        gt_xy_px = gt_yx[..., ::-1] * PIX        # (N,T,2) [x,y] px
        qpts_px = qpts.copy()
        qpts_px[:, 1:] *= PIX                    # [t, y_px, x_px]

        # Apply ReEntry
        pred_vis_corrected = apply_reentry_combined(
            pred_xy_px, pred_vis, L_min=L_min, jump_threshold_px=jump_threshold_px
        )
        n_flipped = int((pred_vis != pred_vis_corrected).sum())
        pred_occ_corrected = ~pred_vis_corrected
        gt_occ = ~gt_vis

        # Official TAP-Vid metrics
        tapvid = compute_tapvid_metrics_official(
            query_points=qpts_px[None].astype(np.float32),
            gt_occluded=gt_occ[None].astype(bool),
            gt_tracks=gt_xy_px[None].astype(np.float32),
            pred_occluded=pred_occ_corrected[None],
            pred_tracks=pred_xy_px[None].astype(np.float32),
            query_mode="first", thresholds=(1, 2, 4, 8, 16),
        )
        tapvid = {k: float(v.item() if hasattr(v, 'item') else v) for k, v in tapvid.items()}

        # AJ_RD
        try:
            pred_t_b = torch.from_numpy(pred_xy_px.copy()).float().unsqueeze(0).permute(0, 2, 1, 3)
            pred_v_b = torch.from_numpy(pred_vis_corrected).bool().unsqueeze(0).permute(0, 2, 1)
            gt_t_b = torch.from_numpy(gt_xy_px.copy()).float().unsqueeze(0).permute(0, 2, 1, 3)
            gt_v_b = torch.from_numpy(gt_vis).bool().unsqueeze(0).permute(0, 2, 1)
            ajrd = compute_redetection_metrics(
                pred_tracks=pred_t_b, pred_visible=pred_v_b,
                gt_tracks=gt_t_b, gt_visible=gt_v_b,
            )
            ajrd_val = ajrd.get("AJ_RD")
            if ajrd_val is not None and ajrd_val == ajrd_val:
                all_ajrd.append(float(ajrd_val))
        except Exception:
            ajrd_val = None

        per_video[vid] = {
            "queries": N, "frames": T, "n_flipped": n_flipped,
            "tapvid": tapvid,
            "aj_rd": float(ajrd_val) if ajrd_val is not None and ajrd_val == ajrd_val else None,
        }

    aj_vals = [v["tapvid"]["average_jaccard"] for v in per_video.values()]
    oa_vals = [v["tapvid"]["occlusion_accuracy"] for v in per_video.values()]
    da_vals = [v["tapvid"]["average_pts_within_thresh"] for v in per_video.values()]
    d4_vals = [v["tapvid"]["pts_within_4"] for v in per_video.values()]
    valid_ajrd = [v["aj_rd"] for v in per_video.values() if v["aj_rd"] is not None]

    summary = {
        "label": label,
        "cache": cache_path,
        "L_min": L_min,
        "jump_threshold_px": jump_threshold_px,
        "n_videos": len(per_video),
        "AJ": round(float(np.mean(aj_vals)) * 100, 2),
        "OA": round(float(np.mean(oa_vals)) * 100, 2),
        "d_avg": round(float(np.mean(da_vals)) * 100, 2),
        "d_4px": round(float(np.mean(d4_vals)) * 100, 2),
        "AJ_RD": round(float(np.mean(valid_ajrd)), 4) if valid_ajrd else None,
        "n_ajrd_valid": len(valid_ajrd),
        "per_video": {k: {kk: vv for kk, vv in v.items() if kk != "tapvid"}
                      for k, v in per_video.items()},
    }

    out_path = output_dir / f"reentry_{label}_L{L_min}_j{jump_threshold_px:.0f}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    OUT_DIR = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    caches = {
        "cotracker3_baseline": str(ROOT / "outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt"),
        "cotracker3_offline": str(ROOT / "outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt"),
        "trackon2": str(ROOT / "outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt"),
    }

    # Test L_min values
    for L_min in [2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"L_min = {L_min}")
        print(f"{'='*60}")
        for label, cache_path in caches.items():
            p = Path(cache_path)
            if not p.exists():
                print(f"  {label}: SKIP (cache not found)")
                continue
            s = eval_cache_with_reentry(
                cache_path, OUT_DIR,
                label=label, L_min=L_min,
            )
            print(f"  {label:25s}  AJ={s['AJ']:.2f}%  OA={s['OA']:.2f}%  "
                  f"d_avg={s['d_avg']:.2f}%  d_4px={s['d_4px']:.2f}%  "
                  f"AJ_RD={s['AJ_RD']}")

    # Also test jump filter
    for jump in [32, 64, 128]:
        print(f"\n{'='*60}")
        print(f"Jump filter = {jump}px")
        print(f"{'='*60}")
        for label, cache_path in caches.items():
            p = Path(cache_path)
            if not p.exists():
                continue
            s = eval_cache_with_reentry(
                cache_path, OUT_DIR,
                label=label, L_min=0, jump_threshold_px=jump,
            )
            print(f"  {label:25s}  AJ={s['AJ']:.2f}%  OA={s['OA']:.2f}%  "
                  f"d_avg={s['d_avg']:.2f}%  d_4px={s['d_4px']:.2f}%  "
                  f"AJ_RD={s['AJ_RD']}")


if __name__ == "__main__":
    main()
