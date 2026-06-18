#!/usr/bin/env python3
"""Compute AJ_RD and re-entry metrics from unified strided+original cache.

AJ_RD = average Jaccard on re-entry detection: for each query, evaluate only
at the first re-entry frame. Complements the standard AJ which averages over
all evaluation frames.

Also reports: re-entry first-frame error, long-occ bucket metrics, n_reentry.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import (
    find_first_reentry, pixel_l2_error,
    yx_norm_to_xy_pixel, assert_coord_range,
)
from utils.attempt0_schema import load_attempt0_cache


def compute_aj_rd(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_vis: np.ndarray,
    gt_vis: np.ndarray,
    query_points: np.ndarray,
    height: int,
    width: int,
    thresholds: tuple = (1, 2, 4, 8, 16),
) -> Dict[str, Any]:
    """Compute AJ_RD and re-entry metrics for a single video.

    Args:
        pred_tracks: (N, T, 2) [y,x] normalized
        gt_tracks: (N, T, 2) [y,x] normalized
        pred_vis: (N, T) bool (True = visible)
        gt_vis: (N, T) bool (True = visible)
        query_points: (N, 3) [t, y, x] normalized
        height, width: original image dimensions

    Returns:
        dict with per-query re-entry metrics and video-level summary
    """
    N, T = gt_tracks.shape[:2]
    per_query = []
    reentry_errors = []
    reentry_pred_vis_errors = []  # error when model thinks visible at re-entry

    for i in range(N):
        qt = int(round(float(query_points[i, 0])))
        re = find_first_reentry(gt_vis[i], qt)
        if re is None:
            continue

        t_re = re["reentry_frame"]
        occ_len = re["occ_length"]

        # Error at re-entry frame
        pred_yx = pred_tracks[i, t_re]
        gt_yx = gt_tracks[i, t_re]
        err = float(pixel_l2_error(
            pred_yx[None, :], gt_yx[None, :], height, width,
            pred_fmt="yx_norm", gt_fmt="yx_norm"
        )[0])

        # Predicted visibility at re-entry
        pred_visible_at_re = bool(pred_vis[i, t_re])

        # AJ at re-entry frame: for each threshold, compute Jaccard
        jaccards = {}
        for thr in thresholds:
            # Jaccard at a single frame = indicator(pred < thr)
            # (standard Jaccard: |pred ∩ gt| / |pred ∪ gt| = 1 if correct else 0 for single point)
            jaccards[f"jaccard_{thr}"] = 1.0 if err < thr else 0.0

        aj = np.mean(list(jaccards.values()))

        per_query.append({
            "query_idx": i,
            "query_t": qt,
            "reentry_t": t_re,
            "occ_length": occ_len,
            "error_px": round(err, 3),
            "pred_visible": pred_visible_at_re,
            **{k: round(v, 3) for k, v in jaccards.items()},
            "aj": round(aj, 3),
        })
        reentry_errors.append(err)

    n_q = len(per_query)
    if n_q == 0:
        return {"n_reentry_queries": 0}

    errs = np.array(reentry_errors)

    # Long-occ splits
    occ_lengths = np.array([q["occ_length"] for q in per_query])
    long_20 = occ_lengths >= 20
    long_50 = occ_lengths >= 50

    def _bucket_stats(mask):
        if not mask.any():
            return None
        e = errs[mask]
        return {
            "n": int(mask.sum()),
            "median_px": round(float(np.median(e)), 2),
            "mean_px": round(float(np.mean(e)), 2),
            "p95_px": round(float(np.percentile(e, 95)), 2),
            "lt4px": round(float(np.mean(e < 4)), 4),
            "lt8px": round(float(np.mean(e < 8)), 4),
            "lt16px": round(float(np.mean(e < 16)), 4),
        }

    aj_vals = np.array([q["aj"] for q in per_query])

    return {
        "n_reentry_queries": n_q,
        "aj_rd": round(float(np.mean(aj_vals)), 4),
        "reentry_error": _bucket_stats(np.ones(n_q, dtype=bool)),
        "long_occ_ge20": _bucket_stats(long_20),
        "long_occ_ge50": _bucket_stats(long_50),
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute AJ_RD from unified cache")
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    payload = load_attempt0_cache(Path(args.cache_path))
    records = payload["records"]
    if args.max_videos > 0:
        records = records[:args.max_videos]

    all_results = []
    for r in records:
        vid = r["video_id"]
        h, w = int(r["original_size"][0]), int(r["original_size"][1])
        pred = np.asarray(r["pred_tracks"], dtype=np.float32)
        gt = np.asarray(r["gt_tracks"], dtype=np.float32)
        pvis = np.asarray(r["pred_visibility"], dtype=bool)
        gvis = np.asarray(r["gt_visibility"], dtype=bool)
        qpts = np.asarray(r["query_points"], dtype=np.float32)

        vid_result = compute_aj_rd(pred, gt, pvis, gvis, qpts, h, w)
        vid_result["video_id"] = vid
        all_results.append(vid_result)

    # Aggregate across videos
    total_n = sum(r["n_reentry_queries"] for r in all_results)
    all_aj = []
    all_errs = []
    all_long20_errs = []
    all_long50_errs = []

    for r in all_results:
        for q in r.get("per_query", []):
            all_aj.append(q["aj"])
            all_errs.append(q["error_px"])
            if q["occ_length"] >= 20:
                all_long20_errs.append(q["error_px"])
            if q["occ_length"] >= 50:
                all_long50_errs.append(q["error_px"])

    summary = {
        "cache_path": args.cache_path,
        "protocol": payload.get("protocol", "unknown"),
        "model_name": payload.get("model_name", "unknown"),
        "n_videos": len(all_results),
        "n_reentry_queries_total": total_n,
        "aj_rd": round(float(np.mean(all_aj)) if all_aj else 0, 4),
        "reentry_error": {
            "median_px": round(float(np.median(all_errs)) if all_errs else 0, 2),
            "mean_px": round(float(np.mean(all_errs)) if all_errs else 0, 2),
            "lt4px": round(float(np.mean(np.array(all_errs) < 4)) if all_errs else 0, 4),
        },
        "long_occ_ge20": {
            "n": len(all_long20_errs),
            "median_px": round(float(np.median(all_long20_errs)) if all_long20_errs else 0, 2),
            "lt4px": round(float(np.mean(np.array(all_long20_errs) < 4)) if all_long20_errs else 0, 4),
        } if all_long20_errs else None,
        "long_occ_ge50": {
            "n": len(all_long50_errs),
            "median_px": round(float(np.median(all_long50_errs)) if all_long50_errs else 0, 2),
            "lt4px": round(float(np.mean(np.array(all_long50_errs) < 4)) if all_long50_errs else 0, 4),
        } if all_long50_errs else None,
        "per_video": [
            {k: v for k, v in r.items() if k != "per_query"}
            for r in all_results
        ],
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output_json}")
    print(f"AJ_RD={summary['aj_rd']:.4f}, n_reentry={total_n}, "
          f"median={summary['reentry_error']['median_px']:.1f}px, "
          f"<4px={summary['reentry_error']['lt4px']*100:.1f}%")


if __name__ == "__main__":
    main()
