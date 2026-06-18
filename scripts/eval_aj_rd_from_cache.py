#!/usr/bin/env python3
"""Compute re-entry metrics from unified strided+original cache.

This script reports two distinct quantities:

  - `aj_proxy`: the historical first-reentry-frame proxy used for diagnostics
  - `aj_rd`: TAPNext++-style post-reappearance AJ_RD over eligible events

The two metrics are intentionally kept separate so that proxy-based sanity
checks do not get confused with the paper metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache
from utils.reentry_metrics import (
    DEFAULT_AJRD_D_MINS,
    DEFAULT_AJ_THRESHOLDS,
    DEFAULT_PROXY_THRESHOLDS,
    compute_first_reentry_proxy,
    compute_reappearance_segment_aj,
    eligible_reentry_events,
    summarize_reappearance_ajrd,
)


def _bucket_stats(vals: np.ndarray) -> Dict[str, Any]:
    if vals.size == 0:
        return {"n": 0, "median_px": 0.0, "mean_px": 0.0, "p95_px": 0.0, "lt4px": 0.0, "lt8px": 0.0, "lt16px": 0.0}
    return {
        "n": int(vals.size),
        "median_px": round(float(np.median(vals)), 2),
        "mean_px": round(float(np.mean(vals)), 2),
        "p95_px": round(float(np.percentile(vals, 95)), 2),
        "lt4px": round(float(np.mean(vals < 4)), 4),
        "lt8px": round(float(np.mean(vals < 8)), 4),
        "lt16px": round(float(np.mean(vals < 16)), 4),
    }


def compute_reentry_metrics(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_vis: np.ndarray,
    gt_vis: np.ndarray,
    query_points: np.ndarray,
    height: int,
    width: int,
    thresholds: Sequence[int] = DEFAULT_PROXY_THRESHOLDS,
    d_mins: Sequence[int] = DEFAULT_AJRD_D_MINS,
) -> Dict[str, Any]:
    """Compute per-query proxy and TAPNext++-style AJ_RD metrics for one video."""
    per_query: List[Dict[str, Any]] = []
    proxy_errors: List[float] = []
    proxy_aj_terms: List[float] = []
    eligible_counts = {int(d): 0 for d in d_mins}

    for i in range(gt_tracks.shape[0]):
        qt = int(round(float(query_points[i, 0])))
        proxy = compute_first_reentry_proxy(
            pred_tracks=pred_tracks[i],
            gt_tracks=gt_tracks[i],
            pred_visibility=pred_vis[i],
            gt_visibility=gt_vis[i],
            query_t=qt,
            height=height,
            width=width,
            thresholds=thresholds,
        )
        if proxy is None:
            continue

        eligible_events = eligible_reentry_events(gt_vis[i], qt)
        for evt in eligible_events:
            occ_len = int(evt["occ_length"])
            for d in d_mins:
                if occ_len >= int(d):
                    eligible_counts[int(d)] += 1

        proxy_errors.append(float(proxy["error_px"]))
        proxy_aj_terms.append(float(proxy["aj_proxy"]))
        ajrd_events = []
        for evt in eligible_events:
            ajrd = compute_reappearance_segment_aj(
                pred_tracks=pred_tracks[i],
                gt_tracks=gt_tracks[i],
                pred_visibility=pred_vis[i],
                gt_visibility=gt_vis[i],
                event=evt,
                height=height,
                width=width,
                thresholds=DEFAULT_AJ_THRESHOLDS,
            )
            if ajrd is not None:
                ajrd_events.append(ajrd)

        ajrd_summary = summarize_reappearance_ajrd(ajrd_events, d_mins=d_mins)
        per_query.append({
            "query_idx": i,
            "query_t": qt,
            "reentry_t": int(proxy["reentry_t"]),
            "occ_length": int(proxy["occ_length"]),
            "proxy_error_px": round(float(proxy["error_px"]), 3),
            "proxy_pred_visible": bool(proxy["pred_visible"]),
            "proxy_gt_visible": bool(proxy["gt_visible"]),
            "aj_proxy": round(float(proxy["aj_proxy"]), 4),
            "ajrd_events": ajrd_events,
            "ajrd_summary": ajrd_summary,
            **{k: v for k, v in proxy.items() if k.startswith("jaccard_")},
        })

    n_q = len(per_query)
    if n_q == 0:
        return {
            "n_reentry_queries": 0,
            "n_eligible_events_by_dmin": {str(int(d)): 0 for d in d_mins},
            "per_query": [],
        }

    occ_lengths = np.array([q["occ_length"] for q in per_query], dtype=np.int64)
    long_20 = occ_lengths >= 20
    long_50 = occ_lengths >= 50

    return {
        "n_reentry_queries": n_q,
        "first_reentry_frame_proxy": round(float(np.mean(proxy_aj_terms)) if proxy_aj_terms else 0.0, 4),
        "true_AJ_RD": round(float(np.mean([q["ajrd_summary"]["aj_rd"] for q in per_query if q.get("ajrd_summary", {}).get("aj_rd") is not None])) if any(q.get("ajrd_summary", {}).get("aj_rd") is not None for q in per_query) else 0.0, 4) if any(q.get("ajrd_summary", {}).get("aj_rd") is not None for q in per_query) else None,
        "n_eligible_events_by_dmin": {str(int(d)): int(eligible_counts[int(d)]) for d in d_mins},
        "reentry_error": _bucket_stats(np.asarray(proxy_errors, dtype=np.float32)),
        "long_occ_ge20": _bucket_stats(np.asarray([q["proxy_error_px"] for q in per_query if q["occ_length"] >= 20], dtype=np.float32)),
        "long_occ_ge50": _bucket_stats(np.asarray([q["proxy_error_px"] for q in per_query if q["occ_length"] >= 50], dtype=np.float32)),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute re-entry metrics from unified cache")
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all")
    parser.add_argument("--thresholds", type=str, default="1,2,4,8,16")
    parser.add_argument("--ajrd-d-mins", type=str, default="1,4,16,64,256")
    args = parser.parse_args()

    proxy_thresholds = tuple(int(t) for t in args.thresholds.split(",") if t.strip())
    d_mins = tuple(int(t) for t in args.ajrd_d_mins.split(",") if t.strip())

    payload = load_attempt0_cache(Path(args.cache_path))
    records = payload["records"]
    if args.max_videos > 0:
        records = records[:args.max_videos]

    all_results: List[Dict[str, Any]] = []
    for r in records:
        vid = str(r["video_id"])
        h, w = int(r["original_size"][0]), int(r["original_size"][1])
        pred = np.asarray(r["pred_tracks"], dtype=np.float32)
        gt = np.asarray(r["gt_tracks"], dtype=np.float32)
        pvis = np.asarray(r["pred_visibility"], dtype=bool)
        gvis = np.asarray(r["gt_visibility"], dtype=bool)
        qpts = np.asarray(r["query_points"], dtype=np.float32)

        vid_result = compute_reentry_metrics(
            pred_tracks=pred,
            gt_tracks=gt,
            pred_vis=pvis,
            gt_vis=gvis,
            query_points=qpts,
            height=h,
            width=w,
            thresholds=proxy_thresholds,
            d_mins=d_mins,
        )
        vid_result["video_id"] = vid
        all_results.append(vid_result)

    total_n = sum(int(r["n_reentry_queries"]) for r in all_results)
    proxy_vals: List[float] = []
    ajrd_vals: List[float] = []
    proxy_errs: List[float] = []
    long20_errs: List[float] = []
    long50_errs: List[float] = []

    for r in all_results:
        for q in r.get("per_query", []):
            proxy_vals.append(float(q["aj_proxy"]))
            proxy_errs.append(float(q["proxy_error_px"]))
            if q.get("ajrd_summary", {}).get("aj_rd") is not None:
                ajrd_vals.append(float(q["ajrd_summary"]["aj_rd"]))
            if q["occ_length"] >= 20:
                long20_errs.append(float(q["proxy_error_px"]))
            if q["occ_length"] >= 50:
                long50_errs.append(float(q["proxy_error_px"]))

    summary = {
        "cache_path": args.cache_path,
        "protocol": payload.get("protocol", "unknown"),
        "model_name": payload.get("model_name", "unknown"),
        "metric_name": "reentry_proxy_and_ajrd",
        "note": (
            "first_reentry_frame_proxy = single-frame Jaccard at first re-entry frame. "
            "true_AJ_RD = TAPNext++ style AJ computed over full post-reappearance trajectory. "
            "Only true AJ_RD should be used for paper-level comparisons."
        ),
        "thresholds": list(proxy_thresholds),
        "ajrd_d_mins": list(d_mins),
        "n_videos": len(all_results),
        "n_reentry_queries_total": total_n,
        "first_reentry_frame_proxy": round(float(np.mean(proxy_vals)) if proxy_vals else 0.0, 4),
        "true_AJ_RD": round(float(np.mean(ajrd_vals)) if ajrd_vals else 0.0, 4) if ajrd_vals else None,
        "n_eligible_events_by_dmin": {
            str(int(d)): int(sum(int(r["n_eligible_events_by_dmin"][str(int(d))]) for r in all_results))
            for d in d_mins
        },
        "reentry_error": _bucket_stats(np.asarray(proxy_errs, dtype=np.float32)),
        "long_occ_ge20": _bucket_stats(np.asarray(long20_errs, dtype=np.float32)),
        "long_occ_ge50": _bucket_stats(np.asarray(long50_errs, dtype=np.float32)),
        "per_video": [{k: v for k, v in r.items() if k != "per_query"} for r in all_results],
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.output_json}")
    print(
        f"first_reentry_proxy={summary['first_reentry_frame_proxy']:.4f}, "
        f"true_AJ_RD={summary['true_AJ_RD']}, "
        f"n_reentry={total_n}, "
        f"median={summary['reentry_error']['median_px']:.1f}px, "
        f"<4px={summary['reentry_error']['lt4px']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
