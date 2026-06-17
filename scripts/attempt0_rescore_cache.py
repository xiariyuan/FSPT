#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.attempt0_schema import load_attempt0_cache, validate_attempt0_cache, write_json_report


def _mean_metric(rows: List[Dict[str, float]], key: str) -> Optional[float]:
    values = [float(r[key]) for r in rows if key in r and np.isfinite(r[key])]
    if not values:
        return None
    return float(np.mean(values))


def _max_reappearance_occlusion_run(gt_visibility: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    n_queries, num_frames = gt_visibility.shape
    out = np.zeros((n_queries,), dtype=np.int64)
    gt_occluded = ~gt_visibility
    for q in range(n_queries):
        start = int(query_t[q]) + 1
        if start >= num_frames:
            continue
        best = 0
        run = 0
        for val in gt_occluded[q, start:].tolist():
            if val:
                run += 1
                continue
            if run > best:
                best = run
            run = 0
        out[q] = best
    return out


def _score_subset(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_visibility: np.ndarray,
    gt_visibility: np.ndarray,
    query_points: np.ndarray,
    resolution_hw: tuple[int, int],
    query_mode: str,
) -> Optional[Dict[str, float]]:
    if pred_tracks.shape[0] == 0:
        return None
    metrics = compute_tapvid_metrics(
        pred_tracks=torch.from_numpy(pred_tracks).float(),
        gt_tracks=torch.from_numpy(gt_tracks).float(),
        pred_visibility=torch.from_numpy(pred_visibility),
        gt_visibility=torch.from_numpy(gt_visibility),
        query_points=torch.from_numpy(query_points).float(),
        resolution=resolution_hw,
        exclude_query_frame=True,
        query_mode=query_mode,
    )
    keep = ("AJ", "OA", "<avg", "<4px", "avg_error_px", "median_error_px")
    return {k: float(metrics[k]) for k in keep if k in metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified rescoring on an Attempt 0 cache.")
    parser.add_argument("--cache", type=str, required=True, help="Path to unified .pt/.pth/.json cache.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON summary path.")
    parser.add_argument("--query-mode", type=str, default="strided", choices=("first", "strided"))
    parser.add_argument(
        "--metric-resolution-mode",
        type=str,
        default="original",
        choices=("original", "input"),
        help="Use original_size or model_input_size from cache metadata.",
    )
    parser.add_argument("--long-occ-min-run", type=int, default=20)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--norm-tol", type=float, default=1.5,
                        help="Max absolute value for normalized coordinates (default 1.5).")
    args = parser.parse_args()

    payload = load_attempt0_cache(args.cache)
    report = validate_attempt0_cache(payload, require_gt=True, normalized_coord_threshold=args.norm_tol)
    if not report["valid"]:
        raise SystemExit(
            f"Cache validation failed. Run attempt0_validate_cache.py first. errors={len(report['errors'])}"
        )

    records = payload["records"]
    if args.limit_records > 0:
        records = records[: args.limit_records]

    per_record: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, float]] = []
    long_occ_rows: List[Dict[str, float]] = []

    for idx, record in enumerate(records):
        pred_tracks = np.asarray(record["pred_tracks"], dtype=np.float32)
        gt_tracks = np.asarray(record["gt_tracks"], dtype=np.float32)
        pred_visibility = np.asarray(record["pred_visibility"])
        gt_visibility = np.asarray(record["gt_visibility"])
        query_points = np.asarray(record["query_points"], dtype=np.float32)

        if pred_visibility.dtype != np.bool_:
            pred_visibility = pred_visibility > 0.5
        if gt_visibility.dtype != np.bool_:
            gt_visibility = gt_visibility > 0.5

        if args.metric_resolution_mode == "input" and "model_input_size" in record:
            resolution_hw = tuple(int(v) for v in np.asarray(record["model_input_size"]).reshape(-1)[:2])
        else:
            resolution_hw = tuple(int(v) for v in np.asarray(record["original_size"]).reshape(-1)[:2])

        overall = _score_subset(
            pred_tracks=pred_tracks,
            gt_tracks=gt_tracks,
            pred_visibility=pred_visibility,
            gt_visibility=gt_visibility,
            query_points=query_points,
            resolution_hw=resolution_hw,
            query_mode=args.query_mode,
        )
        if overall is None:
            continue

        overall_rows.append(overall)

        query_t = np.clip(np.round(query_points[:, 0]).astype(np.int64), 0, pred_tracks.shape[1] - 1)
        reapp_runs = _max_reappearance_occlusion_run(gt_visibility, query_t)
        long_mask = reapp_runs >= int(args.long_occ_min_run)
        long_occ = None
        if long_mask.any():
            long_occ = _score_subset(
                pred_tracks=pred_tracks[long_mask],
                gt_tracks=gt_tracks[long_mask],
                pred_visibility=pred_visibility[long_mask],
                gt_visibility=gt_visibility[long_mask],
                query_points=query_points[long_mask],
                resolution_hw=resolution_hw,
                query_mode=args.query_mode,
            )
            if long_occ is not None:
                long_occ_rows.append(long_occ)

        per_record.append(
            {
                "record_index": idx,
                "video_id": str(record["video_id"]),
                "num_queries": int(query_points.shape[0]),
                "num_long_occ_queries": int(long_mask.sum()),
                "resolution_hw": [int(resolution_hw[0]), int(resolution_hw[1])],
                "overall": overall,
                "long_occ": long_occ,
            }
        )

    summary = {
        "model_name": payload.get("model_name", ""),
        "dataset_name": payload.get("dataset_name", ""),
        "query_mode": args.query_mode,
        "metric_resolution_mode": args.metric_resolution_mode,
        "long_occ_min_run": int(args.long_occ_min_run),
        "num_records": len(per_record),
        "overall_mean": {
            key: _mean_metric(overall_rows, key)
            for key in ("AJ", "OA", "<avg", "<4px", "avg_error_px", "median_error_px")
        },
        "long_occ_mean": {
            key: _mean_metric(long_occ_rows, key)
            for key in ("AJ", "OA", "<avg", "<4px", "avg_error_px", "median_error_px")
        },
        "per_record": per_record,
    }
    write_json_report(args.out, summary)
    print(json.dumps(summary["overall_mean"], ensure_ascii=True, indent=2))
    print(f"[ok] wrote unified rescoring summary to {args.out}")


if __name__ == "__main__":
    main()

