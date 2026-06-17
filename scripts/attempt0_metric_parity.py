#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official
from utils.attempt0_schema import load_attempt0_cache, validate_attempt0_cache, write_json_report


def _convert_to_official_inputs(
    pred_tracks_yx: np.ndarray,
    gt_tracks_yx: np.ndarray,
    pred_visibility: np.ndarray,
    gt_visibility: np.ndarray,
    query_points_tyx: np.ndarray,
    resolution_hw: tuple[int, int],
) -> Dict[str, np.ndarray]:
    h, w = float(resolution_hw[0]), float(resolution_hw[1])
    scale_w = max(w - 1.0, 1.0)
    scale_h = max(h - 1.0, 1.0)

    pred_tracks_xy = pred_tracks_yx[..., [1, 0]].astype(np.float32) * np.array([scale_w, scale_h], dtype=np.float32)
    gt_tracks_xy = gt_tracks_yx[..., [1, 0]].astype(np.float32) * np.array([scale_w, scale_h], dtype=np.float32)
    query_points_px = query_points_tyx.astype(np.float32).copy()
    query_points_px[:, 1] *= scale_h
    query_points_px[:, 2] *= scale_w
    pred_occluded = ~pred_visibility.astype(bool)
    gt_occluded = ~gt_visibility.astype(bool)
    return {
        "query_points": query_points_px[None, ...],
        "gt_occluded": gt_occluded[None, ...],
        "gt_tracks": gt_tracks_xy[None, ...],
        "pred_occluded": pred_occluded[None, ...],
        "pred_tracks": pred_tracks_xy[None, ...],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FSPT metric wrapper against official TAP-Vid core.")
    parser.add_argument("--cache", type=str, required=True, help="Unified cache containing GT.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--query-mode", type=str, default="strided", choices=("first", "strided"))
    parser.add_argument(
        "--metric-resolution-mode",
        type=str,
        default="original",
        choices=("original", "input"),
        help="Use original_size or model_input_size from cache metadata.",
    )
    parser.add_argument("--norm-tol", type=float, default=1.5,
                        help="Max absolute value for normalized coordinates (default 1.5).")
    parser.add_argument("--limit-records", type=int, default=3)
    args = parser.parse_args()

    payload = load_attempt0_cache(args.cache)
    report = validate_attempt0_cache(payload, require_gt=True, normalized_coord_threshold=args.norm_tol)
    if not report["valid"]:
        raise SystemExit("Cache validation failed; fix cache before parity audit.")

    records = payload["records"][: args.limit_records] if args.limit_records > 0 else payload["records"]
    per_record: List[Dict[str, Any]] = []
    max_abs_diff = 0.0

    for idx, record in enumerate(records):
        pred_tracks = np.asarray(record["pred_tracks"], dtype=np.float32)
        gt_tracks = np.asarray(record["gt_tracks"], dtype=np.float32)
        pred_visibility = np.asarray(record["pred_visibility"])
        gt_visibility = np.asarray(record["gt_visibility"])
        query_points = np.asarray(record["query_points"], dtype=np.float32)
        if args.metric_resolution_mode == "input" and "model_input_size" in record:
            resolution_hw = tuple(int(v) for v in np.asarray(record["model_input_size"]).reshape(-1)[:2])
        else:
            resolution_hw = tuple(int(v) for v in np.asarray(record["original_size"]).reshape(-1)[:2])

        wrapped = compute_tapvid_metrics(
            pred_tracks=torch.from_numpy(pred_tracks).float(),
            gt_tracks=torch.from_numpy(gt_tracks).float(),
            pred_visibility=torch.from_numpy(pred_visibility),
            gt_visibility=torch.from_numpy(gt_visibility),
            query_points=torch.from_numpy(query_points).float(),
            resolution=resolution_hw,
            exclude_query_frame=True,
            query_mode=args.query_mode,
        )

        off_inputs = _convert_to_official_inputs(
            pred_tracks_yx=pred_tracks,
            gt_tracks_yx=gt_tracks,
            pred_visibility=pred_visibility > 0.5 if pred_visibility.dtype != np.bool_ else pred_visibility,
            gt_visibility=gt_visibility > 0.5 if gt_visibility.dtype != np.bool_ else gt_visibility,
            query_points_tyx=query_points,
            resolution_hw=resolution_hw,
        )
        official = compute_tapvid_metrics_official(
            query_points=off_inputs["query_points"],
            gt_occluded=off_inputs["gt_occluded"],
            gt_tracks=off_inputs["gt_tracks"],
            pred_occluded=off_inputs["pred_occluded"],
            pred_tracks=off_inputs["pred_tracks"],
            query_mode=args.query_mode,
            thresholds=(1, 2, 4, 8, 16),
            get_trackwise_metrics=False,
        )

        alias_official = {
            "AJ": float(np.asarray(official["average_jaccard"]).reshape(-1)[0]),
            "OA": float(np.asarray(official["occlusion_accuracy"]).reshape(-1)[0]),
            "<avg": float(np.asarray(official["average_pts_within_thresh"]).reshape(-1)[0]),
            "<4px": float(np.asarray(official["pts_within_4"]).reshape(-1)[0]),
        }
        wrapped_view = {
            "AJ": float(wrapped["AJ"]),
            "OA": float(wrapped["OA"]),
            "<avg": float(wrapped["<avg"]),
            "<4px": float(wrapped["<4px"]),
        }
        diffs = {k: abs(wrapped_view[k] - alias_official[k]) for k in wrapped_view.keys()}
        max_abs_diff = max(max_abs_diff, max(diffs.values()))

        per_record.append(
            {
                "record_index": idx,
                "video_id": str(record["video_id"]),
                "wrapped": wrapped_view,
                "official": alias_official,
                "abs_diff": diffs,
            }
        )

    summary = {
        "query_mode": args.query_mode,
        "metric_resolution_mode": args.metric_resolution_mode,
        "num_records": len(per_record),
        "max_abs_diff": float(max_abs_diff),
        "pass_threshold_1e-3": bool(max_abs_diff <= 1e-3),
        "per_record": per_record,
    }
    write_json_report(args.out, summary)
    print(json.dumps({k: summary[k] for k in ("num_records", "max_abs_diff", "pass_threshold_1e-3")}, indent=2))
    if not summary["pass_threshold_1e-3"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
