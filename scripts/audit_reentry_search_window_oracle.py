#!/usr/bin/env python3
"""Audit whether frozen baseline priors cover re-entry ground truth within local windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache
from utils.coords import find_reentry_events, pixel_l2_error

WINDOW_RADII = (16, 32, 64, 128, 256)
BASELINES = ("cotracker3_online", "cotracker3_offline", "trackon2", "oracle_teacher")
BASELINE_ALIASES = {
    "cotracker3_baseline": "cotracker3_online",
    "cotracker3_online": "cotracker3_online",
    "cotracker3_offline": "cotracker3_offline",
    "trackon2": "trackon2",
    "trackon2_dinov3": "trackon2",
}


def get_record_source(cache: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    records = cache.get("records", [])
    if not isinstance(records, list):
        raise KeyError("Unsupported cache format: expected top-level 'records' list.")
    return ((str(record.get("video_id", f"record_{idx}")), record) for idx, record in enumerate(records))


def canonical_baseline_name(cache: Dict[str, Any], path: Path) -> str:
    model_name = str(cache.get("model_name", "")).strip().lower()
    stem = path.stem.lower()
    for key, alias in BASELINE_ALIASES.items():
        if key in model_name or key in stem:
            return alias
    return model_name or stem


def infer_size(record: Dict[str, Any]) -> Tuple[int, int]:
    size = record.get("original_size")
    if size is None:
        size = record.get("size")
    if size is None:
        raise KeyError("Missing original_size/size in cache record.")
    return int(size[0]), int(size[1])


def as_array(value: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(value).astype(dtype, copy=False)


def summarize_distances(distances: Sequence[float]) -> Dict[str, Optional[float]]:
    if not distances:
        return {"median": None, "p75": None, "p90": None, "p95": None}
    arr = np.asarray(distances, dtype=np.float32)
    return {
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def reentry_events_for_track(gt_visibility: np.ndarray, query_t: int, min_occ_length: int) -> List[Dict[str, int]]:
    return [dict(evt) for evt in find_reentry_events(gt_visibility, query_t) if int(evt["occ_length"]) >= int(min_occ_length)]


def evaluate_baseline(record: Dict[str, Any], baseline_key: str, min_occ_length: int, d_mins: Sequence[int]) -> Dict[str, Any]:
    height, width = infer_size(record)
    gt_tracks = as_array(record["gt_tracks"], np.float32)
    gt_visibility = as_array(record["gt_visibility"], bool)
    pred_tracks = as_array(record["pred_tracks"], np.float32)
    pred_visibility = as_array(record["pred_visibility"], bool)
    query_points = as_array(record.get("query_points"), np.float32)

    if pred_tracks.ndim != 3 or pred_visibility.ndim != 2:
        raise ValueError(f"Unexpected prediction shapes: pred_tracks={pred_tracks.shape}, pred_visibility={pred_visibility.shape}")
    if pred_tracks.shape[0] == gt_tracks.shape[1] and pred_tracks.shape[1] == gt_tracks.shape[0]:
        pred_tracks = pred_tracks.transpose(1, 0, 2)
    if pred_visibility.shape[0] == gt_visibility.shape[1] and pred_visibility.shape[1] == gt_visibility.shape[0]:
        pred_visibility = pred_visibility.transpose(1, 0)

    distances_by_radius: Dict[int, List[float]] = {radius: [] for radius in WINDOW_RADII}
    all_distances: List[float] = []
    corrected_scores: List[float] = []
    n_events = 0

    for qidx in range(gt_tracks.shape[0]):
        query_t = int(round(float(query_points[qidx, 0])))
        for event in reentry_events_for_track(gt_visibility[qidx], query_t, min_occ_length):
            t_re = int(event["reentry_frame"])
            n_events += 1
            pred_xy = np.asarray(pred_tracks[qidx, t_re], dtype=np.float32)
            gt_xy = np.asarray(gt_tracks[qidx, t_re], dtype=np.float32)
            err = float(
                pixel_l2_error(
                    pred_xy[None, :],
                    gt_xy[None, :],
                    height,
                    width,
                    pred_fmt="yx_norm",
                    gt_fmt="yx_norm",
                )[0]
            )
            all_distances.append(err)
            for radius in WINDOW_RADII:
                distances_by_radius[radius].append(1.0 if err <= float(radius) else 0.0)

            if any(int(event["occ_length"]) >= int(d) for d in d_mins):
                corrected_scores.append(float(np.mean([1.0 if err <= float(d) else 0.0 for d in d_mins if int(event["occ_length"]) >= int(d)])))

    coverage = {f"coverage@{radius}": round(float(np.mean(values)), 4) if values else None for radius, values in distances_by_radius.items()}
    coverage.update(summarize_distances(all_distances))
    return {
        "baseline": baseline_key,
        "n_events": int(n_events),
        "coverage": coverage,
        "oracle_aj_rd": round(float(np.mean(corrected_scores)), 4) if corrected_scores else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-occ", type=int, default=16)
    parser.add_argument("--d-mins", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    args = parser.parse_args()

    out_dir = args.output_dir / "p0_search_window_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_baseline: Dict[str, List[Dict[str, Any]]] = {key: [] for key in BASELINES}
    for cache_path in args.cache:
        cache = load_attempt0_cache(cache_path)
        baseline_key = canonical_baseline_name(cache, cache_path)
        if baseline_key not in per_baseline:
            per_baseline[baseline_key] = []
        for _record_name, record in get_record_source(cache):
            per_baseline[baseline_key].append(evaluate_baseline(record, baseline_key, args.min_occ, args.d_mins))

    summary_lines = ["# Search-window oracle summary", ""]
    decision_lines = ["# Decision", ""]
    gate_rows: List[Tuple[str, float, float]] = []
    for key, results in per_baseline.items():
        if not results:
            continue
        cov64 = [r["coverage"]["coverage@64"] for r in results if r["coverage"]["coverage@64"] is not None]
        cov128 = [r["coverage"]["coverage@128"] for r in results if r["coverage"]["coverage@128"] is not None]
        med = [r["coverage"]["median"] for r in results if r["coverage"]["median"] is not None]
        aj = [r["oracle_aj_rd"] for r in results if r["oracle_aj_rd"] is not None]
        summary_lines.append(f"## {key}")
        summary_lines.append(f"- n_samples: {len(results)}")
        summary_lines.append(f"- coverage@64: {round(float(np.mean(cov64)), 4) if cov64 else None}")
        summary_lines.append(f"- coverage@128: {round(float(np.mean(cov128)), 4) if cov128 else None}")
        summary_lines.append(f"- median distance: {round(float(np.mean(med)), 4) if med else None}")
        summary_lines.append(f"- oracle_aj_rd: {round(float(np.mean(aj)), 4) if aj else None}")
        gate_rows.append((key, float(np.mean(cov128)) if cov128 else 0.0, float(np.mean(aj)) if aj else 0.0))

    if gate_rows:
        decision_lines.append("- Keep direction only if every baseline clears the oracle gates.")
        for key, cov128, aj in gate_rows:
            verdict = "pass" if cov128 >= 0.70 and aj >= 0.05 else "fail"
            decision_lines.append(f"- {key}: {verdict} (coverage@128={cov128:.4f}, oracle_aj_rd={aj:.4f})")
    else:
        decision_lines.append("- No usable baselines found in cache.")

    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (out_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(per_baseline, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
