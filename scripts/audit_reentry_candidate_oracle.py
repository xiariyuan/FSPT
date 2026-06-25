#!/usr/bin/env python3
"""Audit whether candidate generators can recover GT in top-k under re-entry."""
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
from utils.coords import find_reentry_events, pixel_l2_error, yx_norm_to_xy_pixel

TOP_KS = (1, 5, 10, 20)
THRESHOLDS = (4, 8, 16)
BASELINES = ("cotracker3_online", "cotracker3_offline", "trackon2")
BASELINE_ALIASES = {
    "cotracker3_baseline": "cotracker3_online",
    "cotracker3_online": "cotracker3_online",
    "cotracker3_offline": "cotracker3_offline",
    "trackon2": "trackon2",
    "trackon2_dinov3": "trackon2",
}


def get_records(cache: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
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
        raise KeyError("Missing original_size/size.")
    return int(size[0]), int(size[1])


def as_array(value: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(value).astype(dtype, copy=False)


def make_grid_candidates(center_xy: np.ndarray, radius: int, grid_size: int) -> np.ndarray:
    offsets = np.linspace(-radius, radius, grid_size, dtype=np.float32)
    rel_offsets = [np.array([dx, dy], dtype=np.float32) for dy in offsets for dx in offsets]
    rel_offsets = sorted(rel_offsets, key=lambda off: float(np.linalg.norm(off)))
    candidates = [center_xy + off for off in rel_offsets]
    return np.stack(candidates, axis=0)


def hit_at_k(candidates: np.ndarray, gt_xy: np.ndarray, top_k: int, threshold: float) -> bool:
    dist = np.sqrt(np.sum((candidates[:top_k] - gt_xy[None, :]) ** 2, axis=-1))
    return bool(np.min(dist) <= float(threshold))


def evaluate_variant(record: Dict[str, Any], baseline_key: str, min_occ: int) -> Dict[str, Any]:
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

    rows: List[Dict[str, Any]] = []
    topk_hits = {(k, thr): [] for k in TOP_KS for thr in THRESHOLDS}
    miss_but_hit = {k: [] for k in TOP_KS if k > 1}
    oracle_scores: List[float] = []

    for qidx in range(gt_tracks.shape[0]):
        query_t = int(round(float(query_points[qidx, 0])))
        events = [e for e in find_reentry_events(gt_visibility[qidx], query_t) if int(e["occ_length"]) >= int(min_occ)]
        for event in events:
            t_re = int(event["reentry_frame"])
            center_xy = np.asarray(yx_norm_to_xy_pixel(pred_tracks[qidx, t_re], height, width), dtype=np.float32)
            gt_xy = np.asarray(yx_norm_to_xy_pixel(gt_tracks[qidx, t_re], height, width), dtype=np.float32)
            candidates = make_grid_candidates(center_xy, radius=64, grid_size=9)

            base_err = float(
                pixel_l2_error(
                    pred_tracks[qidx, t_re][None, :],
                    gt_tracks[qidx, t_re][None, :],
                    height,
                    width,
                    pred_fmt="yx_norm",
                    gt_fmt="yx_norm",
                )[0]
            )

            row = {
                "query_idx": int(qidx),
                "reentry_t": t_re,
                "occ_length": int(event["occ_length"]),
                "base_error_px": round(base_err, 4),
                "base_visible": bool(pred_visibility[qidx, t_re]),
                "gt_visible": bool(gt_visibility[qidx, t_re]),
            }
            for top_k in TOP_KS:
                for thr in THRESHOLDS:
                    hit = hit_at_k(candidates, gt_xy, top_k, thr)
                    topk_hits[(top_k, thr)].append(1.0 if hit else 0.0)
                    if top_k > 1 and not hit and hit_at_k(candidates, gt_xy, 1, thr):
                        miss_but_hit[top_k].append(1.0)
            oracle_scores.append(1.0 if np.min(np.linalg.norm(candidates - gt_xy[None, :], axis=-1)) <= 16 else 0.0)
            rows.append(row)

    metrics = {f"top{k}@{thr}": round(float(np.mean(vals)), 4) if vals else None for (k, thr), vals in topk_hits.items()}
    metrics.update({f"miss_top1_hit_top{k}": round(float(np.mean(vals)), 4) if vals else None for k, vals in miss_but_hit.items()})
    return {
        "baseline": baseline_key,
        "n_events": int(len(rows)),
        "metrics": metrics,
        "candidate_oracle_aj_rd": round(float(np.mean(oracle_scores)), 4) if oracle_scores else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-occ", type=int, default=16)
    args = parser.parse_args()

    out_dir = args.output_dir / "p1_candidate_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_baseline: Dict[str, List[Dict[str, Any]]] = {k: [] for k in BASELINES}
    for cache_path in args.cache:
        cache = load_attempt0_cache(cache_path)
        baseline_key = canonical_baseline_name(cache, cache_path)
        if baseline_key not in per_baseline:
            per_baseline[baseline_key] = []
        for _record_name, record in get_records(cache):
            per_baseline[baseline_key].append(evaluate_variant(record, baseline_key, args.min_occ))

    summary_lines = ["# Candidate oracle summary", ""]
    decision_lines = ["# Decision", ""]
    for key, results in per_baseline.items():
        if not results:
            continue
        top10_16 = [r["metrics"]["top10@16"] for r in results if r["metrics"]["top10@16"] is not None]
        oracle_gain = [r["candidate_oracle_aj_rd"] for r in results if r["candidate_oracle_aj_rd"] is not None]
        summary_lines.append(f"## {key}")
        summary_lines.append(f"- n_samples: {len(results)}")
        summary_lines.append(f"- top10@16: {round(float(np.mean(top10_16)), 4) if top10_16 else None}")
        summary_lines.append(f"- candidate_oracle_aj_rd: {round(float(np.mean(oracle_gain)), 4) if oracle_gain else None}")
        verdict = "pass" if (np.mean(top10_16) if top10_16 else 0.0) >= 0.50 and (np.mean(oracle_gain) if oracle_gain else 0.0) >= 0.05 else "fail"
        decision_lines.append(f"- {key}: {verdict}")

    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (out_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(per_baseline, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
