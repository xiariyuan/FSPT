#!/usr/bin/env python3
"""
Stage 1: Predicted-depth world-state vs 2D tracker on PointOdyssey.

Low-memory implementation:
  - stream sequences one by one
  - keep only compact numeric arrays per noise level
  - keep at most a small sample of per-query rows for JSON inspection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_world_state_stage0 import (
    compute_reentry_errors,
    discover_sequences,
    find_reentry_queries,
    load_sequence,
    project_3d_to_2d,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUMMARY_OCC_THRESHOLDS = [10, 15, 20, 30, 50]
SUMMARY_CAM_THRESHOLDS = [0.0, 0.1, 0.3, 0.5, 1.0]
TABLE_SUBSETS = ["overall", "occ20_cam0.30", "occ30_cam0.30", "cam_motion_gte_0.30"]
ERROR_KEYS = [
    "hold_2d_error_px",
    "hold_3d_reproj_error_px",
    "gt3d_reproj_error_px",
    "pred_depth_3d_hold_error_px",
]


def lift_2d_to_3d_from_scalar(
    point_2d: np.ndarray,
    depth: float,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """Lift a 2D point to a 3D world point from a scalar camera-space depth."""
    k_inv = np.linalg.inv(intrinsics)
    pixel_h = np.array([float(point_2d[0]), float(point_2d[1]), 1.0], dtype=np.float64)
    pt_cam = float(depth) * (k_inv @ pixel_h)
    e_inv = np.linalg.inv(extrinsics)
    return e_inv[:3, :3] @ pt_cam + e_inv[:3, 3]


def sample_log_depth(
    z_depth: float,
    sigma: float,
    rng: np.random.Generator,
    clip_log_abs: float,
) -> float:
    """Sample multiplicative depth noise in log space with clipping."""
    if sigma <= 0.0:
        return float(z_depth)
    eps = float(rng.normal(0.0, sigma))
    eps = float(np.clip(eps, -clip_log_abs, clip_log_abs))
    return float(z_depth * np.exp(eps))


def summarize_numeric_rows(rows: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Summarize a compact numeric table with Stage 0-compatible field names."""
    count = int(rows["occ_length"].shape[0])
    out: Dict[str, Any] = {"label": "subset", "count": count}
    if count == 0:
        return out

    for key in ERROR_KEYS:
        arr = rows[key]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        out[f"{key}_mean"] = float(finite.mean())
        out[f"{key}_median"] = float(np.median(finite))
        p99 = float(np.percentile(finite, 99))
        clipped = finite[finite <= p99]
        out[f"{key}_robust_mean"] = float(clipped.mean()) if clipped.size > 0 else float("nan")
        for thr in [1, 2, 4, 8, 16, 32]:
            out[f"{key}_lt{thr}px"] = float((finite < thr).mean())

    # Keep Stage 0 field for backward compatibility with old readers.
    for suffix in ["mean", "median", "robust_mean", "lt1px", "lt2px", "lt4px", "lt8px", "lt16px", "lt32px"]:
        pred_key = f"pred_depth_3d_hold_error_px_{suffix}"
        if pred_key in out:
            out[f"extrap_3d_reproj_error_px_{suffix}"] = out[pred_key]
    return out


def stratified_summary_numeric(rows: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Stage 0-style summary over compact numeric arrays."""
    summary: Dict[str, Any] = {}
    n = rows["occ_length"].shape[0]
    all_mask = np.ones(n, dtype=bool)

    def subset(mask: np.ndarray, label: str) -> Dict[str, Any]:
        subset_rows = {k: v[mask] for k, v in rows.items()}
        out = summarize_numeric_rows(subset_rows)
        out["label"] = label
        return out

    summary["overall"] = subset(all_mask, "all")

    for occ_thr in SUMMARY_OCC_THRESHOLDS:
        mask = rows["occ_length"] >= occ_thr
        summary[f"occ_gte_{occ_thr}"] = subset(mask, f"occ>={occ_thr}")

    for cam_thr in SUMMARY_CAM_THRESHOLDS:
        mask = rows["camera_motion_rotation"] >= cam_thr
        summary[f"cam_motion_gte_{cam_thr:.2f}"] = subset(mask, f"cam_motion>={cam_thr:.2f}")

    for occ_thr in SUMMARY_OCC_THRESHOLDS:
        for cam_thr in SUMMARY_CAM_THRESHOLDS:
            mask = (rows["occ_length"] >= occ_thr) & (rows["camera_motion_rotation"] >= cam_thr)
            if int(mask.sum()) >= 5:
                summary[f"occ{occ_thr}_cam{cam_thr:.2f}"] = subset(mask, f"occ{occ_thr}_cam{cam_thr:.2f}")

    return summary


def init_store() -> Dict[str, List[np.ndarray]]:
    return {k: [] for k in ["occ_length", "camera_motion_rotation", *ERROR_KEYS]}


def materialize_store(store: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, chunks in store.items():
        if chunks:
            out[key] = np.concatenate(chunks, axis=0)
        else:
            dtype = np.int32 if key == "occ_length" else np.float32
            out[key] = np.empty((0,), dtype=dtype)
    return out


def append_examples(
    example_store: List[Dict[str, Any]],
    example_rows: List[Dict[str, Any]],
    limit: int = 500,
) -> None:
    if len(example_store) >= limit:
        return
    example_store.extend(example_rows[: max(0, limit - len(example_store))])


def build_sequence_numeric(
    seq: Dict[str, np.ndarray],
    queries,
    base_rows: List[Dict[str, Any]],
    noise_sigma: float,
    rng: np.random.Generator,
    clip_log_abs: float,
) -> tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Compute compact numeric arrays and a small sample of inspectable rows."""
    trajs_2d = seq["trajs_2d"]
    trajs_3d = seq["trajs_3d"]
    intrinsics = seq["intrinsics"]
    extrinsics = seq["extrinsics"]

    n = len(queries)
    occ = np.empty((n,), dtype=np.int32)
    cam_rot = np.empty((n,), dtype=np.float32)
    err_2d = np.empty((n,), dtype=np.float32)
    err_3d = np.empty((n,), dtype=np.float32)
    err_gt = np.empty((n,), dtype=np.float32)
    err_pred = np.empty((n,), dtype=np.float32)
    examples: List[Dict[str, Any]] = []

    write_idx = 0
    for q, base_row in zip(queries, base_rows):
        t_q = q.query_frame
        t_re = q.reentry_frame
        i = q.point_idx

        hold_3d = trajs_3d[t_q, i]
        pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
        z_depth = float(pt_cam[2])
        if not np.isfinite(z_depth) or z_depth <= 1e-6:
            continue

        if noise_sigma == 0.0:
            pred_world = hold_3d
            noisy_z = z_depth
        else:
            noisy_z = sample_log_depth(z_depth, noise_sigma, rng, clip_log_abs)
            pred_world = lift_2d_to_3d_from_scalar(
                trajs_2d[t_q, i],
                noisy_z,
                intrinsics[t_q],
                extrinsics[t_q],
            )

        reproj_pred = project_3d_to_2d(pred_world, intrinsics[t_re], extrinsics[t_re])
        gt_2d_reentry = trajs_2d[t_re, i]
        pred_error = float(np.linalg.norm(reproj_pred - gt_2d_reentry))

        occ[write_idx] = int(base_row["occ_length"])
        cam_rot[write_idx] = float(base_row["camera_motion_rotation"])
        err_2d[write_idx] = float(base_row["hold_2d_error_px"])
        err_3d[write_idx] = float(base_row["hold_3d_reproj_error_px"])
        err_gt[write_idx] = float(base_row["gt3d_reproj_error_px"])
        err_pred[write_idx] = pred_error

        if len(examples) < 500:
            examples.append(
                {
                    "seq_name": base_row["seq_name"],
                    "point_idx": base_row["point_idx"],
                    "query_frame": base_row["query_frame"],
                    "reentry_frame": base_row["reentry_frame"],
                    "occ_length": base_row["occ_length"],
                    "camera_motion_rotation": base_row["camera_motion_rotation"],
                    "camera_motion_translation": base_row["camera_motion_translation"],
                    "gt3d_reproj_error_px": float(base_row["gt3d_reproj_error_px"]),
                    "hold_2d_error_px": float(base_row["hold_2d_error_px"]),
                    "hold_3d_reproj_error_px": float(base_row["hold_3d_reproj_error_px"]),
                    "pred_depth_3d_hold_error_px": pred_error,
                    "pred_depth_noise_sigma": float(noise_sigma),
                    "pred_depth_query_z": noisy_z,
                    "gt_2d_reentry": base_row["gt_2d_reentry"],
                }
            )

        write_idx += 1

    data = {
        "occ_length": occ[:write_idx],
        "camera_motion_rotation": cam_rot[:write_idx],
        "hold_2d_error_px": err_2d[:write_idx],
        "hold_3d_reproj_error_px": err_3d[:write_idx],
        "gt3d_reproj_error_px": err_gt[:write_idx],
        "pred_depth_3d_hold_error_px": err_pred[:write_idx],
    }
    return data, examples


def print_table(noise_results: Dict[str, Dict[str, Any]], noise_scales: List[float]) -> None:
    print("\n" + "=" * 120)
    print("STAGE 1 RESULTS: Predicted-depth world-state vs 2D hold")
    print("=" * 120)
    for subset_key in TABLE_SUBSETS:
        if subset_key not in noise_results[str(noise_scales[0])]["summary"]:
            continue
        base = noise_results[str(noise_scales[0])]["summary"][subset_key]
        print(f"\n--- {subset_key} (n={base['count']:,}) ---")
        header = (
            f"{'noise':>6} | {'2D_med':>9} {'2D_<4':>7} | "
            f"{'GT3D_med':>9} {'GT3D_<4':>8} | {'Pred3D_med':>10} {'Pred3D_<4':>9}"
        )
        print(header)
        print("-" * len(header))
        for noise_sigma in noise_scales:
            s = noise_results[str(noise_sigma)]["summary"][subset_key]
            print(
                f"{noise_sigma:>6.2f} | "
                f"{s['hold_2d_error_px_median']:>9.2f} {s['hold_2d_error_px_lt4px']:>6.1%} | "
                f"{s['hold_3d_reproj_error_px_median']:>9.2f} {s['hold_3d_reproj_error_px_lt4px']:>7.1%} | "
                f"{s['pred_depth_3d_hold_error_px_median']:>10.2f} {s['pred_depth_3d_hold_error_px_lt4px']:>8.1%}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Predicted-depth robustness")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--min-occ-length", type=int, default=10)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--noise-scales",
        type=str,
        default="0.0,0.02,0.05,0.10,0.15",
        help="Log-depth sigma values. 0.15 means multiplicative exp(N(0,0.15)).",
    )
    parser.add_argument(
        "--clip-log-abs",
        type=float,
        default=0.35,
        help="Clamp absolute log-depth perturbation to keep synthetic noise realistic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    noise_scales = [float(x) for x in args.noise_scales.split(",") if x.strip()]
    rngs = {str(ns): np.random.default_rng(args.seed + idx) for idx, ns in enumerate(noise_scales)}

    logger.info("Discovering PointOdyssey sequences")
    sequences = discover_sequences(data_root, splits)
    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]
    logger.info("Using %d sequences", len(sequences))

    store_by_noise = {str(ns): init_store() for ns in noise_scales}
    examples_by_noise: Dict[str, List[Dict[str, Any]]] = {str(ns): [] for ns in noise_scales}
    total_queries = 0

    for seq_idx, seq_path in enumerate(sequences):
        seq = load_sequence(seq_path)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        if not queries:
            continue

        base_rows = compute_reentry_errors(seq, queries)
        total_queries += len(queries)

        for noise_sigma in noise_scales:
            numeric, examples = build_sequence_numeric(
                seq=seq,
                queries=queries,
                base_rows=base_rows,
                noise_sigma=noise_sigma,
                rng=rngs[str(noise_sigma)],
                clip_log_abs=args.clip_log_abs,
            )
            for key, arr in numeric.items():
                store_by_noise[str(noise_sigma)][key].append(arr)
            append_examples(examples_by_noise[str(noise_sigma)], examples, limit=500)

        logger.info(
            "[%d/%d] %s: %d queries, accumulated=%d",
            seq_idx + 1,
            len(sequences),
            seq_path.name,
            len(queries),
            total_queries,
        )

    if total_queries == 0:
        raise RuntimeError("No re-entry queries found for Stage 1.")

    noise_results: Dict[str, Dict[str, Any]] = {}
    for noise_sigma in noise_scales:
        rows = materialize_store(store_by_noise[str(noise_sigma)])
        summary = stratified_summary_numeric(rows)
        noise_results[str(noise_sigma)] = {
            "summary": summary,
            "per_query_results": examples_by_noise[str(noise_sigma)],
        }

    print_table(noise_results, noise_scales)

    zero_summary = noise_results[str(0.0)]["summary"]["overall"]
    zero_gap = abs(
        zero_summary["pred_depth_3d_hold_error_px_median"] - zero_summary["hold_3d_reproj_error_px_median"]
    )
    print("\n" + "=" * 120)
    print(f"Sanity: noise=0 median gap to GT3D hold = {zero_gap:.8f} px")
    print("=" * 120)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(args),
            "total_queries": total_queries,
            "noise_results": noise_results,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
