#!/usr/bin/env python3
"""
Stage 0: Oracle 3D world-state vs 2D tracker on PointOdyssey.

This script answers the foundational question for the world-state tracking pivot:
  "Does GT 3D world-state + camera projection outperform a 2D tracker at
   recovering points after long occlusion and camera motion?"

It loads PointOdyssey sequences with GT 2D/3D trajectories, camera intrinsics/
extrinsics, and visibility masks.  For each sequence it:

  1. Identifies "long-occlusion re-entry" queries (point occluded for >= N frames
     then reappears).
  2. Computes camera-motion magnitude between query frame and re-entry frame.
  3. Evaluates three baselines at re-entry and surrounding frames:
       - GT 2D oracle (upper bound for 2D)
       - GT 3D oracle → reproject (upper bound for world-state)
       - "Last-seen 2D hold" baseline (cheap 2D lower bound: just freeze the
         2D position from the last visible frame before occlusion)
  4. Reports per-threshold error and success rates.

The "CoTracker3 2D baseline" will be added in a follow-up once we verify the
oracle gap is large enough to justify running inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequence(seq_path: Path) -> Dict[str, np.ndarray]:
    """Load a PointOdyssey sequence's annotation and return a dict."""
    anno_path = seq_path / "anno.npz"
    if not anno_path.exists():
        raise FileNotFoundError(f"No anno.npz at {anno_path}")
    data = np.load(str(anno_path), allow_pickle=True)
    result = {k: data[k] for k in data.keys()}
    result["_path"] = str(seq_path)
    result["_name"] = seq_path.name
    return result


def discover_sequences(root: Path, splits: List[str]) -> List[Path]:
    """Find all sequence directories that contain anno.npz."""
    seqs = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and (d / "anno.npz").exists():
                seqs.append(d)
    return seqs


# ---------------------------------------------------------------------------
# Re-entry query extraction
# ---------------------------------------------------------------------------

@dataclass
class ReEntryQuery:
    point_idx: int
    query_frame: int          # last visible frame before occlusion
    reentry_frame: int        # first visible frame after occlusion
    occ_length: int           # number of occluded frames
    camera_motion: float      # camera motion magnitude (see below)
    seq_name: str


def find_reentry_queries(
    seq: Dict[str, np.ndarray],
    min_occ_length: int = 10,
    min_camera_motion: float = 0.0,
) -> List[ReEntryQuery]:
    """Find points that are visible, become occluded for >= min_occ_length frames,
    then become visible again.  Optionally filter by camera motion magnitude."""
    visibs = seq["visibs"]   # (T, N) bool
    extrinsics = seq["extrinsics"]  # (T, 4, 4)
    T, N = visibs.shape

    queries = []
    for i in range(N):
        vis = visibs[:, i]  # (T,)
        if not vis.any():
            continue

        # Walk through frames, find occlusion-then-reappearance events
        run_start = -1  # start of current occlusion run
        last_vis_before = -1
        for t in range(T):
            if vis[t]:
                if run_start >= 0 and (t - run_start) >= min_occ_length and last_vis_before >= 0:
                    # Found a re-entry event
                    occ_len = t - run_start
                    # Camera motion: Frobenius norm of relative pose change
                    cam_motion = 0.0
                    if min_camera_motion > 0 and extrinsics is not None:
                        E_before = extrinsics[last_vis_before]
                        E_after = extrinsics[t]
                        E_rel = E_after @ np.linalg.inv(E_before)
                        cam_motion = float(np.linalg.norm(E_rel[:3, :3] - np.eye(3)))
                    queries.append(ReEntryQuery(
                        point_idx=i,
                        query_frame=last_vis_before,
                        reentry_frame=t,
                        occ_length=occ_len,
                        camera_motion=cam_motion,
                        seq_name=seq.get("_name", "unknown"),
                    ))
                last_vis_before = t
                run_start = -1
            else:
                if run_start < 0:
                    run_start = t

    if min_camera_motion > 0:
        queries = [q for q in queries if q.camera_motion >= min_camera_motion]

    return queries


# ---------------------------------------------------------------------------
# 3D reprojection
# ---------------------------------------------------------------------------

def project_3d_to_2d(
    points_3d: np.ndarray,   # (3,) or (N, 3) world coords
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) world-to-camera
) -> np.ndarray:
    """Project 3D world points to 2D pixel coordinates using camera params."""
    single = points_3d.ndim == 1
    if single:
        points_3d = points_3d[None, :]

    # World to camera
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    pts_cam = (R @ points_3d.T).T + t  # (N, 3)

    # Camera to pixel
    K = intrinsics
    pts_2d_h = (K @ pts_cam.T).T  # (N, 3)
    # Avoid division by zero
    z = pts_2d_h[:, 2:3].clip(min=1e-6)
    pts_2d = pts_2d_h[:, :2] / z

    if single:
        return pts_2d[0]
    return pts_2d


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------

def compute_reentry_errors(
    seq: Dict[str, np.ndarray],
    queries: List[ReEntryQuery],
) -> List[Dict[str, Any]]:
    """For each re-entry query, compute errors for multiple baselines."""
    trajs_2d = seq["trajs_2d"]       # (T, N, 2)
    trajs_3d = seq["trajs_3d"]       # (T, N, 3)
    visibs = seq["visibs"]           # (T, N)
    intrinsics = seq["intrinsics"]   # (T, 3, 3)
    extrinsics = seq["extrinsics"]   # (T, 4, 4)
    T, N, _ = trajs_2d.shape

    results = []
    for q in queries:
        t_re = q.reentry_frame
        t_q = q.query_frame
        i = q.point_idx

        gt_2d_re = trajs_2d[t_re, i]  # GT 2D at re-entry
        gt_3d_re = trajs_3d[t_re, i]  # GT 3D at re-entry

        # --- Baseline 1: GT 2D oracle (just GT 2D at re-entry) ---
        # Error = 0 by definition, but we use it as reference for the
        # "what would a perfect 2D predictor achieve" question.

        # --- Baseline 2: GT 3D oracle → reproject ---
        # Use GT 3D position at re-entry frame, project with re-entry camera
        reproj_gt3d = project_3d_to_2d(
            gt_3d_re, intrinsics[t_re], extrinsics[t_re]
        )
        # This should match GT 2D (sanity check), but we report it anyway
        gt3d_reproj_error = float(np.linalg.norm(reproj_gt3d - gt_2d_re))

        # --- Baseline 3: "Last-seen 2D hold" ---
        # Freeze 2D position from last visible frame (query frame)
        hold_2d = trajs_2d[t_q, i]
        hold_error = float(np.linalg.norm(hold_2d - gt_2d_re))

        # --- Baseline 4: "3D hold + camera motion" ---
        # Freeze 3D world position from last visible frame, then reproject
        # with the re-entry camera (this accounts for camera motion!)
        hold_3d = trajs_3d[t_q, i]
        reproj_hold3d = project_3d_to_2d(
            hold_3d, intrinsics[t_re], extrinsics[t_re]
        )
        hold3d_error = float(np.linalg.norm(reproj_hold3d - gt_2d_re))

        # --- Baseline 5: "3D hold with interpolation" ---
        # During occlusion, predict 3D position by linear extrapolation
        # from the last 2 visible positions
        interp_error = hold3d_error  # default: same as hold
        if t_q >= 1:
            # Find previous visible frame
            t_prev = t_q - 1
            while t_prev >= 0 and not visibs[t_prev, i]:
                t_prev -= 1
            if t_prev >= 0:
                # Linear extrapolation in 3D
                dt = t_re - t_q
                v_3d = trajs_3d[t_q, i] - trajs_3d[t_prev, i]  # velocity
                extrap_3d = trajs_3d[t_q, i] + v_3d * dt
                reproj_extrap = project_3d_to_2d(
                    extrap_3d, intrinsics[t_re], extrinsics[t_re]
                )
                interp_error = float(np.linalg.norm(reproj_extrap - gt_2d_re))

        # Camera motion magnitude
        E_before = extrinsics[t_q]
        E_after = extrinsics[t_re]
        E_rel = E_after @ np.linalg.inv(E_before)
        cam_motion = float(np.linalg.norm(E_rel[:3, :3] - np.eye(3)))
        cam_translation = float(np.linalg.norm(E_rel[:3, 3]))

        results.append({
            "seq_name": q.seq_name,
            "point_idx": q.point_idx,
            "query_frame": t_q,
            "reentry_frame": t_re,
            "occ_length": q.occ_length,
            "camera_motion_rotation": cam_motion,
            "camera_motion_translation": cam_translation,
            # Errors
            "gt3d_reproj_error_px": gt3d_reproj_error,
            "hold_2d_error_px": hold_error,
            "hold_3d_reproj_error_px": hold3d_error,
            "extrap_3d_reproj_error_px": interp_error,
            # GT reference
            "gt_2d_reentry": gt_2d_re.tolist(),
        })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def stratified_summary(
    all_results: List[Dict[str, Any]],
    occ_thresholds: List[int],
    cam_motion_thresholds: List[float],
) -> Dict[str, Any]:
    """Compute summary statistics stratified by occlusion length and camera motion."""
    summary = {}

    # Overall
    summary["overall"] = _summarize_rows(all_results, "all")

    # By occlusion length
    for thr in occ_thresholds:
        subset = [r for r in all_results if r["occ_length"] >= thr]
        summary[f"occ_gte_{thr}"] = _summarize_rows(subset, f"occ>={thr}")

    # By camera motion
    for thr in cam_motion_thresholds:
        subset = [r for r in all_results if r["camera_motion_rotation"] >= thr]
        summary[f"cam_motion_gte_{thr:.2f}"] = _summarize_rows(subset, f"cam_motion>={thr:.2f}")

    # Joint: long occ + camera motion
    for occ_thr in occ_thresholds:
        for cam_thr in cam_motion_thresholds:
            subset = [
                r for r in all_results
                if r["occ_length"] >= occ_thr and r["camera_motion_rotation"] >= cam_thr
            ]
            if len(subset) >= 5:  # need minimum samples
                key = f"occ{occ_thr}_cam{cam_thr:.2f}"
                summary[key] = _summarize_rows(subset, key)

    return summary


def _summarize_rows(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if not rows:
        return {"label": label, "count": 0}

    keys = ["hold_2d_error_px", "hold_3d_reproj_error_px", "extrap_3d_reproj_error_px", "gt3d_reproj_error_px"]
    out: Dict[str, Any] = {"label": label, "count": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if np.isfinite(r.get(k, float("nan")))]
        if vals:
            arr = np.array(vals)
            out[f"{k}_mean"] = float(arr.mean())
            out[f"{k}_median"] = float(np.median(arr))
            # Winsorize at 99th percentile for a robust mean (avoids outlier blowup)
            p99 = float(np.percentile(arr, 99))
            clipped = arr[arr <= p99]
            out[f"{k}_robust_mean"] = float(clipped.mean()) if len(clipped) > 0 else float("nan")
            # Success rate at various thresholds
            for thr in [1, 2, 4, 8, 16, 32]:
                out[f"{k}_lt{thr}px"] = float((arr < thr).mean())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0: Oracle 3D vs 2D on PointOdyssey")
    parser.add_argument("--data-root", type=str,
                        default="/gemini/code/FSPT/datasets/pointodyssey",
                        help="PointOdyssey root directory")
    parser.add_argument("--splits", type=str, default="val,test",
                        help="Comma-separated splits to evaluate")
    parser.add_argument("--min-occ-length", type=int, default=10,
                        help="Minimum occlusion length for re-entry queries")
    parser.add_argument("--max-sequences", type=int, default=0,
                        help="Limit number of sequences (0=all)")
    parser.add_argument("--output-json", type=str, default="",
                        help="Output JSON path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    logger.info(f"Discovering sequences in {data_root} (splits={splits})")
    sequences = discover_sequences(data_root, splits)
    logger.info(f"Found {len(sequences)} sequences with anno.npz")

    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]
        logger.info(f"Limited to {len(sequences)} sequences")

    all_results: List[Dict[str, Any]] = []
    total_queries = 0

    for idx, seq_path in enumerate(sequences):
        seq = load_sequence(seq_path)
        T, N = seq["visibs"].shape
        logger.info(f"[{idx+1}/{len(sequences)}] {seq['_name']}: {T} frames, {N} points")

        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        if not queries:
            continue

        logger.info(f"  Found {len(queries)} re-entry queries (occ>={args.min_occ_length})")
        total_queries += len(queries)

        errors = compute_reentry_errors(seq, queries)
        all_results.extend(errors)

    logger.info(f"\nTotal re-entry queries: {total_queries}")
    logger.info(f"Total results: {len(all_results)}")

    if not all_results:
        logger.warning("No re-entry queries found. Try reducing --min-occ-length.")
        return

    # Stratified summary
    summary = stratified_summary(
        all_results,
        occ_thresholds=[10, 15, 20, 30, 50],
        cam_motion_thresholds=[0.0, 0.1, 0.3, 0.5, 1.0],
    )

    # Print key results
    print("\n" + "=" * 80)
    print("STAGE 0 RESULTS: 2D hold vs 3D world-state on PointOdyssey re-entry")
    print("=" * 80)
    print(f"Total re-entry queries: {total_queries}")
    print()

    for key in ["overall", "occ_gte_10", "occ_gte_20", "occ_gte_30",
                 "occ10_cam0.00", "occ10_cam0.10", "occ10_cam0.30",
                 "occ20_cam0.10", "occ20_cam0.30", "occ20_cam0.50"]:
        if key not in summary:
            continue
        s = summary[key]
        if s["count"] == 0:
            continue
        print(f"--- {key} (n={s['count']}) ---")
        for method in ["hold_2d_error_px", "hold_3d_reproj_error_px", "extrap_3d_reproj_error_px"]:
            mean = s.get(f"{method}_robust_mean", s.get(f"{method}_mean", float("nan")))
            med = s.get(f"{method}_median", float("nan"))
            s4 = s.get(f"{method}_lt4px", float("nan"))
            s16 = s.get(f"{method}_lt16px", float("nan"))
            label = method.replace("_error_px", "").replace("_", " ")
            print(f"  {label:28s}: robust_mean={mean:8.2f}  median={med:8.2f}  <4px={s4:5.1%}  <16px={s16:5.1%}")
        print()

    # Determine: should we continue?
    # Check the "hold_2d vs hold_3d" gap on the hardest subset
    hardest_key = None
    for key in ["occ20_cam0.30", "occ20_cam0.10", "occ_gte_20", "occ_gte_10"]:
        if key in summary and summary[key]["count"] >= 5:
            hardest_key = key
            break

    if hardest_key:
        s = summary[hardest_key]
        med_2d = s.get("hold_2d_error_px_median", float("inf"))
        med_3d = s.get("hold_3d_reproj_error_px_median", float("inf"))
        lt4_2d = s.get("hold_2d_error_px_lt4px", 0.0)
        lt4_3d = s.get("hold_3d_reproj_error_px_lt4px", 0.0)
        lt16_2d = s.get("hold_2d_error_px_lt16px", 0.0)
        lt16_3d = s.get("hold_3d_reproj_error_px_lt16px", 0.0)
        median_gap = med_2d - med_3d
        lt4_gap = lt4_3d - lt4_2d

        print(f"\n{'=' * 80}")
        print(f"DECISION ({hardest_key}, n={s['count']}):")
        print(f"  2D hold: median={med_2d:.2f} px, <4px={lt4_2d:.1%}, <16px={lt16_2d:.1%}")
        print(f"  3D hold: median={med_3d:.2f} px, <4px={lt4_3d:.1%}, <16px={lt16_3d:.1%}")
        print(f"  Median gap (2D - 3D):    {median_gap:.2f} px")
        print(f"  <4px gap (3D - 2D):      {lt4_gap:.1%}")
        # Decision based on median and success rate, not mean (which is skewed by outliers)
        if median_gap > 2.0 and lt4_gap > 0.05:
            print(f"  VERDICT: 3D world-state CLEARLY better. Continue to Stage 1.")
        elif median_gap > 0.5 or lt4_gap > 0.02:
            print(f"  VERDICT: 3D world-state marginally better. Try more data or tighter filtering.")
        else:
            print(f"  VERDICT: 3D world-state NOT better. Stop here.")
        print(f"{'=' * 80}")

    # Save
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(args),
            "total_queries": total_queries,
            "summary": summary,
            "per_query_results": all_results[:500],  # cap for file size
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
