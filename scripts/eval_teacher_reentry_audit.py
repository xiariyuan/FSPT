#!/usr/bin/env python3
"""Teacher re-entry audit: compare multiple teachers on re-entry error.

For each re-entry query, reports per-teacher error and identifies which
teacher is best. Supports oracle teacher selection (per-event best).

Teachers compared: cotracker3_offline, cotracker3_online, trackon2
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_first_reentry, pixel_l2_error
from utils.attempt0_schema import load_attempt0_cache


def _index_by_video(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_video: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        vid = str(rec["video_id"])
        if vid in by_video:
            raise ValueError(f"Duplicate video_id in cache: {vid}")
        by_video[vid] = rec
    return by_video


def audit_teachers(teacher_caches: Dict[str, str], max_videos: int = 0) -> Dict[str, Any]:
    """Audit multiple teachers on re-entry error.

    All caches must share the same GT data (same videos, same queries).
    """
    teachers = {}
    for name, path in teacher_caches.items():
        payload = load_attempt0_cache(Path(path))
        teachers[name] = payload["records"]
        if max_videos > 0:
            teachers[name] = teachers[name][:max_videos]
        teachers[name] = _index_by_video(teachers[name])

    teacher_names = list(teachers.keys())
    ref_name = teacher_names[0]
    ref_records = list(teachers[ref_name].values())

    # Per-teacher error accumulators
    per_teacher_errors: Dict[str, List[float]] = {n: [] for n in teacher_names}
    per_teacher_long_errors: Dict[str, List[float]] = {n: [] for n in teacher_names}
    per_teacher_very_long_errors: Dict[str, List[float]] = {n: [] for n in teacher_names}

    # Oracle teacher selection
    oracle_errors = []  # best teacher error per event
    oracle_long_errors = []
    oracle_teacher_usage: Dict[str, int] = {n: 0 for n in teacher_names}
    oracle_long_teacher_usage: Dict[str, int] = {n: 0 for n in teacher_names}

    # Fixed-best-teacher errors (for oracle gain computation)
    best_fixed_name = None  # determined after collecting all errors

    per_event = []
    n_events = 0

    for rec_idx in range(len(ref_records)):
        r_ref = ref_records[rec_idx]
        vid = r_ref["video_id"]
        gt_vis = np.asarray(r_ref["gt_visibility"], dtype=bool)
        gt_tracks = np.asarray(r_ref["gt_tracks"], dtype=np.float32)
        qpts = np.asarray(r_ref["query_points"], dtype=np.float32)
        h, w = int(r_ref["original_size"][0]), int(r_ref["original_size"][1])

        N, T = gt_vis.shape

        for qi in range(N):
            qt = int(round(float(qpts[qi, 0])))
            re = find_first_reentry(gt_vis[qi], qt)
            if re is None:
                continue

            t_re = re["reentry_frame"]
            occ_len = re["occ_length"]
            gt_yx = gt_tracks[qi, t_re]

            teacher_errors = {}
            for tname in teacher_names:
                if vid not in teachers[tname]:
                    raise ValueError(f"Teacher cache {tname} missing video_id={vid}")
                pred_tracks = np.asarray(teachers[tname][vid]["pred_tracks"], dtype=np.float32)
                pred_yx = pred_tracks[qi, t_re]
                err = float(pixel_l2_error(
                    pred_yx[None, :], gt_yx[None, :], h, w,
                    pred_fmt="yx_norm", gt_fmt="yx_norm"
                )[0])
                teacher_errors[tname] = err
                per_teacher_errors[tname].append(err)
                if occ_len >= 20:
                    per_teacher_long_errors[tname].append(err)
                if occ_len >= 50:
                    per_teacher_very_long_errors[tname].append(err)

            # Oracle: which teacher is best?
            best_teacher = min(teacher_errors, key=teacher_errors.get)
            best_error = teacher_errors[best_teacher]
            oracle_errors.append(best_error)
            oracle_teacher_usage[best_teacher] += 1
            if occ_len >= 20:
                oracle_long_errors.append(best_error)
                oracle_long_teacher_usage[best_teacher] += 1

            per_event.append({
                "video_id": vid,
                "query_idx": qi,
                "reentry_t": t_re,
                "occ_length": occ_len,
                "teacher_errors": {k: round(v, 3) for k, v in teacher_errors.items()},
                "best_teacher": best_teacher,
                "best_error": round(best_error, 3),
            })
            n_events += 1

    # Determine fixed-best teacher (lowest median error)
    teacher_medians = {
        n: float(np.median(per_teacher_errors[n])) for n in teacher_names
    }
    best_fixed_name = min(teacher_medians, key=teacher_medians.get)
    fixed_best_errors = np.array(per_teacher_errors[best_fixed_name])

    # Oracle gain = fixed_best - oracle
    oracle_gain = fixed_best_errors - np.array(oracle_errors)
    oracle_gain_long = (
        np.array(per_teacher_long_errors[best_fixed_name]) - np.array(oracle_long_errors)
        if oracle_long_errors else np.array([])
    )

    def _teacher_stats(errors: np.ndarray) -> dict:
        return {
            "n": len(errors),
            "median_px": round(float(np.median(errors)), 2),
            "mean_px": round(float(np.mean(errors)), 2),
            "p95_px": round(float(np.percentile(errors, 95)), 2),
            "lt4px": round(float(np.mean(errors < 4)), 4),
            "lt8px": round(float(np.mean(errors < 8)), 4),
            "lt16px": round(float(np.mean(errors < 16)), 4),
        }

    summary = {
        "n_events": n_events,
        "fixed_best_teacher": best_fixed_name,
        "fixed_best_median_px": round(teacher_medians[best_fixed_name], 2),
        "teacher_medians": {k: round(v, 2) for k, v in teacher_medians.items()},
        "teachers": {
            tname: {
                "overall": _teacher_stats(np.array(per_teacher_errors[tname])),
                "long_occ_ge20": _teacher_stats(np.array(per_teacher_long_errors[tname])),
            }
            for tname in teacher_names
        },
        "oracle": {
            "overall": _teacher_stats(np.array(oracle_errors)),
            "long_occ_ge20": _teacher_stats(np.array(oracle_long_errors)),
            "oracle_gain_vs_fixed_best": {
                "median_px": round(float(np.median(oracle_gain)), 2),
                "mean_px": round(float(np.mean(oracle_gain)), 2),
            },
            "teacher_usage": oracle_teacher_usage,
        },
    }

    # AJ_RD comparison: fixed best vs oracle
    thresholds = [1, 2, 4, 8, 16]
    fixed_aj = np.mean([
        np.mean([1.0 if e < thr else 0.0 for thr in thresholds])
        for e in fixed_best_errors
    ])
    oracle_aj = np.mean([
        np.mean([1.0 if e < thr else 0.0 for thr in thresholds])
        for e in oracle_errors
    ])
    summary["aj_rd_comparison"] = {
        f"fixed_best_{best_fixed_name}": round(fixed_aj, 4),
        "oracle_teacher_selection": round(oracle_aj, 4),
        "delta": round(oracle_aj - fixed_aj, 4),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-caches", type=str, nargs="+", required=True,
                        help="name=path pairs")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    cache_map = {}
    for entry in args.teacher_caches:
        name, path = entry.split("=", 1)
        cache_map[name] = path

    results = audit_teachers(cache_map, args.max_videos)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Fixed best teacher: {results['fixed_best_teacher']} "
          f"(median={results['fixed_best_median_px']}px)")
    ora_aj = results['aj_rd_comparison']['oracle_teacher_selection']
    fb_key = f"fixed_best_{results['fixed_best_teacher']}"
    fb_aj = results['aj_rd_comparison'][fb_key]
    print(f"Oracle teacher selection AJ_RD: {ora_aj:.4f}")
    print(f"Fixed best AJ_RD ({results['fixed_best_teacher']}): {fb_aj:.4f}")
    print(f"Oracle gain: {results['aj_rd_comparison']['delta']:.4f}")
    print(f"Oracle teacher usage: {results['oracle']['teacher_usage']}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
