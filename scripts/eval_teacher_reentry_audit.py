#!/usr/bin/env python3
"""Teacher re-entry audit: compare multiple teachers on re-entry metrics.

Reports both:
  - first-reentry proxy error / proxy AJ
  - TAPNext++-style post-reappearance AJ_RD over eligible events
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache
from utils.coords import find_first_reentry, pixel_l2_error
from utils.reentry_metrics import (
    DEFAULT_AJ_THRESHOLDS,
    DEFAULT_AJRD_D_MINS,
    DEFAULT_PROXY_THRESHOLDS,
    aggregate_reappearance_ajrd,
    compute_reentry_frame_proxy,
    compute_reappearance_segment_aj,
    eligible_reentry_events,
    summarize_reappearance_ajrd,
)


def _index_by_video(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_video: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        vid = str(rec["video_id"])
        if vid in by_video:
            raise ValueError(f"Duplicate video_id in cache: {vid}")
        by_video[vid] = rec
    return by_video


def _teacher_stats(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:
        return {"n": 0, "median_px": 0.0, "mean_px": 0.0, "p95_px": 0.0, "lt4px": 0.0, "lt8px": 0.0, "lt16px": 0.0}
    return {
        "n": int(errors.size),
        "median_px": round(float(np.median(errors)), 2),
        "mean_px": round(float(np.mean(errors)), 2),
        "p95_px": round(float(np.percentile(errors, 95)), 2),
        "lt4px": round(float(np.mean(errors < 4)), 4),
        "lt8px": round(float(np.mean(errors < 8)), 4),
        "lt16px": round(float(np.mean(errors < 16)), 4),
    }


def audit_teachers(teacher_caches: Dict[str, str], max_videos: int = 0) -> Dict[str, Any]:
    teachers: Dict[str, Dict[str, Any]] = {}
    for name, path in teacher_caches.items():
        payload = load_attempt0_cache(Path(path))
        records = payload["records"]
        if max_videos > 0:
            records = records[:max_videos]
        teachers[name] = _index_by_video(records)

    teacher_names = list(teachers.keys())
    ref_name = teacher_names[0]
    ref_records = list(teachers[ref_name].values())

    per_teacher_errors: Dict[str, List[float]] = {n: [] for n in teacher_names}
    per_teacher_long_errors: Dict[str, List[float]] = {n: [] for n in teacher_names}
    per_teacher_query_rows: Dict[str, List[Dict[str, Any]]] = {n: [] for n in teacher_names}
    oracle_errors: List[float] = []
    oracle_long_errors: List[float] = []
    oracle_teacher_usage: Dict[str, int] = {n: 0 for n in teacher_names}
    oracle_query_rows: List[Dict[str, Any]] = []
    oracle_ajrd_rows: List[Dict[str, Any]] = []
    oracle_ajrd256_rows: List[Dict[str, Any]] = []
    per_event: List[Dict[str, Any]] = []

    n_events = 0
    for r_ref in ref_records:
        vid = str(r_ref["video_id"])
        gt_vis = np.asarray(r_ref["gt_visibility"], dtype=bool)
        gt_tracks = np.asarray(r_ref["gt_tracks"], dtype=np.float32)
        qpts = np.asarray(r_ref["query_points"], dtype=np.float32)
        h, w = int(r_ref["original_size"][0]), int(r_ref["original_size"][1])

        if any(vid not in teachers[t] for t in teacher_names):
            missing = [t for t in teacher_names if vid not in teachers[t]]
            raise ValueError(f"Missing video_id={vid} in teacher caches: {missing}")

        N = gt_vis.shape[0]
        for qi in range(N):
            qt = int(round(float(qpts[qi, 0])))
            re = find_first_reentry(gt_vis[qi], qt)
            if re is None:
                continue

            t_re = int(re["reentry_frame"])
            occ_len = int(re["occ_length"])
            gt_yx = gt_tracks[qi, t_re]

            teacher_errors: Dict[str, float] = {}
            current_rows: Dict[str, Dict[str, Any]] = {}
            eligible_events = eligible_reentry_events(gt_vis[qi], qt)
            for tname in teacher_names:
                pred_tracks = np.asarray(teachers[tname][vid]["pred_tracks"], dtype=np.float32)
                pred_vis = np.asarray(teachers[tname][vid]["pred_visibility"], dtype=bool)[qi]
                pred_yx = pred_tracks[qi, t_re]
                err = float(pixel_l2_error(
                    pred_yx[None, :], gt_yx[None, :], h, w,
                    pred_fmt="yx_norm", gt_fmt="yx_norm"
                )[0])
                teacher_errors[tname] = err
                per_teacher_errors[tname].append(err)
                if occ_len >= 20:
                    per_teacher_long_errors[tname].append(err)

                proxy = compute_reentry_frame_proxy(
                    pred_tracks=pred_tracks[qi],
                    gt_tracks=gt_tracks[qi],
                    pred_visibility=pred_vis,
                    gt_visibility=gt_vis[qi],
                    event={
                        "reentry_frame": t_re,
                        "occ_length": occ_len,
                        "last_visible_t": int(re["last_visible_t"]),
                        "query_t": qt,
                    },
                    height=h,
                    width=w,
                    thresholds=DEFAULT_PROXY_THRESHOLDS,
                )
                ajrd_events: List[Dict[str, Any]] = []
                ajrd_events_256: List[Dict[str, Any]] = []
                for evt in eligible_events:
                    ajrd_evt = compute_reappearance_segment_aj(
                        pred_tracks=pred_tracks[qi],
                        gt_tracks=gt_tracks[qi],
                        pred_visibility=pred_vis,
                        gt_visibility=gt_vis[qi],
                        event={
                            "reentry_frame": int(evt["reentry_frame"]),
                            "occ_length": int(evt["occ_length"]),
                            "last_visible_t": int(evt["last_visible_t"]),
                            "query_t": qt,
                        },
                        height=h,
                        width=w,
                        thresholds=DEFAULT_PROXY_THRESHOLDS,
                    )
                    if ajrd_evt is not None:
                        ajrd_events.append(ajrd_evt)
                    # 256-space variant for TAPNext++ comparability
                    ajrd_evt_256 = compute_reappearance_segment_aj(
                        pred_tracks=pred_tracks[qi],
                        gt_tracks=gt_tracks[qi],
                        pred_visibility=pred_vis,
                        gt_visibility=gt_vis[qi],
                        event={
                            "reentry_frame": int(evt["reentry_frame"]),
                            "occ_length": int(evt["occ_length"]),
                            "last_visible_t": int(evt["last_visible_t"]),
                            "query_t": qt,
                        },
                        height=h,
                        width=w,
                        thresholds=DEFAULT_AJ_THRESHOLDS,
                        use_256_space=True,
                    )
                    if ajrd_evt_256 is not None:
                        ajrd_events_256.append(ajrd_evt_256)
                ajrd = summarize_reappearance_ajrd(ajrd_events, DEFAULT_AJRD_D_MINS)
                ajrd_256 = summarize_reappearance_ajrd(ajrd_events_256, DEFAULT_AJRD_D_MINS)
                row = {
                    "video_id": vid,
                    "query_idx": qi,
                    "query_t": qt,
                    "reentry_t": t_re,
                    "occ_length": occ_len,
                    "teacher": tname,
                    "error_px": round(err, 3),
                    "proxy_error_px": float(proxy["error_px"]) if proxy is not None else None,
                    "proxy_pred_visible": bool(proxy["pred_visible"]) if proxy is not None else None,
                    "proxy_gt_visible": bool(proxy["gt_visible"]) if proxy is not None else None,
                    "aj_proxy": float(proxy["aj_proxy"]) if proxy is not None else None,
                    "ajrd_summary": ajrd,
                    "ajrd_summary_256": ajrd_256,
                }
                current_rows[tname] = row
                per_teacher_query_rows[tname].append(row)

            best_teacher = min(teacher_errors, key=teacher_errors.get)

            # Oracle by true_AJ_RD (original resolution)
            teacher_ajrd = {n: current_rows[n].get("ajrd_summary", {}).get("aj_rd")
                           for n in teacher_names}
            valid_ajrd = {n: v for n, v in teacher_ajrd.items() if v is not None}
            best_teacher_ajrd = max(valid_ajrd, key=valid_ajrd.get) if valid_ajrd else best_teacher

            # Oracle by true_AJ_RD_256 (256-space, TAPNext++ comparable)
            teacher_ajrd_256 = {n: current_rows[n].get("ajrd_summary_256", {}).get("aj_rd")
                               for n in teacher_names}
            valid_ajrd_256 = {n: v for n, v in teacher_ajrd_256.items() if v is not None}
            best_teacher_ajrd_256 = max(valid_ajrd_256, key=valid_ajrd_256.get) if valid_ajrd_256 else best_teacher

            best_error = teacher_errors[best_teacher]
            oracle_errors.append(best_error)
            oracle_teacher_usage[best_teacher] += 1
            if occ_len >= 20:
                oracle_long_errors.append(best_error)
            oracle_query_rows.append(dict(current_rows[best_teacher]))

            # Oracle rows by true_AJ_RD (original resolution)
            oracle_ajrd_rows.append(dict(current_rows[best_teacher_ajrd]))
            # Oracle rows by true_AJ_RD_256 (256-space)
            oracle_ajrd256_rows.append(dict(current_rows[best_teacher_ajrd_256]))

            per_event.append({
                "video_id": vid,
                "query_idx": qi,
                "reentry_t": t_re,
                "occ_length": occ_len,
                "teacher_errors": {k: round(v, 3) for k, v in teacher_errors.items()},
                "best_teacher": best_teacher,
                "best_error": round(best_error, 3),
                "best_teacher_ajrd": current_rows[best_teacher]["ajrd_summary"]["aj_rd"] if current_rows[best_teacher]["ajrd_summary"] and current_rows[best_teacher]["ajrd_summary"].get("aj_rd") is not None else None,
            })
            n_events += 1

    teacher_medians = {n: float(np.median(per_teacher_errors[n])) for n in teacher_names}
    best_fixed_name = min(teacher_medians, key=teacher_medians.get)

    teacher_ajrd_agg = {
        tname: aggregate_reappearance_ajrd(per_teacher_query_rows[tname], DEFAULT_AJRD_D_MINS)
        for tname in teacher_names
    }
    oracle_ajrd_agg = aggregate_reappearance_ajrd(oracle_query_rows, DEFAULT_AJRD_D_MINS)
    oracle_ajrd_only_agg = aggregate_reappearance_ajrd(oracle_ajrd_rows, DEFAULT_AJRD_D_MINS)
    oracle_ajrd256_only_agg = aggregate_reappearance_ajrd(oracle_ajrd256_rows, DEFAULT_AJRD_D_MINS)
    fixed_ajrd_agg = aggregate_reappearance_ajrd(per_teacher_query_rows[best_fixed_name], DEFAULT_AJRD_D_MINS)

    fixed_proxy_vals = np.asarray(
        [float(row["aj_proxy"]) for row in per_teacher_query_rows[best_fixed_name] if row.get("aj_proxy") is not None],
        dtype=np.float32,
    )
    oracle_proxy_vals = np.asarray(
        [float(row["aj_proxy"]) for row in oracle_query_rows if row.get("aj_proxy") is not None],
        dtype=np.float32,
    )

    summary = {
        "n_events": n_events,
        "fixed_best_teacher": best_fixed_name,
        "fixed_best_median_px": round(teacher_medians[best_fixed_name], 2),
        "teacher_medians": {k: round(v, 2) for k, v in teacher_medians.items()},
        "teachers": {
            tname: {
                "overall": _teacher_stats(np.asarray(per_teacher_errors[tname], dtype=np.float32)),
                "long_occ_ge20": _teacher_stats(np.asarray(per_teacher_long_errors[tname], dtype=np.float32)),
                "aj_rd": teacher_ajrd_agg[tname],
            }
            for tname in teacher_names
        },
        "oracle": {
            "overall": _teacher_stats(np.asarray(oracle_errors, dtype=np.float32)),
            "long_occ_ge20": _teacher_stats(np.asarray(oracle_long_errors, dtype=np.float32)),
            "teacher_usage": oracle_teacher_usage,
            "aj_rd": oracle_ajrd_agg,
        },
        "proxy_comparison": {
            f"fixed_best_{best_fixed_name}": round(float(np.mean(fixed_proxy_vals)) if fixed_proxy_vals.size else 0.0, 4),
            "oracle_teacher_selection": round(float(np.mean(oracle_proxy_vals)) if oracle_proxy_vals.size else 0.0, 4),
            "delta": round(float(np.mean(oracle_proxy_vals) - np.mean(fixed_proxy_vals)) if (oracle_proxy_vals.size and fixed_proxy_vals.size) else 0.0, 4),
        },
        "aj_rd_comparison": {
            f"fixed_best_{best_fixed_name}": fixed_ajrd_agg["aj_rd"],
            "oracle_teacher_selection_by_min_error": oracle_ajrd_agg["aj_rd"],
            "oracle_teacher_selection_by_max_ajrd": oracle_ajrd_only_agg["aj_rd"],
            "oracle_teacher_selection_by_max_ajrd_256": oracle_ajrd256_only_agg["aj_rd"],
            "delta_min_error_vs_fixed": round(float(oracle_ajrd_agg["aj_rd"] - fixed_ajrd_agg["aj_rd"]), 4) if (oracle_ajrd_agg["aj_rd"] is not None and fixed_ajrd_agg["aj_rd"] is not None) else None,
            "delta_max_ajrd_vs_fixed": round(float(oracle_ajrd_only_agg["aj_rd"] - fixed_ajrd_agg["aj_rd"]), 4) if (oracle_ajrd_only_agg["aj_rd"] is not None and fixed_ajrd_agg["aj_rd"] is not None) else None,
            "delta_max_ajrd256_vs_fixed": round(float(oracle_ajrd256_only_agg["aj_rd"] - fixed_ajrd_agg["aj_rd"]), 4) if (oracle_ajrd256_only_agg["aj_rd"] is not None and fixed_ajrd_agg["aj_rd"] is not None) else None,
        },
        "per_event": per_event,
        "oracle_query_rows": oracle_query_rows,
        "teacher_query_rows": per_teacher_query_rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-caches", type=str, nargs="+", required=True, help="name=path pairs")
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

    print(f"Fixed best teacher: {results['fixed_best_teacher']} (median={results['fixed_best_median_px']}px)")
    print(f"Oracle teacher selection proxy (min-error): {results['proxy_comparison']['oracle_teacher_selection']:.4f}")
    print(f"Oracle teacher selection AJ_RD (min-error): {results['aj_rd_comparison']['oracle_teacher_selection_by_min_error']:.4f}")
    print(f"Oracle teacher selection AJ_RD (max-AJ_RD): {results['aj_rd_comparison']['oracle_teacher_selection_by_max_ajrd']:.4f}")
    print(f"Oracle teacher selection AJ_RD_256 (max-AJ_RD_256): {results['aj_rd_comparison']['oracle_teacher_selection_by_max_ajrd_256']:.4f}")
    print(f"Oracle gain (min-error): {results['aj_rd_comparison']['delta_min_error_vs_fixed']:.4f}")
    print(f"Oracle gain (max-AJ_RD): {results['aj_rd_comparison']['delta_max_ajrd_vs_fixed']:.4f}")
    print(f"Oracle gain (max-AJ_RD_256): {results['aj_rd_comparison']['delta_max_ajrd256_vs_fixed']:.4f}")
    print(f"Oracle teacher usage: {results['oracle']['teacher_usage']}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
