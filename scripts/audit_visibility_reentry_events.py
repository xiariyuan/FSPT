#!/usr/bin/env python3
"""Audit visibility and re-entry events across all three baseline caches.

Produces:
  - n_reappearance_events per video and total
  - occ length histogram
  - Cross-baseline consistency: do all caches see the same GT events?
  - Per-model visibility agreement at re-entry frames
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events
from utils.attempt0_schema import load_attempt0_cache


def audit_visibility_reentry(
    cache_paths: Dict[str, str],
    max_videos: int = 0,
) -> Dict[str, Any]:
    """Audit re-entry events across all caches.

    All caches must share the same GT data (same videos, same queries).
    """
    caches = {}
    for name, path in cache_paths.items():
        payload = load_attempt0_cache(Path(path))
        caches[name] = payload["records"]
        if max_videos > 0:
            caches[name] = caches[name][:max_videos]

    # Use first cache as GT reference, but align all caches by video_id.
    ref_name = list(caches.keys())[0]
    ref_records = caches[ref_name]
    cache_by_video = {
        name: {str(rec["video_id"]): rec for rec in records}
        for name, records in caches.items()
    }
    ref_video_ids = [str(rec["video_id"]) for rec in ref_records]
    for name, by_video in cache_by_video.items():
        missing = [vid for vid in ref_video_ids if vid not in by_video]
        if missing:
            raise ValueError(f"Cache {name} is missing videos present in reference cache: {missing[:5]}")

    all_events = []
    per_video = []
    occ_hist = []
    total_queries = 0
    total_reentry = 0

    for rec_idx in range(len(ref_records)):
        r = ref_records[rec_idx]
        vid = r["video_id"]
        gt_vis = np.asarray(r["gt_visibility"], dtype=bool)
        qpts = np.asarray(r["query_points"], dtype=np.float32)
        N, T = gt_vis.shape

        vid_events = 0
        vid_occ_lengths = []

        for qi in range(N):
            qt = int(round(float(qpts[qi, 0])))
            total_queries += 1
            events = find_reentry_events(gt_vis[qi], qt)
            if events:
                total_reentry += 1
                vid_events += 1
                for evt in events:
                    occ_hist.append(evt["occ_length"])
                    vid_occ_lengths.append(evt["occ_length"])

        per_video.append({
            "video_id": vid,
            "n_queries": N,
            "n_reentry_events": vid_events,
            "occ_lengths": vid_occ_lengths,
        })

    # Occ length histogram
    occ_arr = np.array(occ_hist)
    occ_buckets = {
        "1-4": int(np.sum((occ_arr >= 1) & (occ_arr <= 4))),
        "5-9": int(np.sum((occ_arr >= 5) & (occ_arr <= 9))),
        "10-19": int(np.sum((occ_arr >= 10) & (occ_arr <= 19))),
        "20-49": int(np.sum((occ_arr >= 20) & (occ_arr <= 49))),
        "50-99": int(np.sum((occ_arr >= 50) & (occ_arr <= 99))),
        "100+": int(np.sum(occ_arr >= 100)),
    }

    # Cross-baseline visibility agreement at re-entry frames
    # For the first cache's re-entry events, check what other models predict
    cross_model_agreement = {}
    if len(caches) > 1:
        ref_name = list(caches.keys())[0]
        for other_name in list(caches.keys())[1:]:
            agree_count = 0
            disagree_count = 0
            n_checked = 0
            for rec_idx in range(len(ref_records)):
                r_ref = ref_records[rec_idx]
                vid = str(r_ref["video_id"])
                r_other = cache_by_video[other_name][vid]
                gt_vis = np.asarray(r_ref["gt_visibility"], dtype=bool)
                other_pvis = np.asarray(r_other["pred_visibility"], dtype=bool)
                qpts = np.asarray(r_ref["query_points"], dtype=np.float32)
                N, T = gt_vis.shape

                for qi in range(N):
                    qt = int(round(float(qpts[qi, 0])))
                    events = find_reentry_events(gt_vis[qi], qt)
                    for evt in events:
                        t_re = evt["reentry_frame"]
                        n_checked += 1
                        if other_pvis[qi, t_re] == gt_vis[qi, t_re]:
                            agree_count += 1
                        else:
                            disagree_count += 1

            cross_model_agreement[f"{ref_name}_vs_{other_name}"] = {
                "n_checked": n_checked,
                "agree": agree_count,
                "disagree": disagree_count,
                "agree_rate": round(agree_count / max(n_checked, 1), 4),
            }

    return {
        "gt_source": ref_name,
        "n_videos": len(per_video),
        "n_total_queries": total_queries,
        "n_total_reentry_queries": total_reentry,
        "n_total_reentry_events": len(occ_hist),
        "occ_length_buckets": occ_buckets,
        "occ_length_stats": {
            "min": int(occ_arr.min()) if len(occ_arr) else 0,
            "median": float(np.median(occ_arr)) if len(occ_arr) else 0,
            "mean": float(np.mean(occ_arr)) if len(occ_arr) else 0,
            "max": int(occ_arr.max()) if len(occ_arr) else 0,
        },
        "per_video": per_video,
        "cross_model_visibility_agreement": cross_model_agreement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-paths", type=str, nargs="+", required=True,
                        help="Cache paths as name=path pairs, e.g. ct_off=caches/ct_off.pt")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    cache_map = {}
    for entry in args.cache_paths:
        name, path = entry.split("=", 1)
        cache_map[name] = path

    results = audit_visibility_reentry(cache_map, args.max_videos)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"n_total_queries={results['n_total_queries']}, "
          f"n_reentry_queries={results['n_total_reentry_queries']}, "
          f"n_reentry_events={results['n_total_reentry_events']}")
    print(f"Occ buckets: {results['occ_length_buckets']}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
