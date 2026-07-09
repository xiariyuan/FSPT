"""Compute re-entry failure mode statistics on DAVIS and RGB-Stacking.

For each re-entry event (GT visibility goes 0->1 after occlusion):
  - Visibility lag: how many frames after GT reentry_frame until base
    pred_visibility flips to True (lag=0 means base is already visible at
    the re-entry frame; lag>=1 means base is late).
  - False visibility: during the occlusion period (occ_start_t to
    reentry_frame-1), how many frames does base pred_visibility report
    True (false positive visibility).

Usage:
    python scripts/analyze_failure_mode_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.coords import find_reentry_events

DAVIS_OFFLINE = Path(
    "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt"
)
DAVIS_OVERRIDE = Path(
    "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt"
)

RGB_OFFLINE = Path(
    "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt"
)
RGB_OVERRIDE = Path(
    "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt"
)


def load_cache(path):
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=False)
    return cache["records"]


def analyze_records(records, tracker_name="base"):
    """Compute failure mode stats across all records."""
    total_events = 0
    lag_list = []
    false_vis_counts = []
    false_vis_rates = []
    occ_lengths = []

    for rec in records:
        gt_vis = rec["gt_visibility"]  # (N, T) bool
        pred_vis = rec["pred_visibility"]  # (N, T) bool
        query_points = rec["query_points"]  # (N, 3) [t, y, x]
        n_queries = gt_vis.shape[0]

        for qi in range(n_queries):
            query_t = int(query_points[qi, 0])
            events = find_reentry_events(gt_vis[qi], query_t)

            for ev in events:
                total_events += 1
                reentry_t = ev["reentry_frame"]
                occ_start = ev["occ_start_t"]
                occ_end = reentry_t - 1
                occ_len = ev["occ_length"]
                occ_lengths.append(occ_len)

                # --- Visibility lag ---
                # After GT says visible at reentry_t, how long until pred flips to True?
                T = gt_vis.shape[1]
                lag = 0
                for t in range(reentry_t, T):
                    if pred_vis[qi, t]:
                        lag = t - reentry_t
                        break
                else:
                    lag = T - reentry_t  # never recovers
                lag_list.append(lag)

                # --- False visibility during occlusion ---
                if occ_end >= occ_start:
                    occ_frames = pred_vis[qi, occ_start : occ_end + 1]
                    fv = int(occ_frames.sum())
                    fv_rate = fv / max(1, occ_end - occ_start + 1)
                else:
                    fv = 0
                    fv_rate = 0.0
                false_vis_counts.append(fv)
                false_vis_rates.append(fv_rate)

    lag_arr = np.array(lag_list)
    fv_arr = np.array(false_vis_counts)
    fvr_arr = np.array(false_vis_rates)
    occ_arr = np.array(occ_lengths)

    stats = {
        "tracker": tracker_name,
        "total_events": total_events,
        "occ_length_mean": float(np.mean(occ_arr)),
        "occ_length_median": float(np.median(occ_arr)),
        "visibility_lag_mean": float(np.mean(lag_arr)),
        "visibility_lag_median": float(np.median(lag_arr)),
        "visibility_lag_0_pct": float(np.mean(lag_arr == 0) * 100),
        "visibility_lag_1to2_pct": float(np.mean((lag_arr >= 1) & (lag_arr <= 2)) * 100),
        "visibility_lag_3to5_pct": float(np.mean((lag_arr >= 3) & (lag_arr <= 5)) * 100),
        "visibility_lag_6plus_pct": float(np.mean(lag_arr >= 6) * 100),
        "false_vis_mean_frames": float(np.mean(fv_arr)),
        "false_vis_rate_mean": float(np.mean(fvr_arr)),
        "false_vis_rate_median": float(np.median(fvr_arr)),
        "any_false_vis_pct": float(np.mean(fv_arr > 0) * 100),
    }
    return stats


def main():
    results = {}

    # --- DAVIS ---
    print("=== DAVIS strided/original ===")
    if DAVIS_OFFLINE.exists():
        print(f"Loading offline base: {DAVIS_OFFLINE}")
        records = load_cache(DAVIS_OFFLINE)
        stats_off = analyze_records(records, "CT3 offline (base)")
        results["davis_offline"] = stats_off
        print(json.dumps(stats_off, indent=2))
    else:
        print(f"NOT FOUND: {DAVIS_OFFLINE}")

    if DAVIS_OVERRIDE.exists():
        print(f"\nLoading override: {DAVIS_OVERRIDE}")
        records_ov = load_cache(DAVIS_OVERRIDE)
        stats_ov = analyze_records(records_ov, "vis4_gated288 (override)")
        results["davis_override"] = stats_ov
        print(json.dumps(stats_ov, indent=2))
    else:
        print(f"NOT FOUND: {DAVIS_OVERRIDE}")

    # --- RGB-Stacking ---
    print("\n=== RGB-Stacking full50 ===")
    if RGB_OFFLINE.exists():
        print(f"Loading offline base: {RGB_OFFLINE}")
        records_rgb = load_cache(RGB_OFFLINE)
        stats_rgb = analyze_records(records_rgb, "CT3 offline (base)")
        results["rgb_offline"] = stats_rgb
        print(json.dumps(stats_rgb, indent=2))
    else:
        print(f"NOT FOUND: {RGB_OFFLINE}")

    if RGB_OVERRIDE.exists():
        print(f"\nLoading override: {RGB_OVERRIDE}")
        records_rgb_ov = load_cache(RGB_OVERRIDE)
        stats_rgb_ov = analyze_records(records_rgb_ov, "vis4_gated288 (override)")
        results["rgb_override"] = stats_rgb_ov
        print(json.dumps(stats_rgb_ov, indent=2))
    else:
        print(f"NOT FOUND: {RGB_OVERRIDE}")

    # --- Save ---
    out_path = Path("docs/failure_mode_stats_2026-07-05.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
