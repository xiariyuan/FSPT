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
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics
from utils.coords import find_reentry_events

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_per_video_stability")
METHODS = {
    "fixed_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "global_b1_vis4": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt",
    "b2_mainline": "outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline/b2_predicted_mainline.pt",
    "b2_persist2": "outputs/paper_discovery_2026-06-27/teacher_expansion/b2_trigger_refinement/refine_k1_pre1_post9999_cnt1_pers2_tau192.pt",
    "b2_gt_oracle": "outputs/paper_discovery_2026-06-27/teacher_expansion/b2_localized_oracle/fixed_offline__override_vis4_gated288__pre0_post32.pt",
}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def standard_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    pred = torch.from_numpy(npy(record["pred_tracks"], np.float32))
    gt = torch.from_numpy(npy(record["gt_tracks"], np.float32))
    pv = torch.from_numpy(npy(record["pred_visibility"], bool))
    gv = torch.from_numpy(npy(record["gt_visibility"], bool))
    qp = torch.from_numpy(npy(record["query_points"], np.float32))
    m = compute_tapvid_metrics(pred, gt, pv, gv, qp, resolution=256, query_mode="strided")
    return {
        "AJ_256_pct": round(float(m.get("AJ", 0.0)) * 100.0, 4),
        "OA_256_pct": round(float(m.get("OA", 0.0)) * 100.0, 4),
        "delta_avg_256_pct": round(float(m.get("average_pts_within_thresh", 0.0)) * 100.0, 4),
    }


def reentry_visible_stats(record: Dict[str, Any]) -> Dict[str, Any]:
    pv = npy(record["pred_visibility"], bool)
    gv = npy(record["gt_visibility"], bool)
    qpts = npy(record["query_points"], np.float32)
    n, t_len = gv.shape
    first_pred = []
    misses = 0
    total = 0
    for qi in range(n):
        qt = int(round(float(qpts[qi, 0])))
        evs = find_reentry_events(gv[qi], qt)
        if not evs:
            continue
        rt = int(evs[0]["reentry_frame"])
        total += 1
        first_pred.append(bool(pv[qi, rt]))
        misses += int(not bool(pv[qi, rt]))
    return {
        "reentry_tracks": int(total),
        "reentry_pred_visible_rate": round(float(np.mean(first_pred)), 6) if first_pred else None,
        "reentry_missed_visible_rate": round(float(misses / max(total, 1)), 6) if total else None,
    }


def method_video_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    h, w = int(record["original_size"][0]), int(record["original_size"][1])
    re = compute_reentry_metrics(
        pred_tracks=npy(record["pred_tracks"], np.float32),
        gt_tracks=npy(record["gt_tracks"], np.float32),
        pred_vis=npy(record["pred_visibility"], bool),
        gt_vis=npy(record["gt_visibility"], bool),
        query_points=npy(record["query_points"], np.float32),
        height=h,
        width=w,
    )
    out = {
        "video_id": str(record["video_id"]),
        "n_queries": int(npy(record["gt_visibility"]).shape[0]),
        "n_reentry_queries": int(re.get("n_reentry_queries", 0)),
        "true_AJ_RD_256": re.get("true_AJ_RD_256"),
        "true_AJ_RD": re.get("true_AJ_RD"),
        "proxy": re.get("first_reentry_frame_proxy"),
        "dmin1_256": (re.get("aj_rd_by_dmin_256") or {}).get("1"),
        "dmin4_256": (re.get("aj_rd_by_dmin_256") or {}).get("4"),
        "dmin16_256": (re.get("aj_rd_by_dmin_256") or {}).get("16"),
    }
    out.update(standard_metrics(record))
    out.update(reentry_visible_stats(record))
    return out


def safe_delta(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    caches = {name: load_cache(path) for name, path in METHODS.items()}
    records_by_method = {name: payload["records"] for name, payload in caches.items()}
    n = len(next(iter(records_by_method.values())))
    for name, records in records_by_method.items():
        if len(records) != n:
            raise ValueError(f"record count mismatch for {name}")

    per_video: List[Dict[str, Any]] = []
    method_tables: Dict[str, List[Dict[str, Any]]] = {name: [] for name in METHODS}
    for idx in range(n):
        vid = str(next(iter(records_by_method.values()))[idx]["video_id"])
        row = {"video_id": vid}
        for name, records in records_by_method.items():
            m = method_video_metrics(records[idx])
            method_tables[name].append(m)
            row[name] = m
        # common deltas
        b2 = row["b2_mainline"]
        fixed = row["fixed_offline"]
        b1 = row["global_b1_vis4"]
        oracle = row["b2_gt_oracle"]
        p2 = row["b2_persist2"]
        row["deltas"] = {
            "b2_vs_fixed_AJ_RD_256": safe_delta(b2.get("true_AJ_RD_256"), fixed.get("true_AJ_RD_256")),
            "b2_vs_fixed_AJ_256": safe_delta(b2.get("AJ_256_pct"), fixed.get("AJ_256_pct")),
            "b2_vs_b1_AJ_RD_256": safe_delta(b2.get("true_AJ_RD_256"), b1.get("true_AJ_RD_256")),
            "b2_vs_b1_AJ_256": safe_delta(b2.get("AJ_256_pct"), b1.get("AJ_256_pct")),
            "b2_vs_oracle_AJ_RD_256": safe_delta(b2.get("true_AJ_RD_256"), oracle.get("true_AJ_RD_256")),
            "b2_vs_oracle_AJ_256": safe_delta(b2.get("AJ_256_pct"), oracle.get("AJ_256_pct")),
            "persist2_vs_b2_AJ_RD_256": safe_delta(p2.get("true_AJ_RD_256"), b2.get("true_AJ_RD_256")),
            "persist2_vs_b2_AJ_256": safe_delta(p2.get("AJ_256_pct"), b2.get("AJ_256_pct")),
        }
        per_video.append(row)

    def count_if(pred):
        return int(sum(1 for r in per_video if pred(r)))

    def mean_of(method: str, key: str):
        vals = [r[method].get(key) for r in per_video if r[method].get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    aggregate = {name: {
        "video_mean_AJ_RD_256": mean_of(name, "true_AJ_RD_256"),
        "video_mean_AJ_256_pct": mean_of(name, "AJ_256_pct"),
        "video_mean_OA_256_pct": mean_of(name, "OA_256_pct"),
        "video_mean_delta_avg_256_pct": mean_of(name, "delta_avg_256_pct"),
        "video_mean_reentry_miss": mean_of(name, "reentry_missed_visible_rate"),
    } for name in METHODS}

    stability = {
        "n_videos": len(per_video),
        "b2_improves_AJ_RD_vs_fixed": count_if(lambda r: (r["deltas"]["b2_vs_fixed_AJ_RD_256"] or -999) > 0),
        "b2_improves_AJ_RD_vs_fixed_by_0p01": count_if(lambda r: (r["deltas"]["b2_vs_fixed_AJ_RD_256"] or -999) >= 0.01),
        "b2_AJ_drop_vs_fixed_le_1p5": count_if(lambda r: (r["deltas"]["b2_vs_fixed_AJ_256"] is not None and r["deltas"]["b2_vs_fixed_AJ_256"] >= -1.5)),
        "b2_AJ_drop_vs_fixed_gt_3": count_if(lambda r: (r["deltas"]["b2_vs_fixed_AJ_256"] is not None and r["deltas"]["b2_vs_fixed_AJ_256"] < -3.0)),
        "b2_AJ_RD_within_0p005_of_b1": count_if(lambda r: (r["deltas"]["b2_vs_b1_AJ_RD_256"] is not None and r["deltas"]["b2_vs_b1_AJ_RD_256"] >= -0.005)),
        "b2_AJ_RD_loses_to_b1_by_gt_0p01": count_if(lambda r: (r["deltas"]["b2_vs_b1_AJ_RD_256"] is not None and r["deltas"]["b2_vs_b1_AJ_RD_256"] < -0.01)),
        "b2_AJ_beats_b1_by_10": count_if(lambda r: (r["deltas"]["b2_vs_b1_AJ_256"] is not None and r["deltas"]["b2_vs_b1_AJ_256"] >= 10.0)),
    }

    worst_aj_drop = sorted(per_video, key=lambda r: r["deltas"]["b2_vs_fixed_AJ_256"] if r["deltas"]["b2_vs_fixed_AJ_256"] is not None else 999)[:10]
    best_ajrd_gain = sorted(per_video, key=lambda r: r["deltas"]["b2_vs_fixed_AJ_RD_256"] if r["deltas"]["b2_vs_fixed_AJ_RD_256"] is not None else -999, reverse=True)[:10]
    biggest_oracle_gap = sorted(per_video, key=lambda r: r["deltas"]["b2_vs_oracle_AJ_RD_256"] if r["deltas"]["b2_vs_oracle_AJ_RD_256"] is not None else 999)[:10]

    summary = {
        "methods": METHODS,
        "aggregate_video_weighted": aggregate,
        "stability_counts": stability,
        "worst_b2_AJ_drop_vs_fixed": [{"video_id": r["video_id"], **r["deltas"], "fixed_AJ": r["fixed_offline"]["AJ_256_pct"], "b2_AJ": r["b2_mainline"]["AJ_256_pct"], "b2_AJ_RD": r["b2_mainline"]["true_AJ_RD_256"]} for r in worst_aj_drop],
        "best_b2_AJRD_gain_vs_fixed": [{"video_id": r["video_id"], **r["deltas"], "fixed_AJ_RD": r["fixed_offline"]["true_AJ_RD_256"], "b2_AJ_RD": r["b2_mainline"]["true_AJ_RD_256"], "b2_AJ": r["b2_mainline"]["AJ_256_pct"]} for r in best_ajrd_gain],
        "biggest_b2_gap_to_gt_oracle": [{"video_id": r["video_id"], **r["deltas"], "b2_AJ_RD": r["b2_mainline"]["true_AJ_RD_256"], "oracle_AJ_RD": r["b2_gt_oracle"]["true_AJ_RD_256"], "b2_AJ": r["b2_mainline"]["AJ_256_pct"], "oracle_AJ": r["b2_gt_oracle"]["AJ_256_pct"]} for r in biggest_oracle_gap],
        "per_video": per_video,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out_dir / "per_video_rows.jsonl").open("w") as f:
        for r in per_video:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ["aggregate_video_weighted", "stability_counts", "worst_b2_AJ_drop_vs_fixed", "best_b2_AJRD_gain_vs_fixed", "biggest_b2_gap_to_gt_oracle"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
