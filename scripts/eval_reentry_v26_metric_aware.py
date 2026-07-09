#!/usr/bin/env python3
"""V26: Metric-aware selective ReEntry.

Adds a safety filter on top of V1+V22Q that drops proposed recovery frames
where the base coordinate is likely inaccurate (high base-override distance).
This targets the DAVIS strided AJ degradation: V1 opens visibility on some
frames where the base coordinate is wrong, causing false-visible AJ penalties.

Flow:
  V1 model → per-frame prob
  → raw mask: candidate & override_visible & prob >= v1_threshold
  → V26 safety filter: drop frames where base_override_dist_norm >= dist_thresh
  → V22Q interval cleaning on surviving frames
  → final visibility

Also computes the oracle upper bound: if we could perfectly know which frames
to keep (GT visible AND base coord accurate), what AJ/AJ_RD would we get?

Usage:
    python scripts/eval_reentry_v26_metric_aware.py \
        --base-cache outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt \
        --override-cache outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt \
        --v1-model outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1_dev0_6_clean.pt \
        --out-dir outputs/paper_discovery_2026-06-27/reentry_v26_metric_aware/davis_strided \
        --name v26_davis_strided
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from models.reentry_viscalibrator import build_model_from_checkpoint_payload
from utils.reentry_viscalibrator_features import (
    CandidateConfig,
    build_candidate_example,
    check_alignment,
    find_candidate_triggers,
    npy,
)
from scripts.eval_reentry_viscalibrator_v22_interval import (
    contiguous_segments,
    clean_recovery_mask,
    load_cache,
    run_ajrd,
)

FEATURE_NAMES = [
    "base_visible", "override_visible", "visibility_disagree",
    "candidate_window_mask", "relative_time_to_trigger", "time_since_query_norm",
    "is_after_query", "base_x_norm", "base_y_norm", "override_x_norm",
    "override_y_norm", "base_override_dist_norm", "base_speed_norm",
    "override_speed_norm", "speed_diff_norm", "base_acc_norm",
    "override_acc_norm", "base_invisible_run_norm", "base_visible_run_norm",
    "override_visible_run_norm", "override_invisible_run_norm",
    "base_border_dist_norm", "override_border_dist_norm",
    "override_future4_vis_rate", "override_future8_vis_rate",
    "base_past8_invis_rate", "base_override_xdiff_norm",
    "base_override_ydiff_norm",
]
DIST_IDX = FEATURE_NAMES.index("base_override_dist_norm")


def eval_standard(cache: Dict[str, Any], query_mode: str = "strided") -> Dict[str, Any]:
    """Compute both 256-space and original-resolution metrics."""
    aj_256, oa_256, da_256 = [], [], []
    aj_orig, oa_orig, da_orig = [], [], []
    q_total = 0
    for r in cache["records"]:
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        q = torch.from_numpy(npy(r["query_points"], np.float32))
        q_total += int(q.shape[0])
        orig = np.asarray(r.get("original_size", [480, 854]))
        res_orig = (int(orig[0]), int(orig[1]))
        m_256 = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode=query_mode)
        aj_256.append(float(m_256.get("AJ", 0.0)))
        oa_256.append(float(m_256.get("OA", 0.0)))
        da_256.append(float(m_256.get("average_pts_within_thresh", 0.0)))
        m_orig = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=res_orig, query_mode=query_mode)
        aj_orig.append(float(m_orig.get("AJ", 0.0)))
        oa_orig.append(float(m_orig.get("OA", 0.0)))
        da_orig.append(float(m_orig.get("average_pts_within_thresh", 0.0)))
    return {
        "AJ_256": round(float(np.mean(aj_256)) * 100.0, 4) if aj_256 else None,
        "OA_256": round(float(np.mean(oa_256)) * 100.0, 4) if oa_256 else None,
        "delta_avg_256": round(float(np.mean(da_256)) * 100.0, 4) if da_256 else None,
        "AJ_orig": round(float(np.mean(aj_orig)) * 100.0, 4) if aj_orig else None,
        "OA_orig": round(float(np.mean(oa_orig)) * 100.0, 4) if oa_orig else None,
        "delta_avg_orig": round(float(np.mean(da_orig)) * 100.0, 4) if da_orig else None,
        "n_records": len(cache["records"]),
        "n_queries": int(q_total),
    }


def apply_v26(
    base: Dict[str, Any],
    override: Dict[str, Any],
    v1_ckpt: Dict[str, Any],
    *,
    v1_threshold: float,
    dist_threshold: float,
    candidate_cfg: CandidateConfig,
    min_segment_len: int,
    edge_threshold: float,
    core_threshold: float,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v1_model = build_model_from_checkpoint_payload(v1_ckpt).to(device).eval()
    v1_mean = np.asarray(v1_ckpt["feature_mean"], dtype=np.float32)
    v1_std = np.asarray(v1_ckpt["feature_std"], dtype=np.float32)

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "v1_threshold": float(v1_threshold),
        "dist_threshold": float(dist_threshold),
        "candidate_windows": 0,
        "raw_recovery_frames": 0,
        "v26_dropped_frames": 0,
        "final_recovery_frames": 0,
        "per_video": [],
    }

    for br, orr in zip(base["records"], override["records"]):
        base_vis = npy(br["pred_visibility"], bool)
        override_vis = npy(orr["pred_visibility"], bool)
        qpts = npy(br["query_points"], np.float32)
        pred_vis = base_vis.copy()
        n_tracks, t_len = pred_vis.shape
        video_stats = {
            "video_id": str(br["video_id"]),
            "raw_recovery_frames": 0,
            "v26_dropped_frames": 0,
            "final_recovery_frames": 0,
        }

        for qi in range(n_tracks):
            query_t = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            triggers = find_candidate_triggers(base_vis[qi], override_vis[qi], query_t, candidate_cfg)

            for trigger_t in triggers:
                ex = build_candidate_example(br, orr, qi, int(trigger_t), candidate_cfg)
                xf = (ex["x"].astype(np.float32) - v1_mean[None, :]) / v1_std[None, :]
                xf[~ex["valid_mask"]] = 0.0
                xb = torch.from_numpy(xf[None]).to(device).float()
                valid = torch.from_numpy(ex["valid_mask"][None]).to(device).bool()
                with torch.no_grad():
                    prob = torch.sigmoid(v1_model(xb, valid_mask=valid))[0].detach().cpu().numpy()

                candidate_mask = ex["loss_mask"].astype(bool)
                candidate_mask &= ex["override_visibility"] > 0.5
                raw = candidate_mask & (prob >= float(v1_threshold))

                # V26 safety filter: drop frames where base-override distance is too large
                frame_dist = ex["x"][:, DIST_IDX]
                safe_mask = raw & (frame_dist < float(dist_threshold))
                v26_dropped = int(raw.sum()) - int(safe_mask.sum())

                # V22Q interval cleaning on surviving frames
                clean, _ = clean_recovery_mask(
                    safe_mask,
                    prob,
                    min_segment_len=min_segment_len,
                    edge_threshold=edge_threshold,
                    core_threshold=core_threshold,
                    keep_peak_if_trim_empty=True,
                )

                frame_indices = ex["frame_indices"]
                valid_frames = clean & (frame_indices >= 0) & (frame_indices < t_len)
                frames = frame_indices[valid_frames]
                if frames.size:
                    pred_vis[qi, frames] = True

                stats["candidate_windows"] += 1
                stats["raw_recovery_frames"] += int(raw.sum())
                stats["v26_dropped_frames"] += v26_dropped
                stats["final_recovery_frames"] += int(valid_frames.sum())
                video_stats["raw_recovery_frames"] += int(raw.sum())
                video_stats["v26_dropped_frames"] += v26_dropped
                video_stats["final_recovery_frames"] += int(valid_frames.sum())

        nr = dict(br)
        nr["pred_tracks"] = npy(br["pred_tracks"], np.float32).copy()
        nr["pred_visibility"] = pred_vis.astype(bool)
        records.append(nr)
        stats["per_video"].append(video_stats)

    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="V26 metric-aware selective ReEntry")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--override-cache", required=True)
    ap.add_argument("--v1-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="v26_metric_aware")
    ap.add_argument("--v1-threshold", type=float, nargs="+", default=[0.10])
    ap.add_argument("--dist-thresholds", type=float, nargs="+",
                    default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 999.0])
    ap.add_argument("--min-segment-len", type=int, default=2)
    ap.add_argument("--edge-threshold", type=float, default=0.15)
    ap.add_argument("--core-threshold", type=float, default=0.10)
    ap.add_argument("--context-before", type=int, default=16)
    ap.add_argument("--context-after", type=int, default=16)
    ap.add_argument("--candidate-pre", type=int, default=1)
    ap.add_argument("--candidate-post", type=int, default=16)
    ap.add_argument("--trigger-k", type=int, default=1)
    ap.add_argument("--trigger-persist", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_cache(args.base_cache)
    override = load_cache(args.override_cache)
    check_alignment(base, override, "override")
    v1_ckpt = torch.load(args.v1_model, map_location="cpu", weights_only=False)
    cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )

    results = []
    for v1_thr in args.v1_threshold:
     for dist_thr in args.dist_thresholds:
        label_v1 = f"v1{v1_thr:.2f}"
        label_d = f"dist{dist_thr:.2f}" if dist_thr < 900 else "distinf"
        name = f"{args.name}_{label_v1}_{label_d}"
        print(f"\n=== V26 v1_threshold={v1_thr:.2f} dist_threshold={dist_thr:.2f} ===")

        records, stats = apply_v26(
            base, override, v1_ckpt,
            v1_threshold=float(v1_thr),
            dist_threshold=float(dist_thr),
            candidate_cfg=cfg,
            min_segment_len=int(args.min_segment_len),
            edge_threshold=float(args.edge_threshold),
            core_threshold=float(args.core_threshold),
            device=torch.device(args.device),
        )

        payload = dict(base)
        payload["records"] = records
        payload["model_name"] = name

        cache_path = out_dir / f"{name}.pt"
        ajrd_json = out_dir / f"{name}_ajrd.json"
        torch.save(payload, cache_path)

        ajrd = run_ajrd(cache_path, ajrd_json)
        std = eval_standard(payload)

        row = {
            "v1_threshold": v1_thr,
            "dist_threshold": dist_thr,
            "label": f"{label_v1}_{label_d}",
            "AJ_256": std["AJ_256"],
            "OA_256": std["OA_256"],
            "delta_avg_256": std["delta_avg_256"],
            "AJ_orig": std["AJ_orig"],
            "OA_orig": std["OA_orig"],
            "delta_avg_orig": std["delta_avg_orig"],
            "AJ_RD_256": ajrd.get("true_AJ_RD_256"),
            "AJ_RD": ajrd.get("true_AJ_RD"),
            "raw_recovery_frames": stats["raw_recovery_frames"],
            "v26_dropped_frames": stats["v26_dropped_frames"],
            "final_recovery_frames": stats["final_recovery_frames"],
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    summary_path = out_dir / f"{args.name}_sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSweep summary saved to {summary_path}")
    print("\n=== Sweep Summary (original-resolution) ===")
    print(f"{'v1_thr':>8} {'dist':>8} {'AJ':>8} {'OA':>8} {'AJ_RD':>8} {'AJ_RD256':>8} {'drop':>6} {'kept':>6}")
    for r in results:
        print(f"{r['v1_threshold']:>8.2f} {r['dist_threshold']:>8.2f} {r['AJ_orig']:>8.2f} {r['OA_orig']:>8.2f} {r['AJ_RD']:>8.4f} {r['AJ_RD_256']:>8.4f} {r['v26_dropped_frames']:>6} {r['final_recovery_frames']:>6}")


if __name__ == "__main__":
    main()
