#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.attempt0_schema import load_attempt0_cache
from utils.coords import find_reentry_events

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_standard_visibility_audit")
METHODS = {
    "fixed_cotracker3_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "old_masked_median_majority": "outputs/uniform_three_teacher_ensemble_masked_median_2026-06-26.pt",
    "old3_all_median_gated144": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/old3_all_median_gated144.pt",
    "b1_all_median4_gated192": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/b1_all_median4_gated192.pt",
    "vis4_gated288": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt",
    "safe_router_ridge_m0p02": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_safe_router/b1_safe_router_ridge_m0p02.pt",
}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def avg(rows: List[Dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 6) if vals else None


def pct(x: float | None) -> float | None:
    return None if x is None else round(float(x) * 100.0, 4)


def vis_stats(pred_vis: np.ndarray, gt_vis: np.ndarray, qpts: np.ndarray) -> Dict[str, Any]:
    pred_vis = np.asarray(pred_vis, dtype=bool)
    gt_vis = np.asarray(gt_vis, dtype=bool)
    qpts = np.asarray(qpts, dtype=np.float32)
    n, t = gt_vis.shape
    mask = np.ones((n, t), dtype=bool)
    for i in range(n):
        qt = int(round(float(qpts[i, 0])))
        mask[i, max(0, min(t - 1, qt))] = False
    pred = pred_vis[mask]
    gt = gt_vis[mask]
    tp = int(np.sum(pred & gt)); fp = int(np.sum(pred & ~gt))
    fn = int(np.sum((~pred) & gt)); tn = int(np.sum((~pred) & (~gt)))
    re_pred, re_gt = [], []
    for i in range(n):
        qt = int(round(float(qpts[i, 0])))
        ev = find_reentry_events(gt_vis[i], qt)
        if ev:
            rt = int(ev[0]["reentry_frame"])
            re_pred.append(bool(pred_vis[i, rt])); re_gt.append(bool(gt_vis[i, rt]))
    if re_gt:
        rp = np.asarray(re_pred, dtype=bool); rg = np.asarray(re_gt, dtype=bool)
        re_miss = float(np.sum((~rp) & rg)) / max(int(np.sum(rg)), 1)
        re_prate = float(np.mean(rp))
    else:
        re_miss = None; re_prate = None
    total = max(int(pred.size), 1)
    return {
        "visible_precision": round(tp / max(tp + fp, 1), 6),
        "visible_recall": round(tp / max(tp + fn, 1), 6),
        "false_visible_rate_all": round(fp / total, 6),
        "missed_visible_rate_all": round(fn / total, 6),
        "visibility_accuracy": round((tp + tn) / total, 6),
        "pred_visible_rate": round(float(np.mean(pred)), 6),
        "gt_visible_rate": round(float(np.mean(gt)), 6),
        "reentry_pred_visible_rate": round(re_prate, 6) if re_prate is not None else None,
        "reentry_missed_visible_rate": round(re_miss, 6) if re_miss is not None else None,
        "reentry_n": len(re_gt),
    }


def audit(name: str, path: Path, max_videos: int = 0) -> Dict[str, Any]:
    payload = load_attempt0_cache(path)
    records = payload["records"][:max_videos] if max_videos else payload["records"]
    m_orig, m_256, vrows, per_video = [], [], [], []
    for r in records:
        vid = str(r["video_id"])
        h, w = int(r["original_size"][0]), int(r["original_size"][1])
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        qp = torch.from_numpy(npy(r["query_points"], np.float32))
        mo = compute_tapvid_metrics(pred, gt, pv, gv, qp, resolution=(h, w), query_mode="strided")
        m2 = compute_tapvid_metrics(pred, gt, pv, gv, qp, resolution=256, query_mode="strided")
        vs = vis_stats(npy(r["pred_visibility"], bool), npy(r["gt_visibility"], bool), npy(r["query_points"], np.float32))
        m_orig.append(mo); m_256.append(m2); vrows.append(vs)
        per_video.append({
            "video_id": vid,
            "AJ_256_pct": pct(m2.get("AJ")),
            "OA_256_pct": pct(m2.get("OA")),
            "delta_avg_256_pct": pct(m2.get("average_pts_within_thresh")),
            **vs,
        })
    return {
        "method": name,
        "cache_path": str(path),
        "n_videos": len(records),
        "standard_256": {
            "AJ_pct": pct(avg(m_256, "AJ")),
            "OA_pct": pct(avg(m_256, "OA")),
            "delta_avg_pct": pct(avg(m_256, "average_pts_within_thresh")),
            "median_error_px": avg(m_256, "median_error_px"),
            "avg_error_px": avg(m_256, "avg_error_px"),
        },
        "standard_original": {
            "AJ_pct": pct(avg(m_orig, "AJ")),
            "OA_pct": pct(avg(m_orig, "OA")),
            "delta_avg_pct": pct(avg(m_orig, "average_pts_within_thresh")),
            "median_error_px": avg(m_orig, "median_error_px"),
            "avg_error_px": avg(m_orig, "avg_error_px"),
        },
        "visibility": {
            "visible_precision": avg(vrows, "visible_precision"),
            "visible_recall": avg(vrows, "visible_recall"),
            "false_visible_rate_all": avg(vrows, "false_visible_rate_all"),
            "missed_visible_rate_all": avg(vrows, "missed_visible_rate_all"),
            "visibility_accuracy": avg(vrows, "visibility_accuracy"),
            "pred_visible_rate": avg(vrows, "pred_visible_rate"),
            "gt_visible_rate": avg(vrows, "gt_visible_rate"),
            "reentry_pred_visible_rate": avg(vrows, "reentry_pred_visible_rate"),
            "reentry_missed_visible_rate": avg(vrows, "reentry_missed_visible_rate"),
            "reentry_n": int(sum(int(v.get("reentry_n", 0)) for v in vrows)),
        },
        "per_video": per_video,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    details = []
    for name, pstr in METHODS.items():
        p = Path(pstr)
        if not p.exists():
            print(f"SKIP {name} missing {p}", flush=True); continue
        print(f"AUDIT {name}", flush=True)
        d = audit(name, p, args.max_videos)
        details.append(d)
        (out_dir / f"{name}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))
    compact = []
    for d in details:
        compact.append({
            "method": d["method"],
            "AJ_256_pct": d["standard_256"]["AJ_pct"],
            "OA_256_pct": d["standard_256"]["OA_pct"],
            "delta_avg_256_pct": d["standard_256"]["delta_avg_pct"],
            "AJ_orig_pct": d["standard_original"]["AJ_pct"],
            "OA_orig_pct": d["standard_original"]["OA_pct"],
            "delta_avg_orig_pct": d["standard_original"]["delta_avg_pct"],
            "visible_precision": d["visibility"]["visible_precision"],
            "visible_recall": d["visibility"]["visible_recall"],
            "false_visible_rate_all": d["visibility"]["false_visible_rate_all"],
            "missed_visible_rate_all": d["visibility"]["missed_visible_rate_all"],
            "reentry_pred_visible_rate": d["visibility"]["reentry_pred_visible_rate"],
            "reentry_missed_visible_rate": d["visibility"]["reentry_missed_visible_rate"],
        })
    (out_dir / "summary.json").write_text(json.dumps({"methods": compact, "details": details}, indent=2, ensure_ascii=False))
    print("=== STANDARD VISIBILITY SUMMARY ===")
    for r in compact:
        print(r, flush=True)


if __name__ == "__main__":
    main()
