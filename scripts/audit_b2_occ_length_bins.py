#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reentry_metrics import compute_reappearance_segment_aj, eligible_reentry_events

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_occ_length_bins")
METHODS = {
    "fixed_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "global_b1_vis4": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt",
    "b2_mainline": "outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline/b2_predicted_mainline.pt",
    "b2_persist2": "outputs/paper_discovery_2026-06-27/teacher_expansion/b2_trigger_refinement/refine_k1_pre1_post9999_cnt1_pers2_tau192.pt",
    "b2_gt_oracle": "outputs/paper_discovery_2026-06-27/teacher_expansion/b2_localized_oracle/fixed_offline__override_vis4_gated288__pre0_post32.pt",
}
BINS: List[Tuple[str, int, int | None]] = [
    ("occ_1_4", 1, 4),
    ("occ_5_8", 5, 8),
    ("occ_9_16", 9, 16),
    ("occ_17_32", 17, 32),
    ("occ_33_plus", 33, None),
]


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def bin_name(occ_len: int) -> str:
    for name, lo, hi in BINS:
        if int(occ_len) >= lo and (hi is None or int(occ_len) <= hi):
            return name
    return "other"


def event_rows_for_method(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in payload["records"]:
        h, w = int(r["original_size"][0]), int(r["original_size"][1])
        pred = npy(r["pred_tracks"], np.float32)
        gt = npy(r["gt_tracks"], np.float32)
        pv = npy(r["pred_visibility"], bool)
        gv = npy(r["gt_visibility"], bool)
        qpts = npy(r["query_points"], np.float32)
        for qi in range(gt.shape[0]):
            qt = int(round(float(qpts[qi, 0])))
            for ev in eligible_reentry_events(gv[qi], qt):
                ev = dict(ev)
                ev["query_t"] = qt
                aj = compute_reappearance_segment_aj(
                    pred_tracks=pred[qi],
                    gt_tracks=gt[qi],
                    pred_visibility=pv[qi],
                    gt_visibility=gv[qi],
                    event=ev,
                    height=h,
                    width=w,
                    use_256_space=True,
                )
                if aj is None:
                    continue
                rt = int(ev["reentry_frame"])
                rows.append({
                    "video_id": str(r["video_id"]),
                    "query_idx": int(qi),
                    "query_t": int(qt),
                    "reentry_t": rt,
                    "occ_length": int(ev["occ_length"]),
                    "bin": bin_name(int(ev["occ_length"])),
                    "aj_segment_256": float(aj["aj_segment"]),
                    "pred_visible_at_reentry": bool(pv[qi, rt]),
                    "gt_visible_at_reentry": bool(gv[qi, rt]),
                })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, _, _ in BINS:
        br = [r for r in rows if r["bin"] == name]
        vals = [float(r["aj_segment_256"]) for r in br]
        predv = [bool(r["pred_visible_at_reentry"]) for r in br]
        out[name] = {
            "n_events": int(len(br)),
            "mean_AJ_segment_256": round(float(np.mean(vals)), 6) if vals else None,
            "median_AJ_segment_256": round(float(np.median(vals)), 6) if vals else None,
            "reentry_pred_visible_rate": round(float(np.mean(predv)), 6) if predv else None,
        }
    vals_all = [float(r["aj_segment_256"]) for r in rows]
    out["all"] = {
        "n_events": int(len(rows)),
        "mean_AJ_segment_256": round(float(np.mean(vals_all)), 6) if vals_all else None,
    }
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
    rows_by_method: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for name, path in METHODS.items():
        payload = load(path)
        rows = event_rows_for_method(payload)
        rows_by_method[name] = rows
        summaries[name] = summarize(rows)
    # bucket comparison relative to fixed/global B1/oracle
    comparison: Dict[str, Dict[str, Any]] = {}
    for bname, _, _ in BINS:
        comparison[bname] = {}
        for method in METHODS:
            v = summaries[method][bname]["mean_AJ_segment_256"]
            comparison[bname][method] = v
        comparison[bname]["b2_vs_fixed"] = safe_delta(comparison[bname]["b2_mainline"], comparison[bname]["fixed_offline"])
        comparison[bname]["b2_vs_b1"] = safe_delta(comparison[bname]["b2_mainline"], comparison[bname]["global_b1_vis4"])
        comparison[bname]["b2_vs_oracle"] = safe_delta(comparison[bname]["b2_mainline"], comparison[bname]["b2_gt_oracle"])
        comparison[bname]["persist2_vs_b2"] = safe_delta(comparison[bname]["b2_persist2"], comparison[bname]["b2_mainline"])
        comparison[bname]["n_events"] = summaries["fixed_offline"][bname]["n_events"]
    summary = {
        "methods": METHODS,
        "bins": BINS,
        "summaries": summaries,
        "comparison": comparison,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for method, rows in rows_by_method.items():
        with (out_dir / f"{method}_event_rows.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"comparison": comparison, "method_summaries": summaries}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
