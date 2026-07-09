#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sweep_b2_reentry_localized_oracle import (
    npy,
    load_payload,
    check_alignment,
    eval_ajrd,
    eval_standard,
    visibility_stats,
)

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_localized_predicted")
BASES = {
    "fixed_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "old3_gated144": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/old3_all_median_gated144.pt",
}
OVERRIDES = {
    "vis4_gated288": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt",
    "all4_gated192": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/b1_all_median4_gated192.pt",
}
KS = [1, 2, 4, 8, 16]
PRES = [0, 1]
POSTS = [8, 16, 32, 9999]
MODES = ["base_inv_over_vis", "base_inv_over_rise", "base_inv_over_not_base"]


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = t - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def should_trigger(base_v: np.ndarray, over_v: np.ndarray, t: int, k: int, mode: str) -> bool:
    if t <= 0:
        return False
    if invisible_run_before(base_v, t) < int(k):
        return False
    if mode == "base_inv_over_vis":
        return bool(over_v[t])
    if mode == "base_inv_over_rise":
        return bool(over_v[t]) and not bool(over_v[t - 1])
    if mode == "base_inv_over_not_base":
        return bool(over_v[t]) and not bool(base_v[t])
    raise ValueError(mode)


def predicted_records(base: Dict[str, Any], over: Dict[str, Any], k: int, pre: int, post: int, mode: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = []
    total_triggers = 0
    tracks_with_trigger = 0
    for b, o in zip(base["records"], over["records"]):
        pred_tracks = npy(b["pred_tracks"], np.float32).copy()
        pred_vis = npy(b["pred_visibility"], bool).copy()
        base_vis = pred_vis.copy()
        over_tracks = npy(o["pred_tracks"], np.float32)
        over_vis = npy(o["pred_visibility"], bool)
        qpts = npy(b["query_points"], np.float32)
        n, t_len = pred_vis.shape
        for qi in range(n):
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            mask = np.zeros(t_len, dtype=bool)
            t = qt + 1
            q_triggers = 0
            while t < t_len:
                if should_trigger(base_vis[qi], over_vis[qi], t, k, mode):
                    lo = max(0, t - int(pre))
                    hi = t_len if int(post) >= 9999 else min(t_len, t + int(post) + 1)
                    mask[lo:hi] = True
                    q_triggers += 1
                    t = hi
                else:
                    t += 1
            if q_triggers > 0:
                total_triggers += q_triggers
                tracks_with_trigger += 1
                pred_tracks[qi, mask] = over_tracks[qi, mask]
                pred_vis[qi, mask] = over_vis[qi, mask]
        r = dict(b)
        r["pred_tracks"] = pred_tracks.astype(np.float32)
        r["pred_visibility"] = pred_vis.astype(bool)
        r["b2_pred_k"] = int(k)
        r["b2_pred_pre"] = int(pre)
        r["b2_pred_post"] = int(post)
        r["b2_pred_mode"] = mode
        records.append(r)
    return records, {"total_triggers": int(total_triggers), "tracks_with_trigger": int(tracks_with_trigger)}


def run_config(out_dir: Path, base_name: str, base: Dict[str, Any], over_name: str, over: Dict[str, Any], k: int, pre: int, post: int, mode: str) -> Dict[str, Any]:
    name = f"{base_name}__override_{over_name}__{mode}__k{k}_pre{pre}_post{post}"
    records, trig = predicted_records(base, over, k=k, pre=pre, post=post, mode=mode)
    payload = dict(base)
    payload["model_name"] = "b2_pred_window_" + name
    payload["b2_base"] = base_name
    payload["b2_override"] = over_name
    payload["b2_pred_k"] = int(k)
    payload["b2_pred_pre"] = int(pre)
    payload["b2_pred_post"] = int(post)
    payload["b2_pred_mode"] = mode
    payload["records"] = records
    cp = out_dir / f"{name}.pt"
    jp = out_dir / f"{name}_ajrd.json"
    torch.save(payload, cp)
    ajrd = eval_ajrd(cp, jp)
    std = eval_standard(records)
    vis = visibility_stats(records)
    bd = ajrd.get("aj_rd_by_dmin_256") or {}
    row = {
        "name": name,
        "base": base_name,
        "override": over_name,
        "mode": mode,
        "k": int(k),
        "pre": int(pre),
        "post": int(post),
        "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"),
        "true_AJ_RD": ajrd.get("true_AJ_RD"),
        "proxy": ajrd.get("first_reentry_frame_proxy"),
        "dmin1_256": bd.get("1"),
        "dmin4_256": bd.get("4"),
        "dmin16_256": bd.get("16"),
        **std,
        **vis,
        **trig,
    }
    row["delta_AJ_RD_vs_fixed"] = round(float(row.get("true_AJ_RD_256") or 0.0) - 0.5546, 4)
    row["delta_AJ_RD_vs_old3"] = round(float(row.get("true_AJ_RD_256") or 0.0) - 0.6189, 4)
    row["delta_AJ_RD_vs_vis4"] = round(float(row.get("true_AJ_RD_256") or 0.0) - 0.6279, 4)
    row["delta_AJ256_vs_fixed"] = round(float(row.get("AJ_256_pct") or 0.0) - 70.051, 4)
    row["delta_AJ256_vs_vis4"] = round(float(row.get("AJ_256_pct") or 0.0) - 47.4046, 4)
    return row


def pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        dominated = False
        for s in rows:
            if s is r:
                continue
            if (float(s.get("true_AJ_RD_256") or -1) >= float(r.get("true_AJ_RD_256") or -1)
                and float(s.get("AJ_256_pct") or -1) >= float(r.get("AJ_256_pct") or -1)
                and (float(s.get("true_AJ_RD_256") or -1) > float(r.get("true_AJ_RD_256") or -1)
                     or float(s.get("AJ_256_pct") or -1) > float(r.get("AJ_256_pct") or -1))):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: (float(r.get("true_AJ_RD_256") or 0), float(r.get("AJ_256_pct") or 0)), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--max-configs", type=int, default=0)
    ap.add_argument("--bases", default="fixed_offline")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_names = [x.strip() for x in args.bases.split(",") if x.strip()]
    bases = {k: load_payload(BASES[k]) for k in base_names}
    overs = {k: load_payload(v) for k, v in OVERRIDES.items()}
    for bname, b in bases.items():
        for oname, o in overs.items():
            check_alignment(b, o)
    configs = []
    for bname in bases:
        for oname in OVERRIDES:
            for mode in MODES:
                for k in KS:
                    for pre in PRES:
                        for post in POSTS:
                            configs.append((bname, oname, k, pre, post, mode))
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    rows = []
    for i, (bname, oname, k, pre, post, mode) in enumerate(configs, 1):
        print(f"RUN {i}/{len(configs)} {bname} {oname} {mode} k={k} pre={pre} post={post}", flush=True)
        row = run_config(out_dir, bname, bases[bname], oname, overs[oname], k, pre, post, mode)
        rows.append(row)
        print("ROW", json.dumps(row, ensure_ascii=False), flush=True)
    best = sorted(rows, key=lambda r: r.get("true_AJ_RD_256") if r.get("true_AJ_RD_256") is not None else -1, reverse=True)
    pareto = pareto_front(rows)
    summary = {
        "references": {
            "fixed_AJ_RD_256": 0.5546,
            "old3_AJ_RD_256": 0.6189,
            "vis4_AJ_RD_256": 0.6279,
            "b2_gt_oracle_best_AJ_RD_256": 0.6290,
            "fixed_AJ_256_pct": 70.051,
            "vis4_AJ_256_pct": 47.4046,
        },
        "n_configs": len(rows),
        "best_by_AJ_RD_256": best[:30],
        "pareto": pareto,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== BEST ===")
    print(json.dumps(best[:10], indent=2, ensure_ascii=False), flush=True)
    print("=== PARETO ===")
    print(json.dumps(pareto[:20], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
