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

from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_localized_oracle")
BASES = {
    "fixed_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "old3_gated144": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/old3_all_median_gated144.pt",
}
OVERRIDES = {
    "vis4_gated288": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt",
    "all4_gated192": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/b1_all_median4_gated192.pt",
}
PRES = [0, 1, 2, 4]
POSTS = [4, 8, 16, 32, 9999]


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_payload(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def check_alignment(base: Dict[str, Any], over: Dict[str, Any]) -> None:
    br = base["records"]
    orr = over["records"]
    if len(br) != len(orr):
        raise ValueError(f"record count mismatch: {len(br)} vs {len(orr)}")
    for i, (b, o) in enumerate(zip(br, orr)):
        if str(b["video_id"]) != str(o["video_id"]):
            raise ValueError(f"video_id mismatch at {i}: {b['video_id']} vs {o['video_id']}")
        for key in ["query_points", "gt_tracks", "gt_visibility", "original_size"]:
            if not np.allclose(npy(b[key]), npy(o[key]), atol=1e-6, rtol=1e-6):
                raise ValueError(f"alignment mismatch at {i}, key={key}")


def localized_records(base: Dict[str, Any], over: Dict[str, Any], pre: int, post: int) -> List[Dict[str, Any]]:
    records = []
    for b, o in zip(base["records"], over["records"]):
        pred_tracks = npy(b["pred_tracks"], np.float32).copy()
        pred_vis = npy(b["pred_visibility"], bool).copy()
        over_tracks = npy(o["pred_tracks"], np.float32)
        over_vis = npy(o["pred_visibility"], bool)
        gt_vis = npy(b["gt_visibility"], bool)
        qpts = npy(b["query_points"], np.float32)
        n, t = pred_vis.shape
        for qi in range(n):
            qt = int(round(float(qpts[qi, 0])))
            events = find_reentry_events(gt_vis[qi], qt)
            if not events:
                continue
            mask = np.zeros(t, dtype=bool)
            for ev in events:
                rt = int(ev["reentry_frame"])
                lo = max(0, rt - int(pre))
                hi = t if int(post) >= 9999 else min(t, rt + int(post) + 1)
                mask[lo:hi] = True
            pred_tracks[qi, mask] = over_tracks[qi, mask]
            pred_vis[qi, mask] = over_vis[qi, mask]
        r = dict(b)
        r["pred_tracks"] = pred_tracks.astype(np.float32)
        r["pred_visibility"] = pred_vis.astype(bool)
        r["b2_pre"] = int(pre)
        r["b2_post"] = int(post)
        records.append(r)
    return records


def eval_ajrd(cache_path: Path, json_path: Path) -> Dict[str, Any]:
    subprocess.run([
        sys.executable,
        "scripts/eval_aj_rd_from_cache.py",
        "--cache-path",
        str(cache_path),
        "--output-json",
        str(json_path),
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(json_path))


def mean(vals: List[float]) -> float | None:
    return round(float(np.mean(vals)), 6) if vals else None


def pct(x: float | None) -> float | None:
    return None if x is None else round(float(x) * 100.0, 4)


def visibility_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_prec, all_rec, all_fp, all_miss, re_pred_rate, re_miss = [], [], [], [], [], []
    for r in records:
        pv = npy(r["pred_visibility"], bool)
        gv = npy(r["gt_visibility"], bool)
        qpts = npy(r["query_points"], np.float32)
        n, t = gv.shape
        mask = np.ones((n, t), dtype=bool)
        for i in range(n):
            qt = max(0, min(t - 1, int(round(float(qpts[i, 0])))))
            mask[i, qt] = False
        pred = pv[mask]
        gt = gv[mask]
        tp = int(np.sum(pred & gt))
        fp = int(np.sum(pred & ~gt))
        fn = int(np.sum((~pred) & gt))
        total = max(int(pred.size), 1)
        all_prec.append(tp / max(tp + fp, 1))
        all_rec.append(tp / max(tp + fn, 1))
        all_fp.append(fp / total)
        all_miss.append(fn / total)
        rp, rg = [], []
        for i in range(n):
            qt = int(round(float(qpts[i, 0])))
            evs = find_reentry_events(gv[i], qt)
            if not evs:
                continue
            rt = int(evs[0]["reentry_frame"])
            rp.append(bool(pv[i, rt]))
            rg.append(bool(gv[i, rt]))
        if rg:
            rp = np.asarray(rp, dtype=bool)
            rg = np.asarray(rg, dtype=bool)
            re_pred_rate.append(float(np.mean(rp)))
            re_miss.append(float(np.sum((~rp) & rg)) / max(int(np.sum(rg)), 1))
    return {
        "visible_precision": mean(all_prec),
        "visible_recall": mean(all_rec),
        "false_visible_rate_all": mean(all_fp),
        "missed_visible_rate_all": mean(all_miss),
        "reentry_pred_visible_rate": mean(re_pred_rate),
        "reentry_missed_visible_rate": mean(re_miss),
    }


def eval_standard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    aj, oa, delta = [], [], []
    for r in records:
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        qp = torch.from_numpy(npy(r["query_points"], np.float32))
        m = compute_tapvid_metrics(pred, gt, pv, gv, qp, resolution=256, query_mode="strided")
        aj.append(float(m.get("AJ", 0.0)))
        oa.append(float(m.get("OA", 0.0)))
        delta.append(float(m.get("average_pts_within_thresh", 0.0)))
    return {
        "AJ_256_pct": pct(mean(aj)),
        "OA_256_pct": pct(mean(oa)),
        "delta_avg_256_pct": pct(mean(delta)),
    }


def run_config(out_dir: Path, base_name: str, base: Dict[str, Any], over_name: str, over: Dict[str, Any], pre: int, post: int) -> Dict[str, Any]:
    name = f"{base_name}__override_{over_name}__pre{pre}_post{post}"
    records = localized_records(base, over, pre=pre, post=post)
    payload = dict(base)
    payload["model_name"] = "b2_gt_window_" + name
    payload["b2_base"] = base_name
    payload["b2_override"] = over_name
    payload["b2_pre"] = int(pre)
    payload["b2_post"] = int(post)
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
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bases = {k: load_payload(v) for k, v in BASES.items()}
    overs = {k: load_payload(v) for k, v in OVERRIDES.items()}
    for bname, b in bases.items():
        for oname, o in overs.items():
            check_alignment(b, o)

    configs = []
    for bname in BASES:
        for oname in OVERRIDES:
            for pre in PRES:
                for post in POSTS:
                    configs.append((bname, oname, pre, post))
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    rows = []
    for idx, (bname, oname, pre, post) in enumerate(configs, 1):
        print(f"RUN {idx}/{len(configs)} {bname} {oname} pre={pre} post={post}", flush=True)
        row = run_config(out_dir, bname, bases[bname], oname, overs[oname], pre, post)
        row["delta_AJ_RD_vs_vis4"] = round(float(row.get("true_AJ_RD_256") or 0.0) - 0.6279, 4)
        row["delta_AJ256_vs_vis4"] = round(float(row.get("AJ_256_pct") or 0.0) - 47.4046, 4)
        row["delta_AJ256_vs_old3"] = round(float(row.get("AJ_256_pct") or 0.0) - 49.6122, 4)
        rows.append(row)
        print("ROW", json.dumps(row, ensure_ascii=False), flush=True)

    rows_by_ajrd = sorted(rows, key=lambda r: r.get("true_AJ_RD_256") if r.get("true_AJ_RD_256") is not None else -1, reverse=True)
    pareto = []
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
            pareto.append(r)
    pareto = sorted(pareto, key=lambda r: (float(r.get("true_AJ_RD_256") or 0), float(r.get("AJ_256_pct") or 0)), reverse=True)
    summary = {
        "baselines_reference": {
            "fixed_offline_AJ_RD_256": 0.5546,
            "old3_AJ_RD_256": 0.6189,
            "vis4_AJ_RD_256": 0.6279,
            "fixed_offline_AJ_256_pct": 70.051,
            "old3_AJ_256_pct": 49.6122,
            "vis4_AJ_256_pct": 47.4046,
        },
        "n_configs": len(rows),
        "best_by_AJ_RD_256": rows_by_ajrd[:20],
        "pareto": pareto,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== BEST BY AJ_RD ===")
    print(json.dumps(rows_by_ajrd[:10], indent=2, ensure_ascii=False), flush=True)
    print("=== PARETO ===")
    print(json.dumps(pareto[:20], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
