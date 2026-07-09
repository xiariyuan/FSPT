#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_trigger_refinement")
BASE_CACHE = Path("outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
OVERRIDE_CACHE = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt")
TEACHERS = {
    "online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
    "tapnext": "outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt",
}

# Focused grid around the current best trigger.
KS = [1, 2, 4]
PRES = [0, 1]
POSTS = [16, 32, 9999]
MIN_COUNTS = [1, 2, 3]
PERSIST = [1, 2, 4]
TAUS = [0, 192, 256, 320]  # 0 = no disagreement gate


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_payload(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def px_scale(original_size: np.ndarray) -> np.ndarray:
    h, w = float(original_size[0]), float(original_size[1])
    return np.array([max(h - 1.0, 1.0), max(w - 1.0, 1.0)], dtype=np.float32)


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = t - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def persist_visible(v: np.ndarray, t: int, p: int) -> bool:
    if p <= 1:
        return bool(v[t])
    hi = min(len(v), t + int(p))
    return bool(np.all(v[t:hi])) and hi > t


def pairwise_mean_at(points: np.ndarray, original_size: np.ndarray, t: int) -> float:
    # points: M,T,2 normalized yx
    pts = points[:, t] * px_scale(original_size)
    ds = []
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            ds.append(float(np.linalg.norm(pts[i] - pts[j])))
    return float(np.mean(ds)) if ds else 0.0


def candidate_trigger(base_v: np.ndarray, over_v: np.ndarray, teacher_vis: np.ndarray, teacher_points: np.ndarray, original_size: np.ndarray, t: int, k: int, min_count: int, persist: int, tau: int) -> bool:
    if t <= 0:
        return False
    if invisible_run_before(base_v, t) < int(k):
        return False
    if not bool(over_v[t]):
        return False
    if int(teacher_vis[:, t].sum()) < int(min_count):
        return False
    if not persist_visible(over_v, t, int(persist)):
        return False
    if int(tau) > 0 and pairwise_mean_at(teacher_points, original_size, t) > float(tau):
        return False
    return True


def build_records(base: Dict[str, Any], over: Dict[str, Any], teacher_payloads: Dict[str, Dict[str, Any]], k: int, pre: int, post: int, min_count: int, persist: int, tau: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    teacher_records = {name: payload["records"] for name, payload in teacher_payloads.items()}
    records = []
    total_triggers = 0
    tracks_with_trigger = 0
    for idx, (b, o) in enumerate(zip(base["records"], over["records"])):
        pred_tracks = npy(b["pred_tracks"], np.float32).copy()
        pred_vis = npy(b["pred_visibility"], bool).copy()
        base_vis = pred_vis.copy()
        over_tracks = npy(o["pred_tracks"], np.float32)
        over_vis = npy(o["pred_visibility"], bool)
        qpts = npy(b["query_points"], np.float32)
        osz = npy(b["original_size"], np.float32)
        teacher_vis = np.stack([npy(teacher_records[name][idx]["pred_visibility"], bool) for name in TEACHERS], axis=0)  # M,N,T
        teacher_points = np.stack([npy(teacher_records[name][idx]["pred_tracks"], np.float32) for name in TEACHERS], axis=0)  # M,N,T,2
        n, t_len = pred_vis.shape
        for qi in range(n):
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            mask = np.zeros(t_len, dtype=bool)
            local_triggers = 0
            t = qt + 1
            while t < t_len:
                if candidate_trigger(
                    base_v=base_vis[qi],
                    over_v=over_vis[qi],
                    teacher_vis=teacher_vis[:, qi],
                    teacher_points=teacher_points[:, qi],
                    original_size=osz,
                    t=t,
                    k=k,
                    min_count=min_count,
                    persist=persist,
                    tau=tau,
                ):
                    lo = max(0, t - int(pre))
                    hi = t_len if int(post) >= 9999 else min(t_len, t + int(post) + 1)
                    mask[lo:hi] = True
                    local_triggers += 1
                    t = hi
                else:
                    t += 1
            if local_triggers > 0:
                total_triggers += local_triggers
                tracks_with_trigger += 1
                pred_tracks[qi, mask] = over_tracks[qi, mask]
                pred_vis[qi, mask] = over_vis[qi, mask]
        r = dict(b)
        r["pred_tracks"] = pred_tracks.astype(np.float32)
        r["pred_visibility"] = pred_vis.astype(bool)
        records.append(r)
    return records, {"total_triggers": int(total_triggers), "tracks_with_trigger": int(tracks_with_trigger)}


def eval_ajrd(cache_path: Path, json_path: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(cache_path), "--output-json", str(json_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(json_path))


def mean(vals: List[float]) -> float | None:
    return round(float(np.mean(vals)), 6) if vals else None


def pct(x: float | None) -> float | None:
    return None if x is None else round(float(x) * 100.0, 4)


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
    return {"AJ_256_pct": pct(mean(aj)), "OA_256_pct": pct(mean(oa)), "delta_avg_256_pct": pct(mean(delta))}


def visibility_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    re_pred, re_miss = [], []
    for r in records:
        pv = npy(r["pred_visibility"], bool)
        gv = npy(r["gt_visibility"], bool)
        qpts = npy(r["query_points"], np.float32)
        n, t_len = gv.shape
        rp, rg = [], []
        for qi in range(n):
            qt = int(round(float(qpts[qi, 0])))
            ev = find_reentry_events(gv[qi], qt)
            if not ev:
                continue
            rt = int(ev[0]["reentry_frame"])
            rp.append(bool(pv[qi, rt]))
            rg.append(bool(gv[qi, rt]))
        if rg:
            rp = np.asarray(rp, dtype=bool)
            rg = np.asarray(rg, dtype=bool)
            re_pred.append(float(np.mean(rp)))
            re_miss.append(float(np.sum((~rp) & rg)) / max(int(np.sum(rg)), 1))
    return {"reentry_pred_visible_rate": mean(re_pred), "reentry_missed_visible_rate": mean(re_miss)}


def trigger_quality(base: Dict[str, Any], over: Dict[str, Any], teacher_payloads: Dict[str, Dict[str, Any]], k: int, min_count: int, persist: int, tau: int) -> Dict[str, Any]:
    total_tracks = gt_reentry_tracks = triggered_tracks = triggered_with = triggered_without = no_trigger_with = 0
    for idx, (b, o) in enumerate(zip(base["records"], over["records"])):
        bvis = npy(b["pred_visibility"], bool)
        ovis = npy(o["pred_visibility"], bool)
        gv = npy(b["gt_visibility"], bool)
        qpts = npy(b["query_points"], np.float32)
        osz = npy(b["original_size"], np.float32)
        teacher_vis = np.stack([npy(teacher_payloads[name]["records"][idx]["pred_visibility"], bool) for name in TEACHERS], axis=0)
        teacher_points = np.stack([npy(teacher_payloads[name]["records"][idx]["pred_tracks"], np.float32) for name in TEACHERS], axis=0)
        n, t_len = gv.shape
        for qi in range(n):
            total_tracks += 1
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            has_re = bool(find_reentry_events(gv[qi], qt))
            gt_reentry_tracks += int(has_re)
            trig = False
            for t in range(qt + 1, t_len):
                if candidate_trigger(bvis[qi], ovis[qi], teacher_vis[:, qi], teacher_points[:, qi], osz, t, k, min_count, persist, tau):
                    trig = True
                    break
            if trig:
                triggered_tracks += 1
                if has_re:
                    triggered_with += 1
                else:
                    triggered_without += 1
            elif has_re:
                no_trigger_with += 1
    return {
        "trigger_precision_track": round(triggered_with / max(triggered_tracks, 1), 6),
        "trigger_recall_track": round(triggered_with / max(gt_reentry_tracks, 1), 6),
        "triggered_tracks": int(triggered_tracks),
        "triggered_with_reentry": int(triggered_with),
        "triggered_without_reentry": int(triggered_without),
        "no_trigger_with_reentry": int(no_trigger_with),
    }


def run_config(out_dir: Path, base: Dict[str, Any], over: Dict[str, Any], teacher_payloads: Dict[str, Dict[str, Any]], k: int, pre: int, post: int, min_count: int, persist: int, tau: int) -> Dict[str, Any]:
    name = f"refine_k{k}_pre{pre}_post{post}_cnt{min_count}_pers{persist}_tau{tau}"
    records, trig = build_records(base, over, teacher_payloads, k, pre, post, min_count, persist, tau)
    payload = dict(base)
    payload["model_name"] = "b2_trigger_refine_" + name
    payload["records"] = records
    cp = out_dir / f"{name}.pt"
    jp = out_dir / f"{name}_ajrd.json"
    torch.save(payload, cp)
    ajrd = eval_ajrd(cp, jp)
    std = eval_standard(records)
    vis = visibility_stats(records)
    tq = trigger_quality(base, over, teacher_payloads, k, min_count, persist, tau)
    bd = ajrd.get("aj_rd_by_dmin_256") or {}
    row = {"name": name, "k": k, "pre": pre, "post": post, "min_count": min_count, "persist": persist, "tau": tau,
           "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"), "true_AJ_RD": ajrd.get("true_AJ_RD"),
           "dmin1_256": bd.get("1"), "dmin4_256": bd.get("4"), "dmin16_256": bd.get("16"), **std, **vis, **trig, **tq}
    row["delta_AJ_RD_vs_current_b2"] = round(float(row.get("true_AJ_RD_256") or 0) - 0.6278, 4)
    row["delta_AJ256_vs_current_b2"] = round(float(row.get("AJ_256_pct") or 0) - 68.9702, 4)
    row["delta_AJ256_vs_fixed"] = round(float(row.get("AJ_256_pct") or 0) - 70.0510, 4)
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
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_payload(BASE_CACHE)
    over = load_payload(OVERRIDE_CACHE)
    teacher_payloads = {name: load_payload(path) for name, path in TEACHERS.items()}
    configs = []
    for k in KS:
        for pre in PRES:
            for post in POSTS:
                for min_count in MIN_COUNTS:
                    for persist in PERSIST:
                        for tau in TAUS:
                            configs.append((k, pre, post, min_count, persist, tau))
    if args.max_configs > 0:
        configs = configs[:args.max_configs]
    rows = []
    for i, cfg in enumerate(configs, 1):
        print(f"RUN {i}/{len(configs)} cfg={cfg}", flush=True)
        row = run_config(out_dir, base, over, teacher_payloads, *cfg)
        rows.append(row)
        print("ROW", json.dumps(row, ensure_ascii=False), flush=True)
    best = sorted(rows, key=lambda r: r.get("true_AJ_RD_256") if r.get("true_AJ_RD_256") is not None else -1, reverse=True)
    pareto = pareto_front(rows)
    summary = {"references": {"current_b2_AJ_RD_256": 0.6278, "current_b2_AJ_256_pct": 68.9702, "fixed_AJ_256_pct": 70.0510},
               "n_configs": len(rows), "best_by_AJ_RD_256": best[:30], "pareto": pareto, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== BEST ===")
    print(json.dumps(best[:10], indent=2, ensure_ascii=False), flush=True)
    print("=== PARETO ===")
    print(json.dumps(pareto[:20], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
