#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_b1_action_value_table import DEFAULT_ACTIONS

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_safe_router")
TABLE = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_action_oracle/action_value_table.jsonl")
BASE = "vis4_gated288"
TEACHERS = ["online", "offline", "trackon2", "tapnext"]
TEACHER_PATHS = {
    "online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
    "tapnext": "outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt",
}
MARGINS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]


def npy(x: Any, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def stats(x):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(np.mean(x)), float(np.std(x)), float(np.percentile(x, 90)), float(np.max(x))]


def bool_stats(v):
    v = np.asarray(v, dtype=bool)
    if v.size == 0:
        return [0.0, 0.0, 0.0]
    trans = float(np.mean(np.abs(np.diff(v.astype(np.float32))))) if v.size > 1 else 0.0
    first = next((i for i, b in enumerate(v) if b), v.size) / max(v.size, 1)
    return [float(np.mean(v)), trans, float(first)]


def px_scale(osz):
    h, w = float(osz[0]), float(osz[1])
    return np.asarray([max(h - 1, 1), max(w - 1, 1)], dtype=np.float32)


def load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def make_features(rows, teacher_payloads):
    by_video = {name: {str(r["video_id"]): r for r in payload["records"]} for name, payload in teacher_payloads.items()}
    X = []
    for row in rows:
        vid = str(row["video_id"])
        qi = int(row["query_idx"])
        ref = by_video[TEACHERS[0]][vid]
        q = npy(ref["query_points"], np.float32)
        osz = npy(ref["original_size"], np.float32)
        T = int(npy(ref["pred_tracks"]).shape[1])
        qt = int(round(float(q[qi, 0])))
        qt = max(0, min(T - 1, qt))
        masks = [np.ones(T, dtype=bool), np.arange(T) >= qt]
        tracks = []
        vis = []
        for t in TEACHERS:
            r = by_video[t][vid]
            tracks.append(npy(r["pred_tracks"], np.float32)[qi])
            vis.append(npy(r["pred_visibility"], bool)[qi])
        tracks = np.stack(tracks, axis=0)
        vis = np.stack(vis, axis=0)
        pts = tracks * px_scale(osz)
        old_med = np.nanmedian(pts[:3], axis=0)
        all_med = np.nanmedian(pts, axis=0)
        feat = [qt / max(T - 1, 1), float(T) / 100.0, float(q[qi, 1]), float(q[qi, 2])]
        count = vis.sum(axis=0)
        for m in masks:
            c = count[m]
            feat += stats(c)
            feat += [float(np.mean(c >= 1)), float(np.mean(c >= 2)), float(np.mean(c >= 3)), float(np.mean(c == 1))]
            for i in range(4):
                feat += bool_stats(vis[i, m])
            pair = []
            for i in range(4):
                for j in range(i + 1, 4):
                    pair.append(np.linalg.norm(pts[i, m] - pts[j, m], axis=-1))
            feat += stats(np.concatenate(pair) if pair else [])
            feat += stats(np.linalg.norm(pts[3, m] - old_med[m], axis=-1))
            for i in range(4):
                feat += stats(np.linalg.norm(pts[i, m] - all_med[m], axis=-1))
                if tracks.shape[1] > 1:
                    sp = np.linalg.norm(np.diff(pts[i], axis=0), axis=-1)
                    feat += stats(sp)
        X.append(feat)
    return np.asarray(X, dtype=np.float32)


def make_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    if kind == "hgb":
        return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=42)
    raise ValueError(kind)


def eval_cache(cp, jp):
    subprocess.run([sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(cp), "--output-json", str(jp)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(jp))


def build_cache(action_payloads, rows, choices, out_path, model_name):
    base = action_payloads[BASE]
    by_action_video = {name: {str(r["video_id"]): r for r in payload["records"]} for name, payload in action_payloads.items()}
    cmap = {(str(r["video_id"]), int(r["query_idx"])): c for r, c in zip(rows, choices)}
    out_records = []
    for br in base["records"]:
        vid = str(br["video_id"])
        nr = dict(br)
        tr = npy(br["pred_tracks"], np.float32).copy()
        vv = npy(br["pred_visibility"], bool).copy()
        for qi in range(tr.shape[0]):
            a = cmap.get((vid, qi), BASE)
            if a != BASE:
                src = by_action_video[a][vid]
                tr[qi] = npy(src["pred_tracks"], np.float32)[qi]
                vv[qi] = npy(src["pred_visibility"], bool)[qi]
        nr["pred_tracks"] = tr.astype(np.float32)
        nr["pred_visibility"] = vv.astype(bool)
        nr["model_name"] = model_name
        out_records.append(nr)
    out = dict(base)
    out["model_name"] = model_name
    out["router_baseline_action"] = BASE
    out["records"] = out_records
    torch.save(out, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ridge,hgb")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in TABLE.read_text().splitlines() if l.strip()]
    actions = list(DEFAULT_ACTIONS.keys())
    base_idx = actions.index(BASE)
    Y = np.asarray([[float(r["scores"].get(a, r["baseline_score"])) for a in actions] for r in rows], dtype=np.float32)
    D = Y - Y[:, base_idx:base_idx + 1]
    groups_s = [str(r["video_id"]) for r in rows]
    vids = sorted(set(groups_s))
    groups = np.asarray([vids.index(v) for v in groups_s], dtype=np.int32)
    teacher_payloads = {k: load_payload(v) for k, v in TEACHER_PATHS.items()}
    X = make_features(rows, teacher_payloads)
    action_payloads = {name: load_payload(path) for name, path in DEFAULT_ACTIONS.items()}
    print({"rows": X.shape[0], "features": X.shape[1], "actions": len(actions), "videos": len(vids)}, flush=True)
    results = []
    gkf = GroupKFold(n_splits=min(args.splits, len(vids)))
    for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
        print("MODEL", kind, flush=True)
        pred = np.zeros_like(D, dtype=np.float32)
        for ai, action in enumerate(actions):
            if action == BASE:
                continue
            y = D[:, ai]
            fp = np.zeros(X.shape[0], dtype=np.float32)
            for tr_idx, te_idx in gkf.split(X, y, groups):
                model = make_model(kind)
                model.fit(X[tr_idx], y[tr_idx])
                fp[te_idx] = model.predict(X[te_idx]).astype(np.float32)
            pred[:, ai] = fp
        np.save(OUT / f"{kind}_pred_delta.npy", pred)
        for margin in MARGINS:
            best_idx = np.argmax(pred, axis=1)
            best_val = pred[np.arange(len(rows)), best_idx]
            chosen_idx = np.where(best_val > margin, best_idx, base_idx)
            choices = [actions[int(i)] for i in chosen_idx]
            name = f"b1_safe_router_{kind}_m{str(margin).replace('.', 'p')}"
            cp = OUT / f"{name}.pt"
            jp = OUT / f"{name}_ajrd.json"
            build_cache(action_payloads, rows, choices, cp, name)
            m = eval_cache(cp, jp)
            bd = m.get("aj_rd_by_dmin_256") or {}
            unique, counts = np.unique(choices, return_counts=True)
            row = {
                "model": kind,
                "margin": margin,
                "name": name,
                "true_AJ_RD_256": m.get("true_AJ_RD_256"),
                "true_AJ_RD": m.get("true_AJ_RD"),
                "proxy": m.get("first_reentry_frame_proxy"),
                "dmin1_256": bd.get("1"),
                "dmin4_256": bd.get("4"),
                "dmin16_256": bd.get("16"),
                "long20_lt4px": (m.get("long_occ_ge20") or {}).get("lt4px"),
                "long20_lt8px": (m.get("long_occ_ge20") or {}).get("lt8px"),
                "n_overrides": int(sum(1 for c in choices if c != BASE)),
                "choice_counts": {str(u): int(c) for u, c in zip(unique, counts)},
                "delta_vs_06279": round(float(m.get("true_AJ_RD_256") or 0) - 0.6279, 4),
            }
            print("ROW", json.dumps(row, ensure_ascii=False), flush=True)
            results.append(row)
    results.sort(key=lambda r: r.get("true_AJ_RD_256") if r.get("true_AJ_RD_256") is not None else -1, reverse=True)
    summary = {"baseline_action": BASE, "baseline_true_AJ_RD_256": 0.6279, "action_oracle_true_AJ_RD_256": 0.6531, "results": results, "best": results[0] if results else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("BEST", json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
