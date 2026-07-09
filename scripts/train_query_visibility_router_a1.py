#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

TEACHERS = ["cotracker3_online", "cotracker3_offline", "trackon2"]
CACHE_PATHS = {
    "cotracker3_online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "cotracker3_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
}


def load_cache(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)

def npy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

def validate(caches):
    ref = caches[TEACHERS[0]]["records"]
    for t in TEACHERS[1:]:
        recs = caches[t]["records"]
        assert len(recs) == len(ref)
        for i, (a, b) in enumerate(zip(ref, recs)):
            assert a["video_id"] == b["video_id"], (t, i)
            for k in ["query_points", "gt_tracks", "gt_visibility", "original_size", "model_input_size"]:
                assert np.allclose(npy(a[k]), npy(b[k]), atol=1e-6, rtol=1e-6), (t, i, k)

def px_scale(osz):
    h, w = float(osz[0]), float(osz[1])
    return np.array([max(h - 1, 1), max(w - 1, 1)], dtype=np.float32)

def finite(pred):
    return np.all(np.isfinite(pred), axis=-1)

def masked_median(pred, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.where(mask[..., None], pred, np.nan), axis=0).astype(np.float32)

def fill(primary, fallback):
    bad = ~np.all(np.isfinite(primary), axis=-1)
    if np.any(bad):
        primary = primary.copy()
        primary[bad] = fallback[bad]
    return primary.astype(np.float32)

def all_median_tracks(pred_stack):
    fm = finite(pred_stack)
    med = masked_median(pred_stack, fm)
    return fill(med, med)

def majority_vis(vis_stack):
    return (vis_stack.sum(axis=0) >= 2).astype(bool)

def union_vis(vis_stack):
    return (vis_stack.sum(axis=0) >= 1).astype(bool)

def gated_mean_vis(pred_stack, vis_stack, osz, tau=144.0):
    d01, d02, d12 = pairwise_dist(pred_stack, osz)
    mean = (d01 + d02 + d12) / 3.0
    return np.where(mean <= tau, union_vis(vis_stack), majority_vis(vis_stack)).astype(bool)

def pairwise_dist(pred_stack, osz):
    pts = pred_stack * px_scale(osz)
    d01 = np.linalg.norm(pts[0] - pts[1], axis=-1)
    d02 = np.linalg.norm(pts[0] - pts[2], axis=-1)
    d12 = np.linalg.norm(pts[1] - pts[2], axis=-1)
    return d01.astype(np.float32), d02.astype(np.float32), d12.astype(np.float32)

def speed(track, osz):
    pts = track * px_scale(osz)
    if pts.shape[1] < 2:
        return np.zeros(pts.shape[:2], dtype=np.float32)
    d = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    return np.concatenate([d[:, :1], d], axis=1).astype(np.float32)

def safe_stats(x):
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0, 0, 0, 0, 0, 0]
    return [float(np.mean(x)), float(np.std(x)), float(np.median(x)), float(np.percentile(x, 75)), float(np.percentile(x, 95)), float(np.max(x))]

def runlen_stats(v):
    v = np.asarray(v, dtype=bool)
    if v.size == 0:
        return [0, 0, 0, 0]
    inv = ~v
    max_inv = cur = 0
    inv_runs = []
    for b in inv:
        if b:
            cur += 1
            max_inv = max(max_inv, cur)
        else:
            if cur:
                inv_runs.append(cur)
            cur = 0
    if cur:
        inv_runs.append(cur)
    transitions = float(np.mean(np.abs(np.diff(v.astype(np.float32))))) if v.size > 1 else 0.0
    first_true = next((i for i, b in enumerate(v) if b), v.size)
    return [max_inv / max(v.size, 1), float(np.mean(inv_runs)) / max(v.size, 1) if inv_runs else 0.0, transitions, first_true / max(v.size, 1)]

def per_query_ajrd(pred_tracks, gt_tracks, pred_vis, gt_vis, qpts, h, w):
    m = compute_reentry_metrics(pred_tracks, gt_tracks, pred_vis, gt_vis, qpts, h, w)
    out = {}
    for q in m.get("per_query", []):
        qi = int(q["query_idx"])
        val = (q.get("ajrd_summary_256") or {}).get("aj_rd")
        out[qi] = 0.0 if val is None else float(val)
    return out

def build_dataset(caches):
    rows = []
    ref_records = caches[TEACHERS[0]]["records"]
    for ridx, ref in enumerate(ref_records):
        video_id = str(ref["video_id"])
        pred_stack = np.stack([npy(caches[t]["records"][ridx]["pred_tracks"], np.float32) for t in TEACHERS], axis=0)
        vis_stack = np.stack([npy(caches[t]["records"][ridx]["pred_visibility"], bool) for t in TEACHERS], axis=0)
        osz = npy(ref["original_size"], np.float32)
        h, w = int(osz[0]), int(osz[1])
        gt_tracks = npy(ref["gt_tracks"], np.float32)
        gt_vis = npy(ref["gt_visibility"], bool)
        qpts = npy(ref["query_points"], np.float32)
        tracks = all_median_tracks(pred_stack)
        maj = majority_vis(vis_stack)
        uni = union_vis(vis_stack)
        gat = gated_mean_vis(pred_stack, vis_stack, osz, tau=144.0)
        aj_maj = per_query_ajrd(tracks, gt_tracks, maj, gt_vis, qpts, h, w)
        aj_uni = per_query_ajrd(tracks, gt_tracks, uni, gt_vis, qpts, h, w)
        aj_gat = per_query_ajrd(tracks, gt_tracks, gat, gt_vis, qpts, h, w)
        d01, d02, d12 = pairwise_dist(pred_stack, osz)
        dmean = (d01 + d02 + d12) / 3.0
        dmax = np.maximum(np.maximum(d01, d02), d12)
        speeds = np.stack([speed(pred_stack[i], osz) for i in range(3)], axis=0)
        count = vis_stack.sum(axis=0)
        N, T = maj.shape
        for qi in sorted(set(aj_maj) | set(aj_uni) | set(aj_gat)):
            qt = int(round(float(qpts[qi, 0]))) if qpts.ndim == 2 else 0
            qt = max(0, min(T - 1, qt))
            masks = {
                "all": np.ones(T, dtype=bool),
                "post": np.arange(T) >= qt,
                "union": uni[qi],
                "majority": maj[qi],
                "one_visible": count[qi] == 1,
                "two_visible": count[qi] == 2,
            }
            feat = [qt / max(T - 1, 1), float(T), float(qpts[qi, 1]), float(qpts[qi, 2])]
            for slname in ["all", "post"]:
                msk = masks[slname]
                vv = vis_stack[:, qi, msk]
                cc = count[qi, msk]
                feat += [float(np.mean(vv[i])) if vv.shape[1] else 0.0 for i in range(3)]
                feat += [float(np.mean(cc >= 1)), float(np.mean(cc >= 2)), float(np.mean(cc == 1)), float(np.mean(cc == 2)), float(np.mean(cc == 3))]
                for i in range(3):
                    feat += runlen_stats(vis_stack[i, qi, msk])
                feat += runlen_stats(maj[qi, msk]) + runlen_stats(uni[qi, msk])
            for msk in masks.values():
                feat += safe_stats(dmean[qi, msk]) + safe_stats(dmax[qi, msk])
                feat += safe_stats(d01[qi, msk]) + safe_stats(d02[qi, msk]) + safe_stats(d12[qi, msk])
            for i in range(3):
                for msk in [masks["all"], masks["post"], vis_stack[i, qi]]:
                    feat += safe_stats(speeds[i, qi, msk])
            vals = [aj_maj.get(qi, 0.0), aj_uni.get(qi, 0.0), aj_gat.get(qi, 0.0)]
            rows.append({
                "video_id": video_id, "record_index": ridx, "query_idx": int(qi), "feature": feat,
                "aj_majority": vals[0], "aj_union": vals[1], "aj_gated": vals[2],
                "label_union": int(vals[1] > vals[0] + 1e-8),
                "label3": int(np.argmax(vals)),
                "gain_union_vs_majority": vals[1] - vals[0],
                "best_of_three": max(vals),
            })
    return rows

def make_clf(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=8, class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=0)
    raise ValueError(kind)

def make_reg(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    if kind == "rfreg":
        return RandomForestRegressor(n_estimators=400, max_depth=7, min_samples_leaf=8, random_state=1, n_jobs=-1)
    if kind == "hgbreg":
        return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=1)
    raise ValueError(kind)

def build_pred_cache(caches, rows, pred_choice, out_path):
    # pred_choice maps (video_id, query_idx) -> 0 majority, 1 union, 2 gated
    refp = caches[TEACHERS[0]]
    records = []
    for ridx, ref in enumerate(refp["records"]):
        video_id = str(ref["video_id"])
        pred_stack = np.stack([npy(caches[t]["records"][ridx]["pred_tracks"], np.float32) for t in TEACHERS], axis=0)
        vis_stack = np.stack([npy(caches[t]["records"][ridx]["pred_visibility"], bool) for t in TEACHERS], axis=0)
        osz = npy(ref["original_size"], np.float32)
        tracks = all_median_tracks(pred_stack)
        opts = [majority_vis(vis_stack), union_vis(vis_stack), gated_mean_vis(pred_stack, vis_stack, osz, 144.0)]
        out_vis = opts[0].copy()
        for qi in range(out_vis.shape[0]):
            c = int(pred_choice.get((video_id, qi), 0))
            c = max(0, min(2, c))
            out_vis[qi] = opts[c][qi]
        rr = dict(ref)
        rr["pred_tracks"] = tracks
        rr["pred_visibility"] = out_vis.astype(bool)
        rr["model_name"] = "learned_reliability_a1"
        records.append(rr)
    out = dict(refp)
    out["model_name"] = "learned_reliability_a1"
    out["records"] = records
    torch.save(out, out_path)

def eval_cache(cache_path, json_path):
    subprocess.run([sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(cache_path), "--output-json", str(json_path)], check=True)
    return json.load(open(json_path))

def lvo_predict(rows, mode, kind):
    X = np.asarray([r["feature"] for r in rows], dtype=np.float32)
    videos = sorted(set(r["video_id"] for r in rows))
    pred_choice = {}
    eval_rows = []
    if mode == "binary":
        y = np.asarray([r["label_union"] for r in rows], dtype=np.int64)
    elif mode == "three":
        y = np.asarray([r["label3"] for r in rows], dtype=np.int64)
    else:
        y = np.asarray([r["gain_union_vs_majority"] for r in rows], dtype=np.float32)
    all_true = []
    all_pred = []
    all_reg_true = []
    all_reg_pred = []
    for vid in videos:
        tr = [i for i, r in enumerate(rows) if r["video_id"] != vid]
        te = [i for i, r in enumerate(rows) if r["video_id"] == vid]
        if mode in ["binary", "three"]:
            model = make_clf(kind)
            weights = np.asarray([abs(rows[i]["gain_union_vs_majority"]) + 0.02 for i in tr], dtype=np.float32)
            try:
                if kind == "logreg":
                    model.fit(X[tr], y[tr], logisticregression__sample_weight=weights)
                else:
                    model.fit(X[tr], y[tr], sample_weight=weights)
            except TypeError:
                model.fit(X[tr], y[tr])
            pred = model.predict(X[te]).astype(int)
            for idx, pp in zip(te, pred):
                r = rows[idx]
                choice = int(pp) if mode == "three" else (1 if int(pp) == 1 else 0)
                pred_choice[(r["video_id"], r["query_idx"])] = choice
                all_true.append(int(y[idx])); all_pred.append(int(pp))
                eval_rows.append({k:v for k,v in r.items() if k != "feature"} | {"pred": int(pp), "choice": choice})
        else:
            model = make_reg(kind)
            weights = np.asarray([abs(rows[i]["gain_union_vs_majority"]) + 0.02 for i in tr], dtype=np.float32)
            try:
                model.fit(X[tr], y[tr], sample_weight=weights)
            except TypeError:
                model.fit(X[tr], y[tr])
            pred = np.asarray(model.predict(X[te]), dtype=np.float32)
            for idx, pp in zip(te, pred):
                r = rows[idx]
                choice = 1 if pp > 0.0 else 0
                pred_choice[(r["video_id"], r["query_idx"])] = choice
                all_reg_true.append(float(y[idx])); all_reg_pred.append(float(pp))
                eval_rows.append({k:v for k,v in r.items() if k != "feature"} | {"pred_gain": float(pp), "choice": choice})
    stats = {}
    if mode in ["binary", "three"]:
        stats["accuracy"] = float(accuracy_score(all_true, all_pred))
        stats["balanced_accuracy"] = float(balanced_accuracy_score(all_true, all_pred))
        stats["pred_choice_counts"] = {str(i): int(sum(1 for p in all_pred if p == i)) for i in sorted(set(all_pred))}
    else:
        stats["mae_gain"] = float(mean_absolute_error(all_reg_true, all_reg_pred))
        stats["pred_union_rate"] = float(np.mean([1 if p > 0 else 0 for p in all_reg_pred]))
    return pred_choice, stats, eval_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/paper_discovery_2026-06-27/learned_reliability/a1")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caches = {t: load_cache(p) for t, p in CACHE_PATHS.items()}
    validate(caches)
    rows = build_dataset(caches)
    X0 = np.asarray(rows[0]["feature"], dtype=np.float32) if rows else np.zeros(0)
    dataset_summary = {
        "n_rows": len(rows), "n_features": int(X0.size),
        "videos": sorted(set(r["video_id"] for r in rows)),
        "label_union_rate": float(np.mean([r["label_union"] for r in rows])) if rows else 0,
        "label3_counts": {str(i): int(sum(1 for r in rows if r["label3"] == i)) for i in range(3)},
        "mean_gain_union_vs_majority": float(np.mean([r["gain_union_vs_majority"] for r in rows])) if rows else 0,
    }
    json.dump(dataset_summary, open(out / "dataset_summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False), flush=True)
    jobs = []
    for kind in ["logreg", "rf", "hgb"]:
        jobs.append(("binary", kind))
    for kind in ["logreg", "rf", "hgb"]:
        jobs.append(("three", kind))
    for kind in ["ridge", "rfreg", "hgbreg"]:
        jobs.append(("regress", kind))
    results = []
    for mode, kind in jobs:
        print("===", mode, kind, "===", flush=True)
        pred_choice, stats, eval_rows = lvo_predict(rows, mode, kind)
        cache_path = out / f"a1_{mode}_{kind}.pt"
        json_path = out / f"a1_{mode}_{kind}_ajrd.json"
        build_pred_cache(caches, rows, pred_choice, cache_path)
        m = eval_cache(cache_path, json_path)
        row = {
            "mode": mode, "kind": kind,
            "true_AJ_RD_256": m.get("true_AJ_RD_256"),
            "true_AJ_RD": m.get("true_AJ_RD"),
            "proxy": m.get("first_reentry_frame_proxy"),
            "long20_lt4px": (m.get("long_occ_ge20") or {}).get("lt4px"),
            "long20_lt8px": (m.get("long_occ_ge20") or {}).get("lt8px"),
            "dmin16": (m.get("aj_rd_by_dmin_256") or {}).get("16"),
            **stats,
        }
        results.append(row)
        json.dump({"row": row, "eval_rows": eval_rows}, open(out / f"a1_{mode}_{kind}_cvpred.json", "w"), indent=2, ensure_ascii=False)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    baselines = {"fixed_best":0.5546,"old_masked_median_majority":0.5871,"all_median_union":0.6171,"heuristic_gated_mean144":0.6189,"oracle":0.6509}
    summary = {"dataset": dataset_summary, "baselines": baselines, "results": sorted(results, key=lambda r:r["true_AJ_RD_256"], reverse=True)}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, ensure_ascii=False)
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
