#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

TEACHERS = ["cotracker3_online", "cotracker3_offline", "trackon2", "tapnext"]
CACHE_PATHS = {
    "cotracker3_online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "cotracker3_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
    "tapnext": "outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt",
}
BASELINE = 0.6189
ORACLE = 0.6509
FIXED_BEST = 0.5546

def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def npy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr

def finite(pred_stack):
    return np.all(np.isfinite(pred_stack), axis=-1)

def masked_mean(pred_stack, mask):
    cnt = mask.sum(axis=0).astype(np.float32)[..., None]
    sm = np.where(mask[..., None], pred_stack, 0.0).sum(axis=0, dtype=np.float32)
    out = np.full(pred_stack.shape[1:], np.nan, dtype=np.float32)
    np.divide(sm, cnt, out=out, where=cnt > 0)
    return out.astype(np.float32)

def masked_median(pred_stack, mask):
    with np.errstate(all="ignore"):
        return np.nanmedian(np.where(mask[..., None], pred_stack, np.nan), axis=0).astype(np.float32)

def fill(a, b):
    bad = ~np.all(np.isfinite(a), axis=-1)
    if np.any(bad):
        a = a.copy(); a[bad] = b[bad]
    return a.astype(np.float32)

def px_scale(osz):
    h, w = float(osz[0]), float(osz[1])
    return np.array([max(h - 1, 1), max(w - 1, 1)], dtype=np.float32)

def pairwise_mean_dist(pred_stack, osz):
    pts = pred_stack * px_scale(osz)
    m = pts.shape[0]
    vals = []
    for i in range(m):
        ds = []
        for j in range(m):
            if i != j:
                ds.append(np.linalg.norm(pts[i] - pts[j], axis=-1))
        vals.append(np.mean(ds, axis=0))
    return np.stack(vals, axis=0).astype(np.float32)

def global_mean_pairwise(pred_stack, osz):
    pts = pred_stack * px_scale(osz)
    ds = []
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            ds.append(np.linalg.norm(pts[i] - pts[j], axis=-1))
    return np.mean(np.stack(ds, axis=0), axis=0).astype(np.float32)

def choose(pred_stack, idx):
    out = np.empty(pred_stack.shape[1:], dtype=np.float32)
    for i in range(pred_stack.shape[0]):
        m = idx == i
        out[m] = pred_stack[i][m]
    return out

def tracks_rule(pred_stack, vis_stack, osz, rule):
    fm = finite(pred_stack)
    fallback_mean = masked_mean(pred_stack, fm)
    fallback_med = masked_median(pred_stack, fm)
    if rule == "all_median": return fallback_med
    if rule == "all_mean": return fallback_mean
    if rule == "visible_median": return fill(masked_median(pred_stack, fm & vis_stack), fallback_med)
    if rule == "visible_mean": return fill(masked_mean(pred_stack, fm & vis_stack), fallback_mean)
    if rule == "first3_all_median": return masked_median(pred_stack[:3], fm[:3])
    if rule == "first3_visible_median": return fill(masked_median(pred_stack[:3], fm[:3] & vis_stack[:3]), masked_median(pred_stack[:3], fm[:3]))
    if rule == "medoid":
        d = pairwise_mean_dist(pred_stack, osz)
        return choose(pred_stack, np.argmin(d, axis=0))
    if rule == "visible_medoid":
        d = pairwise_mean_dist(pred_stack, osz)
        idx = np.argmin(np.where(vis_stack, d, d + 1e6), axis=0)
        idx[~np.any(vis_stack, axis=0)] = int(np.argmin(pairwise_mean_dist(pred_stack, osz), axis=0)[~np.any(vis_stack, axis=0)][0]) if np.any(~np.any(vis_stack, axis=0)) else 0
        return choose(pred_stack, idx)
    if rule in TEACHERS:
        return pred_stack[TEACHERS.index(rule)].astype(np.float32)
    raise ValueError(rule)

def vis_rule(vis_stack, pred_stack, osz, rule):
    c = vis_stack.sum(axis=0)
    m = vis_stack.shape[0]
    if rule == "union": return c >= 1
    if rule == "half": return c >= 2
    if rule == "strict_majority": return c >= 3
    if rule == "intersection": return c >= m
    if rule == "first3_union": return vis_stack[:3].sum(axis=0) >= 1
    if rule == "first3_majority": return vis_stack[:3].sum(axis=0) >= 2
    if rule in TEACHERS: return vis_stack[TEACHERS.index(rule)].astype(bool)
    if rule.startswith("gated_mean"):
        # If teachers agree spatially, keep union to recover re-entry; if disagree, use conservative half vote.
        tau = float(rule.replace("gated_mean", ""))
        agree = global_mean_pairwise(pred_stack, osz) <= tau
        return np.where(agree, c >= 1, c >= 2).astype(bool)
    if rule.startswith("gated_strict"):
        tau = float(rule.replace("gated_strict", ""))
        agree = global_mean_pairwise(pred_stack, osz) <= tau
        return np.where(agree, c >= 1, c >= 3).astype(bool)
    raise ValueError(rule)

def validate(caches):
    ref = caches[TEACHERS[0]]["records"]
    for t in TEACHERS[1:]:
        assert len(caches[t]["records"]) == len(ref), (t, len(caches[t]["records"]), len(ref))
        for i, (a, b) in enumerate(zip(ref, caches[t]["records"])):
            assert a["video_id"] == b["video_id"], (t, i, a["video_id"], b["video_id"])
            assert np.asarray(a["pred_tracks"]).shape == np.asarray(b["pred_tracks"]).shape, (t, i, "pred")
            assert np.asarray(a["pred_visibility"]).shape == np.asarray(b["pred_visibility"]).shape, (t, i, "vis")
            assert np.allclose(npy(a["original_size"]), npy(b["original_size"]), atol=0, rtol=0), (t, i, "size")

def build(name, trule, vrule, caches, out_cache):
    validate(caches)
    refp = caches[TEACHERS[0]]
    records = []
    for i, ref in enumerate(refp["records"]):
        pred = np.stack([npy(caches[t]["records"][i]["pred_tracks"], np.float32) for t in TEACHERS], axis=0)
        vis = np.stack([npy(caches[t]["records"][i]["pred_visibility"], bool) for t in TEACHERS], axis=0)
        osz = npy(ref["original_size"], np.float32)
        r = dict(ref)
        # Use reference query/GT from original 3-teacher cache for exact comparability.
        r["pred_tracks"] = tracks_rule(pred, vis, osz, trule)
        r["pred_visibility"] = vis_rule(vis, pred, osz, vrule)
        r["model_name"] = "route_b1_4teacher_" + name
        r["route_tracks_rule"] = trule
        r["route_visibility_rule"] = vrule
        r["route_teachers"] = TEACHERS
        records.append(r)
    out = dict(refp)
    out["model_name"] = "route_b1_4teacher_" + name
    out["records"] = records
    out["route_tracks_rule"] = trule
    out["route_visibility_rule"] = vrule
    out["route_teachers"] = TEACHERS
    torch.save(out, out_cache)

def eval_cache(cache_path, json_path):
    subprocess.run([sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(cache_path), "--output-json", str(json_path)], check=True)
    return json.load(open(json_path))

def row_from_metric(name, tr, vr, m):
    bd256 = m.get("aj_rd_by_dmin_256") or {}
    return {
        "name": name,
        "tracks": tr,
        "visibility": vr,
        "true_AJ_RD": m.get("true_AJ_RD"),
        "true_AJ_RD_256": m.get("true_AJ_RD_256"),
        "proxy": m.get("first_reentry_frame_proxy"),
        "dmin1_256": bd256.get("1"),
        "dmin4_256": bd256.get("4"),
        "dmin16_256": bd256.get("16"),
        "median_px": (m.get("reentry_error") or {}).get("median_px"),
        "lt4px": (m.get("reentry_error") or {}).get("lt4px"),
        "long20_lt4px": (m.get("long_occ_ge20") or {}).get("lt4px"),
        "long20_lt8px": (m.get("long_occ_ge20") or {}).get("lt8px"),
        "delta_vs_06189": None if m.get("true_AJ_RD_256") is None else round(float(m.get("true_AJ_RD_256")) - BASELINE, 4),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_sweep")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    caches = {t: load_cache(p) for t, p in CACHE_PATHS.items()}
    validate(caches)
    track_rules = ["all_median", "visible_median", "all_mean", "visible_mean", "medoid", "visible_medoid", "first3_all_median", "first3_visible_median", "tapnext"]
    if args.quick:
        track_rules = ["all_median", "visible_median", "first3_all_median", "tapnext"]
    vis_rules = ["union", "half", "strict_majority", "first3_union", "first3_majority", "tapnext"]
    vis_rules += [f"gated_mean{t}" for t in [32, 64, 96, 128, 144, 160, 192, 224, 256]]
    vis_rules += [f"gated_strict{t}" for t in [96, 128, 144, 160, 192]]
    if args.quick:
        vis_rules = ["union", "half", "strict_majority", "first3_majority", "gated_mean144", "gated_strict144", "tapnext"]
    rows = []
    for tr in track_rules:
        for vr in vis_rules:
            name = f"{tr}__vis_{vr}"
            cp = out / f"{name}.pt"
            jp = out / f"{name}_ajrd.json"
            print("===", name, "===", flush=True)
            build(name, tr, vr, caches, cp)
            m = eval_cache(cp, jp)
            row = row_from_metric(name, tr, vr, m)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows = sorted(rows, key=lambda r: r.get("true_AJ_RD_256") if r.get("true_AJ_RD_256") is not None else -1, reverse=True)
    summary = {
        "baseline_3teacher_best_gated_mean144": BASELINE,
        "fixed_best_single": FIXED_BEST,
        "oracle_3teacher_reported": ORACLE,
        "rows": rows,
        "best": rows[0] if rows else None,
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2, ensure_ascii=False)
    print("=== BEST B1 4TEACHER ===")
    print(json.dumps(rows[:25], indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
