#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

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


def validate(caches: Dict[str, Dict[str, Any]]):
    ref = caches[TEACHERS[0]]["records"]
    for t in TEACHERS[1:]:
        recs = caches[t]["records"]
        assert len(recs) == len(ref), (t, len(recs), len(ref))
        for i, (a, b) in enumerate(zip(ref, recs)):
            for k in ["video_id", "frame_count"]:
                assert a[k] == b[k], (t, i, k, a[k], b[k])
            for k in ["query_points", "gt_tracks", "gt_visibility", "original_size", "model_input_size"]:
                aa, bb = npy(a[k]), npy(b[k])
                assert aa.shape == bb.shape and np.allclose(aa, bb, atol=1e-6, rtol=1e-6), (t, i, k)


def finite_mask(pred_stack):
    return np.all(np.isfinite(pred_stack), axis=-1)


def masked_mean(pred_stack, mask):
    cnt = mask.sum(axis=0).astype(np.float32)[..., None]
    sm = np.where(mask[..., None], pred_stack, 0.0).sum(axis=0, dtype=np.float32)
    out = np.full(pred_stack.shape[1:], np.nan, dtype=np.float32)
    np.divide(sm, cnt, out=out, where=cnt > 0)
    return out


def masked_median(pred_stack, mask):
    with np.errstate(all="ignore"):
        return np.nanmedian(np.where(mask[..., None], pred_stack, np.nan), axis=0).astype(np.float32)


def fill(primary, fallback):
    bad = ~np.all(np.isfinite(primary), axis=-1)
    if np.any(bad):
        primary = primary.copy()
        primary[bad] = fallback[bad]
    return primary.astype(np.float32)


def majority_vis(vis_stack):
    return (vis_stack.sum(axis=0) >= int(np.ceil(vis_stack.shape[0] / 2.0))).astype(bool)


def union_vis(vis_stack):
    return np.any(vis_stack, axis=0).astype(bool)


def intersection_vis(vis_stack):
    return np.all(vis_stack, axis=0).astype(bool)


def pairwise_dists_px(pred_stack, original_size):
    # pred_stack M,N,T,2 in normalized yx. Convert to px yx for disagreement.
    h, w = float(original_size[0]), float(original_size[1])
    scale = np.array([max(h - 1.0, 1.0), max(w - 1.0, 1.0)], dtype=np.float32)
    pts = pred_stack * scale
    M = pts.shape[0]
    d = np.zeros((M,) + pts.shape[1:3], dtype=np.float32)
    for i in range(M):
        s = 0
        c = 0
        for j in range(M):
            if i == j: continue
            s = s + np.linalg.norm(pts[i] - pts[j], axis=-1)
            c += 1
        d[i] = s / max(c, 1)
    return d


def local_speed_px(track, original_size):
    h, w = float(original_size[0]), float(original_size[1])
    scale = np.array([max(h - 1.0, 1.0), max(w - 1.0, 1.0)], dtype=np.float32)
    p = track * scale
    diff = np.linalg.norm(np.diff(p, axis=1), axis=-1)  # N,T-1
    left = np.concatenate([diff[:, :1], diff], axis=1)
    right = np.concatenate([diff, diff[:, -1:]], axis=1)
    return 0.5 * (left + right)


def choose_tracks_from_indices(pred_stack, idx):
    # idx N,T; pred_stack M,N,T,2
    M,N,T,_ = pred_stack.shape
    out = np.empty((N,T,2), dtype=np.float32)
    for m in range(M):
        mask = idx == m
        out[mask] = pred_stack[m][mask]
    return out


def choose_vis_from_indices(vis_stack, idx):
    M,N,T = vis_stack.shape
    out = np.empty((N,T), dtype=bool)
    for m in range(M):
        mask = idx == m
        out[mask] = vis_stack[m][mask]
    return out


def aggregate(strategy: str, pred_stack, vis_stack, original_size):
    fm = finite_mask(pred_stack)
    fallback_mean = masked_mean(pred_stack, fm)
    fallback_med = masked_median(pred_stack, fm)

    if strategy == "visible_median_majority":
        tracks = fill(masked_median(pred_stack, fm & vis_stack), fallback_med)
        vis = majority_vis(vis_stack)
    elif strategy == "visible_mean_majority":
        tracks = fill(masked_mean(pred_stack, fm & vis_stack), fallback_mean)
        vis = majority_vis(vis_stack)
    elif strategy == "visible_median_union":
        tracks = fill(masked_median(pred_stack, fm & vis_stack), fallback_med)
        vis = union_vis(vis_stack)
    elif strategy == "visible_median_intersection":
        tracks = fill(masked_median(pred_stack, fm & vis_stack), fallback_med)
        vis = intersection_vis(vis_stack)
    elif strategy == "all_median_majority":
        tracks = fallback_med
        vis = majority_vis(vis_stack)
    elif strategy == "all_mean_majority":
        tracks = fallback_mean
        vis = majority_vis(vis_stack)
    elif strategy.startswith("pair_"):
        # pair_offline_trackon2_visible_median etc.
        name = strategy[len("pair_"):]
        parts = name.split("_")
        pair_key = "_".join(parts[:2])
        pair_map = {
            "online_offline": [0,1],
            "online_trackon2": [0,2],
            "offline_trackon2": [1,2],
        }
        ids = pair_map[pair_key]
        sub_pred, sub_vis = pred_stack[ids], vis_stack[ids]
        sub_fm = finite_mask(sub_pred)
        if "mean" in strategy:
            tracks = fill(masked_mean(sub_pred, sub_fm & sub_vis), masked_mean(sub_pred, sub_fm))
        else:
            tracks = fill(masked_median(sub_pred, sub_fm & sub_vis), masked_median(sub_pred, sub_fm))
        vis = (sub_vis.sum(axis=0) >= 1).astype(bool) if "union" in strategy else (sub_vis.sum(axis=0) >= 1).astype(bool)
    elif strategy == "medoid_frame":
        d = pairwise_dists_px(pred_stack, original_size)
        # Penalize invisible candidates a little but allow if all invisible.
        penalty = np.where(vis_stack, 0.0, 8.0).astype(np.float32)
        idx = np.argmin(d + penalty, axis=0)
        tracks = choose_tracks_from_indices(pred_stack, idx)
        vis = choose_vis_from_indices(vis_stack, idx)
    elif strategy == "medoid_visible_or_offline":
        d = pairwise_dists_px(pred_stack, original_size)
        idx = np.argmin(np.where(vis_stack, d, d + 1e6), axis=0)
        all_inv = ~np.any(vis_stack, axis=0)
        idx[all_inv] = 1  # offline fallback
        tracks = choose_tracks_from_indices(pred_stack, idx)
        vis = choose_vis_from_indices(vis_stack, idx)
    elif strategy == "low_motion_visible":
        speeds = np.stack([local_speed_px(pred_stack[i], original_size) for i in range(pred_stack.shape[0])], axis=0)
        idx = np.argmin(np.where(vis_stack, speeds, speeds + 1e6), axis=0)
        all_inv = ~np.any(vis_stack, axis=0)
        idx[all_inv] = 1
        tracks = choose_tracks_from_indices(pred_stack, idx)
        vis = choose_vis_from_indices(vis_stack, idx)
    elif strategy == "offline_general_trackon2_when_disagree":
        # Start from offline; if offline disagrees > 32px with both others and trackon2 agrees better with online, choose trackon2.
        d = pairwise_dists_px(pred_stack, original_size)
        idx = np.ones(pred_stack.shape[1:3], dtype=np.int64) # offline
        # compare pair distances: choose trackon2 if its mean disagreement is much lower and visible
        choose_t = (d[2] + 8.0 < d[1]) & vis_stack[2]
        idx[choose_t] = 2
        choose_o = (d[0] + 8.0 < d[idx, np.arange(idx.shape[0])[:,None], np.arange(idx.shape[1])[None,:]]) & vis_stack[0]
        # above advanced indexing awkward; skip online override for stability
        tracks = choose_tracks_from_indices(pred_stack, idx)
        vis = choose_vis_from_indices(vis_stack, idx)
    else:
        raise ValueError(strategy)
    tracks = fill(tracks, fallback_med)
    return tracks.astype(np.float32), vis.astype(bool)


def build(strategy: str, caches: Dict[str, Dict[str, Any]], out_cache: Path):
    validate(caches)
    ref_payload = caches[TEACHERS[0]]
    records = []
    for i, ref in enumerate(ref_payload["records"]):
        pred_stack = np.stack([npy(caches[t]["records"][i]["pred_tracks"], np.float32) for t in TEACHERS], axis=0)
        vis_stack = np.stack([npy(caches[t]["records"][i]["pred_visibility"], bool) for t in TEACHERS], axis=0)
        original_size = npy(ref["original_size"], np.float32)
        tracks, vis = aggregate(strategy, pred_stack, vis_stack, original_size)
        r = dict(ref)
        r["pred_tracks"] = tracks
        r["pred_visibility"] = vis
        r["model_name"] = f"route_sweep_{strategy}"
        r["route_sweep_strategy"] = strategy
        r["ensemble_members"] = list(TEACHERS)
        records.append(r)
    out = dict(ref_payload)
    out["model_name"] = f"route_sweep_{strategy}"
    out["records"] = records
    out["route_sweep_strategy"] = strategy
    out["ensemble_members"] = list(TEACHERS)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_cache)


def eval_cache(cache_path: Path, out_json: Path):
    cmd = [sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(cache_path), "--output-json", str(out_json)]
    subprocess.run(cmd, check=True)
    return json.load(open(out_json))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/paper_discovery_2026-06-27/route_sweep")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = {t: load_cache(p) for t,p in CACHE_PATHS.items()}
    strategies = [
        "visible_median_majority", "visible_mean_majority", "visible_median_union", "visible_median_intersection",
        "all_median_majority", "all_mean_majority",
        "pair_online_offline_visible_median", "pair_online_trackon2_visible_median", "pair_offline_trackon2_visible_median",
        "pair_online_offline_visible_mean", "pair_online_trackon2_visible_mean", "pair_offline_trackon2_visible_mean",
        "medoid_frame", "medoid_visible_or_offline", "low_motion_visible", "offline_general_trackon2_when_disagree",
    ]
    rows=[]
    for s in strategies:
        print("===", s, "===", flush=True)
        cache_path = out_dir / f"{s}.pt"
        json_path = out_dir / f"{s}_ajrd.json"
        build(s, caches, cache_path)
        m = eval_cache(cache_path, json_path)
        row = {
            "strategy": s,
            "true_AJ_RD": m.get("true_AJ_RD"),
            "true_AJ_RD_256": m.get("true_AJ_RD_256"),
            "proxy": m.get("first_reentry_frame_proxy"),
            "median_px": (m.get("reentry_error") or {}).get("median_px"),
            "lt4px": (m.get("reentry_error") or {}).get("lt4px"),
            "long20_median_px": (m.get("long_occ_ge20") or {}).get("median_px"),
            "long20_lt4px": (m.get("long_occ_ge20") or {}).get("lt4px"),
            "long20_lt8px": (m.get("long_occ_ge20") or {}).get("lt8px"),
            "long50_median_px": (m.get("long_occ_ge50") or {}).get("median_px"),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows_sorted = sorted(rows, key=lambda r: (r.get("true_AJ_RD_256") or -1), reverse=True)
    summary = {"baseline_visible_median_AJRD256": 0.5871, "fixed_best_AJRD256": 0.5546, "oracle_AJRD256": 0.6509, "rows": rows_sorted}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)
    print("=== BEST ===")
    print(json.dumps(rows_sorted[:10], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
