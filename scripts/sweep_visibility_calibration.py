#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch

TEACHERS = ["cotracker3_online", "cotracker3_offline", "trackon2"]
CACHE_PATHS = {
    "cotracker3_online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "cotracker3_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
}


def load_cache(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def npy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)


def finite(pred_stack: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(pred_stack), axis=-1)


def masked_median(pred_stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmedian(np.where(mask[..., None], pred_stack, np.nan), axis=0).astype(np.float32)


def fill(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    bad = ~np.all(np.isfinite(primary), axis=-1)
    if np.any(bad):
        primary = primary.copy()
        primary[bad] = fallback[bad]
    return primary.astype(np.float32)


def all_median_tracks(pred_stack: np.ndarray) -> np.ndarray:
    fm = finite(pred_stack)
    return fill(masked_median(pred_stack, fm), masked_median(pred_stack, fm))


def px_scale(original_size: np.ndarray) -> np.ndarray:
    h, w = float(original_size[0]), float(original_size[1])
    return np.array([max(h - 1.0, 1.0), max(w - 1.0, 1.0)], dtype=np.float32)


def disagreement_px(pred_stack: np.ndarray, original_size: np.ndarray) -> Dict[str, np.ndarray]:
    # M,N,T,2 normalized yx -> pairwise px distance.
    pts = pred_stack * px_scale(original_size)
    d01 = np.linalg.norm(pts[0] - pts[1], axis=-1)
    d02 = np.linalg.norm(pts[0] - pts[2], axis=-1)
    d12 = np.linalg.norm(pts[1] - pts[2], axis=-1)
    return {
        "max": np.maximum(np.maximum(d01, d02), d12).astype(np.float32),
        "mean": ((d01 + d02 + d12) / 3.0).astype(np.float32),
        "offline_trackon2": d12.astype(np.float32),
        "online_offline": d01.astype(np.float32),
        "online_trackon2": d02.astype(np.float32),
    }


def dilate_time(vis: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return vis.copy()
    out = vis.copy()
    N, T = vis.shape
    for shift in range(1, radius + 1):
        out[:, shift:] |= vis[:, :-shift]
        out[:, :-shift] |= vis[:, shift:]
    return out.astype(bool)


def reentry_override(majority: np.ndarray, union: np.ndarray, gap: int) -> np.ndarray:
    # Deployable: majority by default; if majority has been invisible for >= gap frames
    # and any teacher becomes visible, allow union for that frame.
    out = majority.copy()
    N, T = majority.shape
    invisible_run = np.zeros(N, dtype=np.int32)
    for t in range(T):
        activate = union[:, t] & (~majority[:, t]) & (invisible_run >= gap)
        out[:, t] |= activate
        invisible_run = np.where(majority[:, t], 0, invisible_run + 1)
    return out.astype(bool)


def reentry_window_override(majority: np.ndarray, union: np.ndarray, gap: int, window: int) -> np.ndarray:
    # Once union reappears after a majority-invisible run, keep union enabled for a small window.
    out = majority.copy()
    N, T = majority.shape
    invisible_run = np.zeros(N, dtype=np.int32)
    active_until = np.full(N, -1, dtype=np.int32)
    for t in range(T):
        start = union[:, t] & (~majority[:, t]) & (invisible_run >= gap)
        active_until = np.where(start, t + window, active_until)
        active = union[:, t] & (t <= active_until)
        out[:, t] |= active
        invisible_run = np.where(majority[:, t], 0, invisible_run + 1)
    return out.astype(bool)


def make_visibility(rule: str, vis_stack: np.ndarray, pred_stack: np.ndarray, original_size: np.ndarray) -> np.ndarray:
    count = vis_stack.sum(axis=0)
    union = count >= 1
    majority = count >= 2
    intersection = count >= 3
    if rule == "majority": return majority.astype(bool)
    if rule == "union": return union.astype(bool)
    if rule == "intersection": return intersection.astype(bool)
    if rule == "online": return vis_stack[0].astype(bool)
    if rule == "offline": return vis_stack[1].astype(bool)
    if rule == "trackon2": return vis_stack[2].astype(bool)
    if rule.startswith("union_dilate"):
        r = int(rule.replace("union_dilate", ""))
        return dilate_time(union, r)
    if rule.startswith("majority_dilate"):
        r = int(rule.replace("majority_dilate", ""))
        return dilate_time(majority, r)
    if rule.startswith("gated_mean"):
        thr = float(rule.replace("gated_mean", ""))
        dis = disagreement_px(pred_stack, original_size)["mean"]
        return np.where(dis <= thr, union, majority).astype(bool)
    if rule.startswith("gated_max"):
        thr = float(rule.replace("gated_max", ""))
        dis = disagreement_px(pred_stack, original_size)["max"]
        return np.where(dis <= thr, union, majority).astype(bool)
    if rule.startswith("gated_ot"):
        thr = float(rule.replace("gated_ot", ""))
        dis = disagreement_px(pred_stack, original_size)["offline_trackon2"]
        return np.where(dis <= thr, union, majority).astype(bool)
    if rule.startswith("reentry_gap") and "_w" not in rule:
        gap = int(rule.replace("reentry_gap", ""))
        return reentry_override(majority, union, gap)
    if rule.startswith("reentry_gap") and "_w" in rule:
        left, right = rule.split("_w", 1)
        gap = int(left.replace("reentry_gap", ""))
        window = int(right)
        return reentry_window_override(majority, union, gap, window)
    raise ValueError(rule)


def validate(caches: Dict[str, Dict[str, Any]]) -> None:
    ref = caches[TEACHERS[0]]["records"]
    for t in TEACHERS[1:]:
        recs = caches[t]["records"]
        assert len(recs) == len(ref)
        for i, (a, b) in enumerate(zip(ref, recs)):
            assert a["video_id"] == b["video_id"], (t, i)
            for k in ["query_points", "gt_tracks", "gt_visibility", "original_size", "model_input_size"]:
                assert np.allclose(npy(a[k]), npy(b[k]), atol=1e-6, rtol=1e-6), (t, i, k)


def build(rule: str, caches: Dict[str, Dict[str, Any]], out_cache: Path) -> None:
    refp = caches[TEACHERS[0]]
    records = []
    for i, ref in enumerate(refp["records"]):
        pred_stack = np.stack([npy(caches[t]["records"][i]["pred_tracks"], np.float32) for t in TEACHERS], axis=0)
        vis_stack = np.stack([npy(caches[t]["records"][i]["pred_visibility"], bool) for t in TEACHERS], axis=0)
        osz = npy(ref["original_size"], np.float32)
        r = dict(ref)
        r["pred_tracks"] = all_median_tracks(pred_stack)
        r["pred_visibility"] = make_visibility(rule, vis_stack, pred_stack, osz)
        r["model_name"] = f"all_median_visibility_{rule}"
        r["track_rule"] = "all_median"
        r["visibility_rule"] = rule
        records.append(r)
    out = dict(refp)
    out["model_name"] = f"all_median_visibility_{rule}"
    out["track_rule"] = "all_median"
    out["visibility_rule"] = rule
    out["records"] = records
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_cache)


def eval_cache(cache_path: Path, json_path: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(cache_path), "--output-json", str(json_path)], check=True)
    return json.load(open(json_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/paper_discovery_2026-06-27/visibility_sweep")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = {t: load_cache(p) for t, p in CACHE_PATHS.items()}
    validate(caches)
    rules = ["majority", "union", "intersection", "online", "offline", "trackon2"]
    rules += [f"union_dilate{r}" for r in [1, 2, 3, 4]]
    rules += [f"majority_dilate{r}" for r in [1, 2, 3, 4]]
    rules += [f"gated_mean{thr}" for thr in [4, 8, 12, 16, 24, 32, 48, 64, 96]]
    rules += [f"gated_max{thr}" for thr in [8, 16, 24, 32, 48, 64, 96, 128]]
    rules += [f"gated_ot{thr}" for thr in [4, 8, 12, 16, 24, 32, 48, 64, 96]]
    rules += [f"reentry_gap{g}" for g in [1, 2, 4, 8, 12, 16, 20]]
    rules += [f"reentry_gap{g}_w{w}" for g in [1, 4, 8, 16] for w in [1, 2, 4]]
    rows = []
    for rule in rules:
        print("===", rule, "===", flush=True)
        cp = out_dir / f"all_median__vis_{rule}.pt"
        jp = out_dir / f"all_median__vis_{rule}_ajrd.json"
        build(rule, caches, cp)
        m = eval_cache(cp, jp)
        row = {
            "rule": rule,
            "true_AJ_RD": m.get("true_AJ_RD"),
            "true_AJ_RD_256": m.get("true_AJ_RD_256"),
            "proxy": m.get("first_reentry_frame_proxy"),
            "median_px": (m.get("reentry_error") or {}).get("median_px"),
            "lt4px": (m.get("reentry_error") or {}).get("lt4px"),
            "long20_median_px": (m.get("long_occ_ge20") or {}).get("median_px"),
            "long20_lt4px": (m.get("long_occ_ge20") or {}).get("lt4px"),
            "long20_lt8px": (m.get("long_occ_ge20") or {}).get("lt8px"),
            "dmin16": (m.get("aj_rd_by_dmin_256") or {}).get("16"),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows = sorted(rows, key=lambda r: r.get("true_AJ_RD_256") or -1, reverse=True)
    summary = {
        "fixed_best_AJ_RD_256": 0.5546,
        "old_masked_median_majority_AJ_RD_256": 0.5871,
        "previous_v2_best_all_median_union_AJ_RD_256": 0.6171,
        "oracle_AJ_RD_256": 0.6509,
        "rows": rows,
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)
    print("=== BEST VISIBILITY SWEEP ===")
    print(json.dumps(rows[:20], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
