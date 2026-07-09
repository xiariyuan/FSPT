#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def npy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_percentile(x: np.ndarray, ps=(50, 75, 90, 95, 99)) -> Dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": round(float(np.percentile(x, p)), 6) for p in ps}


def summarize_arr(x: np.ndarray) -> Dict[str, float | int | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, **safe_percentile(x)}
    return {
        "n": int(x.size),
        "mean": round(float(np.mean(x)), 6),
        "median": round(float(np.median(x)), 6),
        "min": round(float(np.min(x)), 6),
        "max": round(float(np.max(x)), 6),
        **safe_percentile(x),
    }


def first_reentry_frames(gt_vis: np.ndarray, q_t: np.ndarray) -> np.ndarray:
    # Find first transition invisible->visible after query frame, per query.
    n, tmax = gt_vis.shape
    out = np.full((n,), -1, dtype=np.int32)
    for i in range(n):
        start = int(q_t[i]) + 1
        if start >= tmax:
            continue
        v = gt_vis[i]
        prev = v[start - 1]
        for t in range(start, tmax):
            if (not prev) and v[t]:
                out[i] = t
                break
            prev = v[t]
    return out


def audit_record(r: Dict[str, Any]) -> Dict[str, Any]:
    q = npy(r["query_points"]).astype(np.float64)  # N, [t,y,x] normalized
    pred = npy(r["pred_tracks"]).astype(np.float64)  # N,T,2 normalized yx
    gt = npy(r["gt_tracks"]).astype(np.float64)  # N,T,2 normalized yx
    pv = npy(r["pred_visibility"]).astype(bool)
    gv = npy(r["gt_visibility"]).astype(bool)
    osz = npy(r.get("original_size", [256, 256])).reshape(-1)
    h, w = int(osz[0]), int(osz[1])
    n, T = pred.shape[:2]
    q_t = np.clip(np.rint(q[:, 0]).astype(int), 0, T - 1)

    # Normalized yx query point -> pixel yx with same denom used by cache writers.
    q_yx_px = np.stack([q[:, 1] * max(h - 1, 1), q[:, 2] * max(w - 1, 1)], axis=-1)
    pred_q_yx_px = np.stack([pred[np.arange(n), q_t, 0] * max(h - 1, 1), pred[np.arange(n), q_t, 1] * max(w - 1, 1)], axis=-1)
    gt_q_yx_px = np.stack([gt[np.arange(n), q_t, 0] * max(h - 1, 1), gt[np.arange(n), q_t, 1] * max(w - 1, 1)], axis=-1)

    query_err = np.linalg.norm(pred_q_yx_px - q_yx_px, axis=-1)
    gt_query_err = np.linalg.norm(gt_q_yx_px - q_yx_px, axis=-1)
    pred_vs_gt_at_query_err = np.linalg.norm(pred_q_yx_px - gt_q_yx_px, axis=-1)

    pred_y_px = pred[..., 0] * max(h - 1, 1)
    pred_x_px = pred[..., 1] * max(w - 1, 1)
    gt_y_px = gt[..., 0] * max(h - 1, 1)
    gt_x_px = gt[..., 1] * max(w - 1, 1)
    all_err = np.sqrt((pred_y_px - gt_y_px) ** 2 + (pred_x_px - gt_x_px) ** 2)
    visible_err = all_err[gv]
    visible_and_pred_visible_err = all_err[gv & pv]

    oob = (pred[..., 0] < -1e-6) | (pred[..., 0] > 1 + 1e-6) | (pred[..., 1] < -1e-6) | (pred[..., 1] > 1 + 1e-6)
    near_border = (pred[..., 0] < 0.02) | (pred[..., 0] > 0.98) | (pred[..., 1] < 0.02) | (pred[..., 1] > 0.98)

    re_t = first_reentry_frames(gv, q_t)
    re_mask = re_t >= 0
    re_err = np.array([], dtype=np.float64)
    re_pv = np.array([], dtype=bool)
    if re_mask.any():
        idx = np.nonzero(re_mask)[0]
        tt = re_t[idx]
        pred_re = np.stack([pred[idx, tt, 0] * max(h - 1, 1), pred[idx, tt, 1] * max(w - 1, 1)], axis=-1)
        gt_re = np.stack([gt[idx, tt, 0] * max(h - 1, 1), gt[idx, tt, 1] * max(w - 1, 1)], axis=-1)
        re_err = np.linalg.norm(pred_re - gt_re, axis=-1)
        re_pv = pv[idx, tt]

    return {
        "video_id": str(r.get("video_id", "")),
        "n_queries": int(n),
        "frames": int(T),
        "original_size": [h, w],
        "query_frame_error_px": summarize_arr(query_err),
        "gt_query_frame_error_px": summarize_arr(gt_query_err),
        "pred_vs_gt_at_query_error_px": summarize_arr(pred_vs_gt_at_query_err),
        "query_frame_pred_visible_rate": round(float(np.mean(pv[np.arange(n), q_t])), 6),
        "query_frame_gt_visible_rate": round(float(np.mean(gv[np.arange(n), q_t])), 6),
        "pred_visibility_rate": round(float(np.mean(pv)), 6),
        "gt_visibility_rate": round(float(np.mean(gv)), 6),
        "pred_norm_range": {
            "y_min": round(float(np.nanmin(pred[..., 0])), 6),
            "y_max": round(float(np.nanmax(pred[..., 0])), 6),
            "x_min": round(float(np.nanmin(pred[..., 1])), 6),
            "x_max": round(float(np.nanmax(pred[..., 1])), 6),
        },
        "pred_pixel_range": {
            "y_min": round(float(np.nanmin(pred_y_px)), 6),
            "y_max": round(float(np.nanmax(pred_y_px)), 6),
            "x_min": round(float(np.nanmin(pred_x_px)), 6),
            "x_max": round(float(np.nanmax(pred_x_px)), 6),
        },
        "oob_rate_all_frames": round(float(np.mean(oob)), 8),
        "near_border_rate_all_frames": round(float(np.mean(near_border)), 8),
        "visible_gt_error_px": summarize_arr(visible_err),
        "visible_gt_and_pred_visible_error_px": summarize_arr(visible_and_pred_visible_err),
        "first_reentry": {
            "n_reentry_queries": int(re_mask.sum()),
            "pred_visible_rate_at_reentry": round(float(np.mean(re_pv)), 6) if re_pv.size else None,
            "error_px_at_reentry": summarize_arr(re_err),
            "lt_1px": round(float(np.mean(re_err < 1)), 6) if re_err.size else None,
            "lt_4px": round(float(np.mean(re_err < 4)), 6) if re_err.size else None,
            "lt_16px": round(float(np.mean(re_err < 16)), 6) if re_err.size else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    records = payload["records"]
    rows = [audit_record(r) for r in records]
    out = {
        "cache": args.cache,
        "model_name": payload.get("model_name", ""),
        "n_records": len(records),
        "records": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
