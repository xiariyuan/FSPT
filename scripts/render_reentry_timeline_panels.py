#!/usr/bin/env python3
"""Render timeline panels for qualitative ReEntry analysis.

This renderer does not require raw video frames. It visualizes:

1. GT / Base / Ours-Det / Ours-Learned visibility states around re-entry.
2. Coordinate error of the preserved base coordinate channel in 256-space.

Method visibility rows use four states:
  0: invisible / true negative
  1: correct visible (pred visible and GT visible)
  2: false visible (pred visible while GT invisible)
  3: missed visible (pred invisible while GT visible)

GT row uses 0 for invisible and 1 for visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_aj_rd_from_cache import compute_reentry_metrics  # noqa: E402
from utils.coords import yx_norm_to_xy_256  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def find_record(payload: Dict[str, Any], video_id: str) -> Dict[str, Any]:
    for r in payload["records"]:
        if str(r["video_id"]) == str(video_id):
            return r
    raise KeyError(f"video_id not found: {video_id}")


def query_ajrd_for_record(record: Dict[str, Any], query_idx: int) -> float | None:
    h, w = int(npy(record["original_size"])[0]), int(npy(record["original_size"])[1])
    m = compute_reentry_metrics(
        pred_tracks=npy(record["pred_tracks"], np.float32),
        gt_tracks=npy(record["gt_tracks"], np.float32),
        pred_vis=npy(record["pred_visibility"], bool),
        gt_vis=npy(record["gt_visibility"], bool),
        query_points=npy(record["query_points"], np.float32),
        height=h,
        width=w,
    )
    for q in m.get("per_query", []):
        if int(q["query_idx"]) == int(query_idx):
            v = q.get("ajrd_summary_256", {}).get("aj_rd")
            return None if v is None else float(v)
    return None


def method_state(pred_vis: np.ndarray, gt_vis: np.ndarray) -> np.ndarray:
    pred_vis = np.asarray(pred_vis, dtype=bool)
    gt_vis = np.asarray(gt_vis, dtype=bool)
    out = np.zeros(pred_vis.shape[0], dtype=np.int32)
    out[pred_vis & gt_vis] = 1
    out[pred_vis & ~gt_vis] = 2
    out[(~pred_vis) & gt_vis] = 3
    return out


def render_case(
    *,
    video_id: str,
    query_idx: int,
    query_t: int,
    reentry_t: int,
    frame_lo: int,
    frame_hi: int,
    title: str,
    base_cache: Path,
    det_cache: Path,
    learned_cache: Path,
    out_png: Path,
    rule_cache: Path | None = None,
) -> Dict[str, Any]:
    payloads = {
        "Base": load_cache(base_cache),
        "Ours-Det": load_cache(det_cache),
        "Ours-Learned": load_cache(learned_cache),
    }
    if rule_cache is not None:
        payloads["Rule W8P2"] = load_cache(rule_cache)

    records = {name: find_record(payload, video_id) for name, payload in payloads.items()}
    base_rec = records["Base"]
    qidx = int(query_idx)
    gt_vis = npy(base_rec["gt_visibility"], bool)[qidx]
    gt_xy = yx_norm_to_xy_256(npy(base_rec["gt_tracks"], np.float32)[qidx])
    base_xy = yx_norm_to_xy_256(npy(base_rec["pred_tracks"], np.float32)[qidx])
    coord_err = np.linalg.norm(base_xy - gt_xy, axis=-1)

    n_frames = int(gt_vis.shape[0])
    lo = max(0, int(frame_lo))
    hi = min(n_frames - 1, int(frame_hi))
    frames = np.arange(lo, hi + 1)

    row_order = ["GT", "Base", "Ours-Det", "Ours-Learned"]
    if rule_cache is not None:
        row_order.insert(2, "Rule W8P2")
    mat = []
    gt_row = np.asarray(gt_vis[frames], dtype=np.int32)
    mat.append(gt_row)
    for name in row_order[1:]:
        pvis = npy(records[name]["pred_visibility"], bool)[qidx]
        mat.append(method_state(pvis, gt_vis)[frames])
    mat = np.stack(mat, axis=0)

    ajrd = {name: query_ajrd_for_record(records[name], qidx) for name in records}

    # Colors: 0 invisible/TN, 1 correct visible, 2 false visible, 3 missed visible.
    cmap = ListedColormap(["#e5e5e5", "#2ca02c", "#d62728", "#1f77b4"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig_h = 4.8 if rule_cache is None else 5.3
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(13.5, fig_h),
        gridspec_kw={"height_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )

    ax0.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax0.set_yticks(np.arange(len(row_order)))
    ax0.set_yticklabels(row_order, fontsize=10)
    ax0.set_xticks(np.arange(len(frames)))
    ax0.set_xticklabels([str(int(f)) for f in frames], rotation=90, fontsize=7)
    ax0.set_ylabel("Visibility state")
    ax0.set_title(title, fontsize=12, fontweight="bold")

    # Mark query and re-entry times if visible in range.
    for t, label, color in [(query_t, "query", "#8c564b"), (reentry_t, "re-entry", "#9467bd")]:
        if lo <= int(t) <= hi:
            x = int(t) - lo
            ax0.axvline(x=x, color=color, linewidth=1.5, linestyle="--")
            ax0.text(x + 0.1, -0.65, label, color=color, fontsize=8, rotation=0)
            ax1.axvline(x=int(t), color=color, linewidth=1.5, linestyle="--")

    legend_items = [
        Patch(facecolor="#e5e5e5", edgecolor="black", label="invisible / true negative"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="correct visible"),
        Patch(facecolor="#d62728", edgecolor="black", label="false visible"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="missed visible"),
    ]
    ax0.legend(handles=legend_items, loc="upper right", fontsize=8, ncol=2, frameon=True)

    ax1.plot(frames, coord_err[frames], linewidth=2.0, label="base coordinate error")
    for thr in [4, 8, 16]:
        ax1.axhline(y=thr, linestyle=":", linewidth=1.0, label=f"{thr}px threshold" if thr == 4 else None)
        ax1.text(frames[-1] + 0.15, thr, f"{thr}px", fontsize=8, va="center")
    ax1.set_xlim(frames[0], frames[-1])
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Coord. error (256-space px)")
    ax1.set_title("Preserved base coordinate channel", fontsize=10)
    ax1.grid(True, linewidth=0.3, alpha=0.5)

    ajrd_text = " | ".join([f"{k}: AJRD={v:.4f}" if v is not None else f"{k}: AJRD=NA" for k, v in ajrd.items()])
    fig.text(0.01, 0.01, ajrd_text, fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    sidecar = out_png.with_suffix(".json")
    summary = {
        "video_id": video_id,
        "query_idx": int(query_idx),
        "query_t": int(query_t),
        "reentry_t": int(reentry_t),
        "frame_lo": int(lo),
        "frame_hi": int(hi),
        "out_png": str(out_png),
        "ajrd": ajrd,
        "row_order": row_order,
    }
    sidecar.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render ReEntry timeline qualitative panel.")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--query-idx", type=int, required=True)
    ap.add_argument("--query-t", type=int, required=True)
    ap.add_argument("--reentry-t", type=int, required=True)
    ap.add_argument("--frame-lo", type=int, required=True)
    ap.add_argument("--frame-hi", type=int, required=True)
    ap.add_argument("--title", default="Re-entry qualitative case")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--det-cache", required=True)
    ap.add_argument("--learned-cache", required=True)
    ap.add_argument("--rule-cache", default="")
    ap.add_argument("--out-png", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = render_case(
        video_id=args.video_id,
        query_idx=args.query_idx,
        query_t=args.query_t,
        reentry_t=args.reentry_t,
        frame_lo=args.frame_lo,
        frame_hi=args.frame_hi,
        title=args.title,
        base_cache=Path(args.base_cache),
        det_cache=Path(args.det_cache),
        learned_cache=Path(args.learned_cache),
        rule_cache=Path(args.rule_cache) if args.rule_cache else None,
        out_png=Path(args.out_png),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
