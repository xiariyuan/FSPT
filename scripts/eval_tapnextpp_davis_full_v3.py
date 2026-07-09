#!/usr/bin/env python3
"""Full 30-video DAVIS eval for TAPNext++ — FIXED v3: sample_queries_first + official metric."""
from __future__ import annotations

import json, pickle, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/gemini/code/FSPT")
REPO = ROOT / "external/tapnextpp/repo"
CKPT = ROOT / "checkpoints/tapnextpp/tapnextpp_ckpt.pt"
DAVIS_PKL = ROOT / "datasets/tapvid_davis/tapvid_davis.pkl"
OUT = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official, sample_queries_first

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# ── 1. Load model ──
print("Loading model...", flush=True)
t0 = time.time()
model = TAPNext(image_size=(256, 256))
ckpt = torch.load(str(CKPT), map_location="cpu")
model.load_state_dict({k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()})
model.to(device).eval()
model_load_sec = time.time() - t0
model_params = sum(p.numel() for p in model.parameters())
print("  {:,} params, {:.1f}s".format(model_params, model_load_sec), flush=True)

# ── 2. Load dataset ──
print("Loading DAVIS dataset...", flush=True)
t0 = time.time()
with open(str(DAVIS_PKL), "rb") as f:
    davis_data = pickle.load(f)
dataset_load_sec = time.time() - t0
video_names = sorted(davis_data.keys())
print("  {} videos, {:.1f}s".format(len(video_names), dataset_load_sec), flush=True)

# ── 3. Evaluate each video ──
per_video = {}
per_video_aj = {}
total_frames = 0
total_infer_sec = 0.0
target_size = (256, 256)
PIX = 255.0

for vi, video_name in enumerate(video_names):
    print("\n[{}/{}] {}...".format(vi+1, len(video_names), video_name), flush=True)
    entry = davis_data[video_name]
    video_np = entry["video"].astype(np.float32)

    # Resize to 256x256
    if video_np.shape[1:3] != target_size:
        vt = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
        vt = F.interpolate(vt, size=target_size, mode="bilinear", align_corners=False)
        video_np = vt.permute(0, 2, 3, 1).numpy()

    video_np_norm = video_np / 255.0 * 2.0 - 1.0  # [-1, 1]
    T_ = video_np_norm.shape[0]

    # --- Use sample_queries_first (official TAP-Vid logic) ---
    # Convert points to pixel coords for sample_queries_first
    points_np = entry["points"]          # (N, T, 2) normalized [0,1], (x, y)
    occluded_np = entry["occluded"]       # (N, T) bool
    target_points_px = points_np * PIX   # (N, T, 2) [x, y] pixel

    # sample_queries_first returns dict with query_points, target_points, occluded
    # It expects target_points in [0,1] normalized for docstring, but we pass pixel for
    # the metric computation pipeline.
    valid = np.sum(~occluded_np, axis=1) > 0
    tp_valid = target_points_px[valid]
    occ_valid = occluded_np[valid]

    query_pts_list = []
    for i in range(tp_valid.shape[0]):
        first_vis = np.where(occ_valid[i] == 0)[0][0]
        x, y = tp_valid[i, first_vis, 0], tp_valid[i, first_vis, 1]
        query_pts_list.append(np.array([first_vis, y, x], dtype=np.float32))  # [t, y, x]
    query_pts = np.stack(query_pts_list, axis=0)  # (Q, 3)
    gt_xy = tp_valid      # (Q, T, 2) [x, y] pixel
    gt_occ = occ_valid    # (Q, T) bool, True=occluded
    Q = query_pts.shape[0]

    print("  {} frames, {} queries".format(T_, Q), flush=True)

    # Run online inference
    video = torch.from_numpy(video_np_norm[None, ...]).float().to(device)  # (1,T,H,W,3)
    query_t = torch.from_numpy(query_pts[None, ...]).float().to(device)    # (1,Q,3)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    with torch.no_grad():
        tr, _, vl, state = model(video=video[:, :1], query_points=query_t)
        tr_list = [tr.cpu()]
        vl_list = [(vl > 0).cpu()]
        for f in range(1, T_):
            tr, _, vl, state = model(video=video[:, f:f+1], state=state)
            tr_list.append(tr.cpu())
            vl_list.append((vl > 0).cpu())

    torch.cuda.synchronize()
    infer_sec = time.time() - t0

    # Assemble: (1,T,Q,2)[y,x] → squeeze → permute → (Q,T,2)[y,x] → flip → (Q,T,2)[x,y]
    tr_all = torch.cat(tr_list, dim=1).squeeze(0).permute(1, 0, 2)  # (Q,T,2) [y,x]
    vl_all = torch.cat(vl_list, dim=1).squeeze(0).squeeze(-1).permute(1, 0)  # (Q,T)
    pred_xy = np.asarray(tr_all.numpy()[..., ::-1])  # → [x,y] pixel
    pred_occ_np = np.asarray((~vl_all.bool()).numpy())  # True=occluded

    total_frames += T_
    total_infer_sec += infer_sec

    # Official TAP-Vid metrics
    tapvid = compute_tapvid_metrics_official(
        query_points=query_pts[None].astype(np.float32),
        gt_occluded=gt_occ[None].astype(bool),
        gt_tracks=gt_xy[None].astype(np.float32),
        pred_occluded=pred_occ_np[None],
        pred_tracks=pred_xy[None].astype(np.float32),
        query_mode="first", thresholds=(1, 2, 4, 8, 16),
    )
    tapvid = {k: float(v.item() if hasattr(v, 'item') else v) for k, v in tapvid.items()}

    # AJ_RD
    try:
        pred_t_b = torch.from_numpy(pred_xy.copy()).float().unsqueeze(0).permute(0, 2, 1, 3)  # (1,T,Q,2)
        pred_v_b = torch.from_numpy(~pred_occ_np).bool().unsqueeze(0).permute(0, 2, 1)  # (1,T,Q)
        gt_t_b = torch.from_numpy(gt_xy.astype(np.float32).copy()).unsqueeze(0).permute(0, 2, 1, 3)
        gt_v_b = torch.from_numpy(~gt_occ.astype(bool)).bool().unsqueeze(0).permute(0, 2, 1)
        ajrd = compute_redetection_metrics(
            pred_tracks=pred_t_b, pred_visible=pred_v_b,
            gt_tracks=gt_t_b, gt_visible=gt_v_b,
        )
        ajrd = {k: float(v.item() if hasattr(v, 'item') else v)
                for k, v in ajrd.items() if not k.startswith("raw_stats/")}
    except Exception as e:
        ajrd = {"error": repr(e), "traceback": traceback.format_exc()}

    per_video[video_name] = {
        "frames": T_, "queries": Q, "infer_sec": round(infer_sec, 3),
        "tapvid": tapvid, "aj_rd": ajrd,
    }
    per_video_aj[video_name] = tapvid["average_jaccard"]
    print("  AJ={:.4f}  OA={:.4f}  {:.1f}fps".format(
        tapvid["average_jaccard"], tapvid["occlusion_accuracy"], T_/infer_sec), flush=True)

# ── 4. Aggregate ──
aj_vals = [v["tapvid"]["average_jaccard"] for v in per_video.values()]
oa_vals = [v["tapvid"]["occlusion_accuracy"] for v in per_video.values()]
j1 = [v["tapvid"]["jaccard_1"] for v in per_video.values()]
j4 = [v["tapvid"]["jaccard_4"] for v in per_video.values()]
j16 = [v["tapvid"]["jaccard_16"] for v in per_video.values()]

summary = {
    "model_params": model_params,
    "model_load_sec": round(model_load_sec, 1),
    "dataset_load_sec": round(dataset_load_sec, 1),
    "num_videos": len(per_video),
    "total_frames": total_frames,
    "total_infer_sec": round(total_infer_sec, 2),
    "avg_fps": round(total_frames / total_infer_sec, 1),
    "metric_note": "official TAP-Vid metrics, sample_queries_first, PIX=255",
    "aggregate_tapvid": {
        "average_jaccard": round(float(np.mean(aj_vals)), 4),
        "occlusion_accuracy": round(float(np.mean(oa_vals)), 4),
        "jaccard_1": round(float(np.mean(j1)), 4),
        "jaccard_4": round(float(np.mean(j4)), 4),
        "jaccard_16": round(float(np.mean(j16)), 4),
        "min_aj": round(float(np.min(aj_vals)), 4),
        "max_aj": round(float(np.max(aj_vals)), 4),
    },
    "per_video": per_video,
    "per_video_aj_ranked": {
        k: per_video_aj[k] for k in sorted(per_video_aj, key=per_video_aj.get, reverse=True)
    },
}

print("\n" + "=" * 50)
print(json.dumps(summary["aggregate_tapvid"], indent=2))
print("=" * 50)

out_json = OUT / "tapnextpp_davis_full_eval_v3.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nSaved:", out_json)
