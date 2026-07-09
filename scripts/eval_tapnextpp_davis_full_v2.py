#!/usr/bin/env python3
"""Full 30-video DAVIS evaluation for TAPNext++ (FIXED: official TAP-Vid metrics)."""
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
sys.path.insert(0, str(ROOT))  # for datasets.tapvid_official_eval

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── 1. Load model ──
print("Loading model...", flush=True)
t0 = time.time()
model = TAPNext(image_size=(256, 256))
ckpt = torch.load(str(CKPT), map_location="cpu")
model.load_state_dict({
    k.replace("tapnext.", ""): v
    for k, v in ckpt["state_dict"].items()
})
model.to(device).eval()
model_load_sec = time.time() - t0
model_params = sum(p.numel() for p in model.parameters())
print(f"  {model_params:,} params, {model_load_sec:.1f}s", flush=True)

# ── 2. Load dataset ──
print("Loading DAVIS dataset...", flush=True)
t0 = time.time()
with open(str(DAVIS_PKL), "rb") as f:
    davis_data = pickle.load(f)
dataset_load_sec = time.time() - t0
video_names = list(davis_data.keys())
print(f"  {len(video_names)} videos, {dataset_load_sec:.1f}s", flush=True)

# ── 3. Evaluate each video ──
per_video = {}
per_video_aj = {}
total_frames = 0
total_infer_sec = 0.0
target_size = (256, 256)
H256, W256 = target_size
PIX_SCALE = 255.0  # official TAP-Vid convention: 1.0 → (size-1)

for vi, video_name in enumerate(video_names):
    print(f"\n[{vi+1}/{len(video_names)}] {video_name}...", flush=True)
    entry = davis_data[video_name]
    video_np = entry["video"].astype(np.float32)

    # Resize
    if video_np.shape[1:3] != target_size:
        vt = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
        vt = F.interpolate(vt, size=target_size, mode="bilinear",
                           align_corners=False)
        video_np = vt.permute(0, 2, 3, 1).numpy()

    video_np = video_np / 255.0 * 2.0 - 1.0
    T_ = video_np.shape[0]
    points_np = entry["points"]          # (N, T, 2) normalized [0,1], (x, y)
    occluded_np = entry["occluded"]       # (N, T) bool

    # GT in pixel coords [x, y] using official convention: norm * 255
    gt_xy = points_np * PIX_SCALE  # (N, T, 2)

    # First-frame queries: only points visible at t=0
    N = points_np.shape[0]
    query_mask = ~occluded_np[:, 0]
    if not query_mask.any():
        query_mask = np.ones(N, dtype=bool)

    # Query points in pixel coords [t, y, x], with official convention
    query_pts = np.stack([
        np.zeros(N, dtype=np.float32),              # t=0
        points_np[:, 0, 1] * PIX_SCALE,              # y pixel
        points_np[:, 0, 0] * PIX_SCALE,              # x pixel
    ], axis=-1)[query_mask]  # (Q, 3)

    gt_xy = gt_xy[query_mask]      # (Q, T, 2) [x, y] pixel
    gt_occ = occluded_np[query_mask]  # (Q, T) bool, True=occluded
    Q = query_pts.shape[0]

    if Q == 0:
        print(f"  No valid queries, skip", flush=True)
        continue

    video = torch.from_numpy(video_np[None, ...]).to(device)  # (1, T, H, W, 3)
    query_t = torch.from_numpy(query_pts[None, ...]).to(device)  # (1, Q, 3)

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

    # Assemble → (Q, T, 2) [y, x] model output → flip to [x, y]
    tr_all = torch.cat(tr_list, dim=1).squeeze(0).permute(1, 0, 2)  # (Q, T, 2) [y,x]
    vl_all = torch.cat(vl_list, dim=1).squeeze(0).squeeze(-1).permute(1, 0)  # (Q, T)
    occ_pred = ~vl_all.bool()

    # Flip model output [y,x] → [x,y] for TAP-Vid official convention
    pred_xy = np.asarray(tr_all.numpy()[..., ::-1])   # (Q, T, 2) [x, y] pixel
    pred_occ_np = np.asarray(occ_pred.numpy())          # (Q, T) bool, True=occluded

    total_frames += T_
    total_infer_sec += infer_sec

    # ── Official TAP-Vid metrics ──
    # Function expects: (B, N, T, 2) [x,y] pixel, (B, N, T) bool occluded,
    #                   (B, N, 3) [t,y,x] pixel, query_mode='first'
    tapvid = compute_tapvid_metrics_official(
        query_points=query_pts[None, ...].astype(np.float32),
        gt_occluded=gt_occ[None, ...].astype(bool),
        gt_tracks=gt_xy[None, ...].astype(np.float32),
        pred_occluded=pred_occ_np[None, ...],
        pred_tracks=pred_xy[None, ...].astype(np.float32),
        query_mode="first",
        thresholds=(1, 2, 4, 8, 16),
    )
    # tapvid returns arrays of shape (1,) — extract scalars
    tapvid = {k: float(v.item() if hasattr(v, 'item') else v)
              for k, v in tapvid.items()}

    # ── AJ_RD metrics (TAPNext++ implementation) ──
    try:
        # TAPNext++ AJ_RD expects (B, T, N, 2) [x,y] pixel, (B, T, N) bool visible
        pred_tracks_b = torch.from_numpy(pred_xy.copy()).float().unsqueeze(0).permute(0, 2, 1, 3)  # (1,T,Q,2)
        pred_visible_b = torch.from_numpy(~pred_occ_np).bool().unsqueeze(0).permute(0, 2, 1)  # (1,T,Q)
        gt_tracks_b = torch.from_numpy(gt_xy.astype(np.float32).copy()).unsqueeze(0).permute(0, 2, 1, 3)
        gt_visible_b = torch.from_numpy(~gt_occ.astype(bool)).bool().unsqueeze(0).permute(0, 2, 1)

        ajrd = compute_redetection_metrics(
            pred_tracks=pred_tracks_b,
            pred_visible=pred_visible_b,
            gt_tracks=gt_tracks_b,
            gt_visible=gt_visible_b,
        )
        ajrd = {k: float(v.item() if hasattr(v, 'item') else v)
                for k, v in ajrd.items()
                if not k.startswith("raw_stats/")}
    except Exception as e:
        ajrd = {"error": repr(e), "traceback": traceback.format_exc()}

    per_video[video_name] = {
        "frames": T_,
        "queries": Q,
        "infer_sec": round(infer_sec, 3),
        "tapvid": tapvid,
        "aj_rd": ajrd,
    }
    per_video_aj[video_name] = tapvid["average_jaccard"]
    print(f"  AJ={tapvid['average_jaccard']:.4f}  OA={tapvid['occlusion_accuracy']:.4f}  "
          f"{T_/infer_sec:.1f}fps", flush=True)

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
    "metric_note": "official TAP-Vid metrics (tapvid_official_eval), GT*255 convention",
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

out_json = OUT / "tapnextpp_davis_full_eval_official.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nSaved: {out_json}")
