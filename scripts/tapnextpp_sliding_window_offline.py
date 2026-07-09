#!/usr/bin/env python3
"""TAPNext++ sliding-window offline: window size W, reset state between windows.
Compare with online baseline. Then apply ReEntry.
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/gemini/code/FSPT")
REPO = ROOT / "external/tapnextpp/repo"
CKPT = ROOT / "checkpoints/tapnextpp/tapnextpp_ckpt.pt"
DAVIS_PKL = ROOT / "datasets/tapvid_davis/tapvid_davis.pkl"
OUT_DIR = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official

device = torch.device("cuda")
model = TAPNext(image_size=(256, 256))
ckpt = torch.load(str(CKPT), map_location="cpu")
model.load_state_dict({k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()})
model.to(device).eval()
print("Model loaded", flush=True)

with open(str(DAVIS_PKL), "rb") as f:
    davis_data = pickle.load(f)
video_names = sorted(davis_data.keys())

target_size = (256, 256)
PIX = 255.0
WINDOW = 8  # sliding window size

offline_records = []
baseline_records = []

for vi, video_name in enumerate(video_names):
    entry = davis_data[video_name]
    video_np = entry["video"].astype(np.float32)
    if video_np.shape[1:3] != target_size:
        vt = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
        vt = F.interpolate(vt, size=target_size, mode="bilinear", align_corners=False)
        video_np = vt.permute(0, 2, 3, 1).numpy()
    video_np_norm = video_np / 255.0 * 2.0 - 1.0
    T_ = video_np_norm.shape[0]

    points_np = entry["points"]
    occluded_np = entry["occluded"]
    tp_px = points_np * PIX
    valid = np.sum(~occluded_np, axis=1) > 0
    tp_valid = tp_px[valid]
    occ_valid = occluded_np[valid]

    query_pts_list = []
    for i in range(tp_valid.shape[0]):
        first_vis = np.where(occ_valid[i] == 0)[0][0]
        x, y = tp_valid[i, first_vis, 0], tp_valid[i, first_vis, 1]
        query_pts_list.append(np.array([first_vis, y, x], dtype=np.float32))
    query_pts = np.stack(query_pts_list, axis=0)
    Q = query_pts.shape[0]

    video = torch.from_numpy(video_np_norm[None, ...]).float().to(device)
    query_t = torch.from_numpy(query_pts[None, ...]).float().to(device)

    # Baseline: online
    with torch.no_grad():
        tr, _, vl, state = model(video=video[:, :1], query_points=query_t)
        base_tr = [tr.cpu()]
        base_vl = [vl.cpu()]
        for f in range(1, T_):
            tr, _, vl, state = model(video=video[:, f:f+1], state=state)
            base_tr.append(tr.cpu())
            base_vl.append(vl.cpu())
    base_tr = torch.cat(base_tr, dim=1).squeeze(0).permute(1, 0, 2)
    base_vl = torch.cat(base_vl, dim=1).squeeze(0).squeeze(-1).permute(1, 0)
    base_vis = (base_vl > 0).bool()

    # Offline: sliding window W=8, reset state
    with torch.no_grad():
        off_tr_list = []
        off_vl_list = []
        for w_start in range(0, T_, WINDOW):
            w_end = min(w_start + WINDOW, T_)
            w_len = w_end - w_start
            # Reset state at start of each window
            # Adjust query points for window offset
            w_qpts = query_t.clone()
            w_qpts[..., 0] = w_qpts[..., 0] - w_start  # adjust query time
            # Clamp negative query times to 0 (query before window start)
            neg_mask = w_qpts[..., 0] < 0
            w_qpts[..., 0] = w_qpts[..., 0].clamp(min=0)

            w_video = video[:, w_start:w_end]
            tr, _, vl, w_state = model(video=w_video[:, :1], query_points=w_qpts)
            w_tr = [tr.cpu()]
            w_vl = [vl.cpu()]
            for f in range(1, w_len):
                tr, _, vl, w_state = model(video=w_video[:, f:f+1], state=w_state)
                w_tr.append(tr.cpu())
                w_vl.append(vl.cpu())

            off_tr_list.extend(w_tr)
            off_vl_list.extend(w_vl)

    off_tr = torch.cat(off_tr_list, dim=1).squeeze(0).permute(1, 0, 2)
    off_vl = torch.cat(off_vl_list, dim=1).squeeze(0).squeeze(-1).permute(1, 0)
    off_vis = (off_vl > 0).bool()

    # Stats
    vis_off = int(off_vis.sum())
    vis_base = int(base_vis.sum())
    recovered = int((base_vis & ~off_vis).sum())
    dropped = int((off_vis & ~base_vis).sum())

    print(f"[{vi+1:2d}/30] {video_name:25s} Q={Q:3d} T={T_:3d} "
          f"off_vis={vis_off:5d} base_vis={vis_base:5d} recovered={recovered:4d} dropped={dropped:4d}", flush=True)

    # Save
    off_tr_norm = np.asarray(off_tr.numpy()) / PIX
    base_tr_norm = np.asarray(base_tr.numpy()) / PIX
    gt_yx_norm = tp_valid[..., ::-1] / PIX
    query_pts_norm = query_pts.copy()
    query_pts_norm[:, 1:] /= PIX

    offline_records.append({
        "video_id": video_name, "sequence_index": vi, "frame_count": T_,
        "query_points": query_pts_norm.astype(np.float32),
        "pred_tracks": off_tr_norm.astype(np.float32),
        "pred_visibility": np.asarray(off_vis.numpy()),
        "gt_tracks": gt_yx_norm.astype(np.float32),
        "gt_visibility": (~occ_valid).astype(bool),
        "original_size": np.array(entry["video"].shape[1:3], dtype=np.int32),
        "model_input_size": np.array([256, 256], dtype=np.int32),
    })
    baseline_records.append({
        "video_id": video_name, "sequence_index": vi, "frame_count": T_,
        "query_points": query_pts_norm.astype(np.float32),
        "pred_tracks": base_tr_norm.astype(np.float32),
        "pred_visibility": np.asarray(base_vis.numpy()),
        "gt_tracks": gt_yx_norm.astype(np.float32),
        "gt_visibility": (~occ_valid).astype(bool),
        "original_size": np.array(entry["video"].shape[1:3], dtype=np.int32),
        "model_input_size": np.array([256, 256], dtype=np.int32),
    })

off_path = OUT_DIR / "tapnextpp_davis_offline_w8_cache.pt"
base_path = OUT_DIR / "tapnextpp_davis_baseline_w8_cache.pt"
torch.save({"model_name": "tapnextpp_offline_w8", "dataset_name": "tapvid_davis",
            "protocol": "first+input+offline_w8", "records": offline_records}, str(off_path))
torch.save({"model_name": "tapnextpp_baseline_w8", "dataset_name": "tapvid_davis",
            "protocol": "first+input+baseline", "records": baseline_records}, str(base_path))
print(f"\nSaved: {off_path}")
print(f"Saved: {base_path}")
