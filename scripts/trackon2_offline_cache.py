#!/usr/bin/env python3
"""TrackOn2 offline cache: run forward() (batch mode, non-causal) on all 30 DAVIS videos."""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/gemini/code/FSPT")
TRACKON_DIR = ROOT / "baselines/track_on"
CKPT = TRACKON_DIR / "checkpoints_trackon2_dinov3.pt"
DAVIS_PKL = ROOT / "datasets/tapvid_davis/tapvid_davis.pkl"
OUT_DIR = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke"

# TrackOn2 imports use relative imports (model.xxx, utils.xxx)
# Run from track_on dir, and DON'T add FSPT root to sys.path (it shadows track_on's utils)
import os
os.chdir(str(TRACKON_DIR))
sys.path.insert(0, str(TRACKON_DIR))
# Only add FSPT root later for our own imports
sys.path.append(str(ROOT))

from model.trackon_predictor import Predictor

device = torch.device("cuda")
print("Device:", device, flush=True)

print("Loading TrackOn2 model...", flush=True)
predictor = Predictor(checkpoint_path=str(CKPT), support_grid_size=20)
predictor = predictor.to(device).eval()
print("  Loaded", flush=True)

with open(str(DAVIS_PKL), "rb") as f:
    davis_data = pickle.load(f)
video_names = sorted(davis_data.keys())

PIX = 255.0
offline_records = []

for vi, video_name in enumerate(video_names):
    print(f"[{vi+1}/30] {video_name}...", flush=True, end=" ")
    entry = davis_data[video_name]
    video_np = entry["video"].astype(np.float32)  # (T, H, W, 3) [0, 255]

    T_ = video_np.shape[0]
    H_in, W_in = video_np.shape[1], video_np.shape[2]

    # Prepare video tensor: (1, T, 3, H, W)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)  # (1, T, 3, H, W)

    # Query points
    points_np = entry["points"]      # (N, T, 2) normalized [0,1], (x, y)
    occluded_np = entry["occluded"]   # (N, T) bool

    target_points_px = points_np * PIX  # (N, T, 2) [x, y] pixel

    valid = np.sum(~occluded_np, axis=1) > 0
    tp_valid = target_points_px[valid]
    occ_valid = occluded_np[valid]

    query_pts_list = []
    for i in range(tp_valid.shape[0]):
        first_vis = np.where(occ_valid[i] == 0)[0][0]
        x, y = tp_valid[i, first_vis, 0], tp_valid[i, first_vis, 1]
        # TrackOn2 expects queries as (t, x, y) in pixel coords
        query_pts_list.append(np.array([first_vis, x, y], dtype=np.float32))
    query_pts = np.stack(query_pts_list, axis=0)  # (N, 3) [t, x, y]
    Q = query_pts.shape[0]

    t0 = time.time()
    try:
        # Offline mode: Track_On2.forward() processes all frames at once
        with torch.no_grad():
            # Use the model's forward directly (not Predictor.forward which is online)
            # But Predictor.forward IS online (frame-by-frame)
            # We need to call model.forward() directly for offline/batch mode

            # Scale queries to [0, 255] range for the model
            queries_t = torch.from_numpy(query_pts[None]).float().to(device)  # (1, N, 3)

            # Call the model's batch forward
            out = predictor.model(
                video=video,
                queries=queries_t,
            )

        # out is a dict: {"P": P, "V_logit": V_logit, ...}
        # P: (B, T, N, 2) in [x, y] pixel coords
        # V_logit: (B, T, N) visibility logits
        pred_traj = out["P"].cpu().numpy()[0]  # (T, N, 2)
        pred_vis_logit = out["V_logit"].cpu().numpy()[0]  # (T, N)

        infer_sec = time.time() - t0

        # Visibility: sigmoid >= delta_v (0.8)
        pred_vis = (1 / (1 + np.exp(-pred_vis_logit))) >= predictor.delta_v  # (T, N)

        # Convert to (N, T, 2) and (N, T)
        pred_traj_NT = pred_traj.transpose(1, 0, 2)  # (N, T, 2) [x, y]
        pred_vis_NT = pred_vis.transpose(1, 0)  # (N, T)

        # Normalize for cache (yx_normalized format)
        pred_yx_norm = pred_traj_NT[..., ::-1] / PIX  # [x,y]→[y,x] / 255
        gt_yx_norm = tp_valid[..., ::-1] / PIX
        query_pts_norm = query_pts.copy()
        query_pts_norm[:, 1:] /= PIX

        n_vis = int(pred_vis_NT.sum())
        print(f"Q={Q} T={T_} vis={n_vis} {infer_sec:.1f}s", flush=True)

        offline_records.append({
            "video_id": video_name, "sequence_index": vi, "frame_count": T_,
            "query_points": query_pts_norm.astype(np.float32),
            "pred_tracks": pred_yx_norm.astype(np.float32),
            "pred_visibility": pred_vis_NT.astype(bool),
            "gt_tracks": gt_yx_norm.astype(np.float32),
            "gt_visibility": (~occ_valid).astype(bool),
            "original_size": np.array([H_in, W_in], dtype=np.int32),
            "model_input_size": np.array([384, 512], dtype=np.int32),
            "raw_coordinate_note": "source_query_protocol=first, source_space=input256, source_track_format=xy, export_track_format=yx_normalized, mode=offline_batch",
        })

    except Exception as e:
        infer_sec = time.time() - t0
        print(f"ERROR: {e} ({infer_sec:.1f}s)", flush=True)
        import traceback
        traceback.print_exc()
        continue

out_path = OUT_DIR / "trackon2_davis_offline_batch_cache.pt"
torch.save({"model_name": "trackon2_dinov3_offline", "dataset_name": "tapvid_davis",
            "protocol": "first+input+offline_batch", "records": offline_records}, str(out_path))
print(f"\nSaved: {out_path}")
print(f"Records: {len(offline_records)}")
