#!/usr/bin/env python3
"""Quick 3-video TAPNext++ eval to create cache for ReEntry testing."""
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from tapnet.tapnext.tapnext_torch import TAPNext

device = torch.device("cpu")  # No GPU
print("Device:", device, flush=True)

print("Loading model...", flush=True)
model = TAPNext(image_size=(256, 256))
ckpt = torch.load(str(CKPT), map_location="cpu")
model.load_state_dict({k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()})
model.eval()
print("  Loaded", flush=True)

with open(str(DAVIS_PKL), "rb") as f:
    davis_data = pickle.load(f)

# Run on 3 diverse videos: one hard (bike-packing), one medium, one easy
test_videos = ["bike-packing", "car-roundabout", "soapbox"]
target_size = (256, 256)
PIX = 255.0

cache_records = []

for vi, video_name in enumerate(test_videos):
    print(f"\n[{vi+1}/3] {video_name}...", flush=True)
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
    gt_xy = tp_valid
    gt_occ = occ_valid
    Q = query_pts.shape[0]

    print(f"  {T_} frames, {Q} queries", flush=True)

    video = torch.from_numpy(video_np_norm[None, ...]).float()
    query_t = torch.from_numpy(query_pts[None, ...]).float()

    t0 = time.time()
    with torch.no_grad():
        tr, _, vl, state = model(video=video[:, :1], query_points=query_t)
        tr_list = [tr.cpu()]
        vl_list = [vl.cpu()]
        for f in range(1, T_):
            tr, _, vl, state = model(video=video[:, f:f+1], state=state)
            tr_list.append(tr.cpu())
            vl_list.append(vl.cpu())

    infer_sec = time.time() - t0
    print(f"  {infer_sec:.1f}s ({T_/infer_sec:.1f} fps)", flush=True)

    tr_all = torch.cat(tr_list, dim=1).squeeze(0).permute(1, 0, 2)
    vl_all = torch.cat(vl_list, dim=1).squeeze(0).squeeze(-1).permute(1, 0)
    vl_binary = (vl_all > 0).bool()

    pred_yx_norm = np.asarray(tr_all.numpy()) / PIX
    gt_yx_norm = gt_xy[..., ::-1] / PIX
    query_pts_norm = query_pts.copy()
    query_pts_norm[:, 1:] /= PIX

    cache_records.append({
        "video_id": video_name,
        "sequence_index": vi,
        "frame_count": T_,
        "query_points": query_pts_norm.astype(np.float32),
        "pred_tracks": pred_yx_norm.astype(np.float32),
        "pred_visibility": np.asarray(vl_binary.numpy()),
        "gt_tracks": gt_yx_norm.astype(np.float32),
        "gt_visibility": (~gt_occ).astype(bool),
        "original_size": np.array(entry["video"].shape[1:3], dtype=np.int32),
        "model_input_size": np.array([256, 256], dtype=np.int32),
        "raw_coordinate_note": "source_query_protocol=first, source_space=input256, source_track_format=xy, export_track_format=yx_normalized",
    })

out_cache = OUT_DIR / "tapnextpp_davis_3video_cache.pt"
torch.save({"model_name": "tapnextpp", "dataset_name": "tapvid_davis",
            "protocol": "first+input", "records": cache_records}, str(out_cache))
print(f"\nSaved: {out_cache}")
