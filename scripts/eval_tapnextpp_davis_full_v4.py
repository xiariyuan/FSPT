#!/usr/bin/env python3
"""TAPNext++ DAVIS full eval v4: save prediction cache with visibility confidence."""
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
OUT_DIR = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

print("Loading model...", flush=True)
t0 = time.time()
model = TAPNext(image_size=(256, 256))
ckpt = torch.load(str(CKPT), map_location="cpu")
model.load_state_dict({k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()})
model.to(device).eval()
print("  {:,} params, {:.1f}s".format(sum(p.numel() for p in model.parameters()), time.time()-t0), flush=True)

print("Loading DAVIS...", flush=True)
with open(str(DAVIS_PKL), "rb") as f:
    davis_data = pickle.load(f)
video_names = sorted(davis_data.keys())
print("  {} videos".format(len(video_names)), flush=True)

per_video = {}
cache_records = []
total_frames = 0
total_infer_sec = 0.0
target_size = (256, 256)
PIX = 255.0

for vi, video_name in enumerate(video_names):
    print("\n[{}/{}] {}".format(vi+1, len(video_names), video_name), flush=True)
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
    target_points_px = points_np * PIX

    valid = np.sum(~occluded_np, axis=1) > 0
    tp_valid = target_points_px[valid]
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

    print("  {} frames, {} queries".format(T_, Q), flush=True)

    video = torch.from_numpy(video_np_norm[None, ...]).float().to(device)
    query_t = torch.from_numpy(query_pts[None, ...]).float().to(device)

    t0 = time.time()
    with torch.no_grad():
        tr, _, vl, state = model(video=video[:, :1], query_points=query_t)
        tr_list = [tr.cpu()]
        vl_list = [vl.cpu()]
        for f in range(1, T_):
            tr, _, vl, state = model(video=video[:, f:f+1], state=state)
            tr_list.append(tr.cpu())
            vl_list.append(vl.cpu())

    torch.cuda.synchronize()
    infer_sec = time.time() - t0
    total_frames += T_
    total_infer_sec += infer_sec

    # Assemble: (1,T,Q,2)[y,x] → squeeze → permute → (Q,T,2)[y,x]
    tr_all = torch.cat(tr_list, dim=1).squeeze(0).permute(1, 0, 2)  # (Q,T,2) [y,x]
    vl_all = torch.cat(vl_list, dim=1).squeeze(0).squeeze(-1).permute(1, 0)  # (Q,T)

    # Binary visibility
    vl_binary = (vl_all > 0).bool()
    # Confidence probability
    vl_conf = torch.sigmoid(vl_all.float())

    # Tracks: model outputs in pixel [y,x], normalize to [0,1] for cache
    pred_xy_px = np.asarray(tr_all.numpy()[..., ::-1])  # (Q,T,2) [x,y] px
    pred_yx_norm = np.asarray(tr_all.numpy()) / PIX  # (Q,T,2) [y,x] norm
    pred_occ_np = np.asarray((~vl_binary).numpy())
    pred_vis_conf_np = np.asarray(vl_conf.numpy())  # (Q,T) [0,1]

    # GT in normalized [y,x] for cache
    gt_yx_norm = gt_xy[..., ::-1] / PIX  # (Q,T,2) [x,y]→[y,x] px→norm

    # Query points in normalized [t, y, x] for cache
    query_pts_norm = query_pts.copy()
    query_pts_norm[:, 1:] /= PIX

    # Official TAP-Vid metrics
    tapvid = compute_tapvid_metrics_official(
        query_points=query_pts[None].astype(np.float32),
        gt_occluded=gt_occ[None].astype(bool),
        gt_tracks=gt_xy[None].astype(np.float32),
        pred_occluded=pred_occ_np[None],
        pred_tracks=pred_xy_px[None].astype(np.float32),
        query_mode="first", thresholds=(1, 2, 4, 8, 16),
    )
    tapvid = {k: float(v.item() if hasattr(v, 'item') else v) for k, v in tapvid.items()}

    # AJ_RD
    try:
        pred_t_b = torch.from_numpy(pred_xy_px.copy()).float().unsqueeze(0).permute(0, 2, 1, 3)
        pred_v_b = torch.from_numpy(~pred_occ_np).bool().unsqueeze(0).permute(0, 2, 1)
        gt_t_b = torch.from_numpy(gt_xy.astype(np.float32).copy()).unsqueeze(0).permute(0, 2, 1, 3)
        gt_v_b = torch.from_numpy(~gt_occ.astype(bool)).bool().unsqueeze(0).permute(0, 2, 1)
        ajrd = compute_redetection_metrics(
            pred_tracks=pred_t_b, pred_visible=pred_v_b,
            gt_tracks=gt_t_b, gt_visible=gt_v_b,
        )
        ajrd = {k: float(v.item() if hasattr(v, 'item') else v)
                for k, v in ajrd.items() if not k.startswith("raw_stats/")}
    except Exception as e:
        ajrd = {"error": repr(e)}

    per_video[video_name] = {
        "frames": T_, "queries": Q, "infer_sec": round(infer_sec, 3),
        "tapvid": tapvid, "aj_rd": ajrd,
        "vis_conf_mean": round(float(vl_conf.mean().item()), 4),
        "vis_conf_std": round(float(vl_conf.std().item()), 4),
    }
    print("  AJ={:.4f}  OA={:.4f}  {:.1f}fps".format(
        tapvid["average_jaccard"], tapvid["occlusion_accuracy"], T_/infer_sec), flush=True)

    # Save cache record
    cache_records.append({
        "video_id": video_name,
        "sequence_index": vi,
        "frame_count": T_,
        "query_points": query_pts_norm.astype(np.float32),
        "pred_tracks": pred_yx_norm.astype(np.float32),
        "pred_visibility": (~pred_occ_np).astype(bool),
        "pred_vis_conf": pred_vis_conf_np.astype(np.float32),
        "gt_tracks": gt_yx_norm.astype(np.float32),
        "gt_visibility": (~gt_occ).astype(bool),
        "original_size": np.array(entry["video"].shape[1:3], dtype=np.int32),
        "model_input_size": np.array([256, 256], dtype=np.int32),
        "raw_coordinate_note": "source_query_protocol=first, source_space=input256, source_track_format=xy, export_track_format=yx_normalized",
    })

# ── Save ──
aj_vals = [v["tapvid"]["average_jaccard"] for v in per_video.values()]
oa_vals = [v["tapvid"]["occlusion_accuracy"] for v in per_video.values()]

summary = {
    "num_videos": len(per_video),
    "total_frames": total_frames,
    "total_infer_sec": round(total_infer_sec, 2),
    "avg_fps": round(total_frames / total_infer_sec, 1),
    "aggregate_tapvid": {
        "average_jaccard": round(float(np.mean(aj_vals)), 4),
        "occlusion_accuracy": round(float(np.mean(oa_vals)), 4),
    },
    "per_video": per_video,
}

out_json = OUT_DIR / "tapnextpp_davis_full_eval_v4.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

out_cache = OUT_DIR / "tapnextpp_davis_first_input_cache_v4.pt"
torch.save({"model_name": "tapnextpp", "dataset_name": "tapvid_davis",
            "protocol": "first+input", "records": cache_records,
            "schema_version": "attempt0_v4"}, str(out_cache))

print("\n" + "=" * 50)
print("AJ={:.2f}%  OA={:.2f}%  fps={:.1f}".format(
    np.mean(aj_vals)*100, np.mean(oa_vals)*100, total_frames/total_infer_sec))
print("Saved: {}".format(out_json))
print("Cache: {}".format(out_cache))
