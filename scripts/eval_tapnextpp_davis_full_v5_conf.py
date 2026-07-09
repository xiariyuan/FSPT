#!/usr/bin/env python3
"""TAPNext++ DAVIS full eval v5-conf (GPU): save tracks, binary visibility, logits, and confidence.

This intentionally does NOT overwrite the v4 cache.
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official

device = torch.device("cuda")
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
    print("[{}/{}] {}...".format(vi+1, len(video_names), video_name), flush=True, end=" ")
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

    tr_all = torch.cat(tr_list, dim=1).squeeze(0).permute(1, 0, 2)  # (Q,T,2) [y,x]
    vl_all = torch.cat(vl_list, dim=1).squeeze(0).squeeze(-1).permute(1, 0)  # (Q,T)
    vl_binary = (vl_all > 0).bool()
    vl_logit_np = np.asarray(vl_all.float().numpy(), dtype=np.float32)
    vl_conf_np = np.asarray(torch.sigmoid(vl_all.float()).numpy(), dtype=np.float32)

    pred_xy_px = np.asarray(tr_all.numpy()[..., ::-1])          # (Q,T,2) [x,y] px
    pred_yx_norm = np.asarray(tr_all.numpy()) / PIX             # (Q,T,2) [y,x] norm
    pred_occ_np = np.asarray((~vl_binary).numpy())
    gt_yx_norm = gt_xy[..., ::-1] / PIX                         # (Q,T,2) [y,x] norm
    query_pts_norm = query_pts.copy()
    query_pts_norm[:, 1:] /= PIX

    # Official TAP-Vid
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
        ajrd = compute_redetection_metrics(pred_t_b, pred_v_b, gt_t_b, gt_v_b)
        ajrd_val = ajrd.get("AJ_RD")
    except Exception:
        ajrd_val = None

    per_video[video_name] = {
        "frames": T_, "queries": Q, "infer_sec": round(infer_sec, 3),
        "tapvid": tapvid,
        "aj_rd": float(ajrd_val) if ajrd_val is not None and ajrd_val == ajrd_val else None,
        "vis_conf_mean": round(float(np.mean(vl_conf_np)), 6),
        "vis_conf_std": round(float(np.std(vl_conf_np)), 6),
        "vis_logit_min": round(float(np.min(vl_logit_np)), 6),
        "vis_logit_max": round(float(np.max(vl_logit_np)), 6),
    }
    print("AJ={:.4f} OA={:.4f} {:.1f}fps".format(
        tapvid["average_jaccard"], tapvid["occlusion_accuracy"], T_/infer_sec), flush=True)

    cache_records.append({
        "video_id": video_name, "sequence_index": vi, "frame_count": T_,
        "query_points": query_pts_norm.astype(np.float32),
        "pred_tracks": pred_yx_norm.astype(np.float32),
        "pred_visibility": np.asarray(vl_binary.numpy()),
        "pred_vis_logit": vl_logit_np.astype(np.float32),
        "pred_vis_conf": vl_conf_np.astype(np.float32),
        "gt_tracks": gt_yx_norm.astype(np.float32),
        "gt_visibility": (~gt_occ).astype(bool),
        "original_size": np.array(entry["video"].shape[1:3], dtype=np.int32),
        "model_input_size": np.array([256, 256], dtype=np.int32),
        "raw_coordinate_note": "source_query_protocol=first, source_space=input256, source_track_format=xy, export_track_format=yx_normalized",
    })

# Save
aj_vals = [v["tapvid"]["average_jaccard"] for v in per_video.values()]
oa_vals = [v["tapvid"]["occlusion_accuracy"] for v in per_video.values()]
ajrd_vals = [v["aj_rd"] for v in per_video.values() if v["aj_rd"] is not None]

conf_means = [v["vis_conf_mean"] for v in per_video.values()]
conf_stds = [v["vis_conf_std"] for v in per_video.values()]

summary = {
    "num_videos": len(per_video), "total_frames": total_frames,
    "total_infer_sec": round(total_infer_sec, 2),
    "avg_fps": round(total_frames / total_infer_sec, 1),
    "aggregate": {
        "AJ": round(float(np.mean(aj_vals)) * 100, 2),
        "OA": round(float(np.mean(oa_vals)) * 100, 2),
        "AJ_RD": round(float(np.mean(ajrd_vals)), 4) if ajrd_vals else None,
    },
    "visibility_confidence": {
        "mean_video_mean": round(float(np.mean(conf_means)), 6) if conf_means else None,
        "mean_video_std": round(float(np.mean(conf_stds)), 6) if conf_stds else None,
    },
}

out_json = OUT_DIR / "tapnextpp_davis_full_eval_v5_conf.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

out_cache = OUT_DIR / "tapnextpp_davis_first_input_cache_v5_conf.pt"
torch.save({"model_name": "tapnextpp", "dataset_name": "tapvid_davis",
            "protocol": "first+input", "schema_version": "tapnextpp_v5_conf",
            "records": cache_records}, str(out_cache))

print("\n" + "=" * 50)
print("AJ={:.2f}% OA={:.2f}% AJ_RD={} fps={:.1f}".format(
    summary["aggregate"]["AJ"], summary["aggregate"]["OA"],
    summary["aggregate"]["AJ_RD"], summary["avg_fps"]))
print("Saved:", out_json)
print("Cache:", out_cache)
