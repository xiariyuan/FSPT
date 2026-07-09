#!/usr/bin/env python3
"""TAPNext++ evaluation on one DAVIS video with TAP-Vid + AJ_RD metrics."""
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

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics

# Vendored TAP-Vid metrics (no tensorflow dependency)
# Copied from: tapvid/evaluation_datasets.py
def compute_tapvid_metrics(
    query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, query_mode,
    thresholds=(1, 2, 4, 8, 16), get_trackwise_metrics=False,
):
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)
    metrics = {}
    T_ = gt_tracks.shape[2]
    eye = np.eye(T_, dtype=np.int32)
    if query_mode == "first":
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError(f"Unknown query mode {query_mode}")
    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points, axis=summing_axis
    ) / np.maximum(np.sum(evaluation_points, axis=summing_axis), 1e-8)
    metrics["occlusion_accuracy"] = occ_acc
    visible = ~gt_occluded
    pred_visible = ~pred_occluded
    all_jaccard = []
    for thresh in thresholds:
        within_dist = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < thresh ** 2
        is_correct = within_dist & visible
        tp = np.sum(is_correct & pred_visible, axis=summing_axis)
        fp = np.sum((~is_correct) & pred_visible, axis=summing_axis) + np.sum(
            (~visible) & pred_visible, axis=summing_axis
        )
        fn = np.sum(is_correct & (~pred_visible), axis=summing_axis)
        jaccard = tp / np.maximum(tp + fp + fn, 1e-8)
        all_jaccard.append(jaccard)
        metrics[f"jaccard_{thresh}"] = jaccard
    metrics["average_jaccard"] = np.mean(np.stack(all_jaccard), axis=0)
    metrics["average_pts"] = 0.0
    return metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

summary = {}

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
summary["model_load_sec"] = round(time.time() - t0, 2)
summary["model_params"] = sum(p.numel() for p in model.parameters())
print(f"  Model: {summary['model_params']:,} params", flush=True)

# ── 2. Load DAVIS dataset (first video only) ──
print(f"Loading DAVIS dataset...", flush=True)
t0 = time.time()
with open(str(DAVIS_PKL), "rb") as f:
    davis_data = pickle.load(f)
summary["dataset_load_sec"] = round(time.time() - t0, 2)
summary["num_videos"] = len(davis_data)
print(f"  Dataset: {summary['num_videos']} videos loaded", flush=True)

video_name = list(davis_data.keys())[0]
print(f"  Eval video: {video_name}", flush=True)

entry = davis_data[video_name]
video_np = entry["video"].astype(np.float32)  # (T, H, W, 3), uint8 [0,255]
orig_h, orig_w = video_np.shape[1:3]

# Resize to 256x256 if needed (using simple bilinear via torch)
target_size = (256, 256)
T_ = video_np.shape[0]
if video_np.shape[1:3] != target_size:
    # torch interpolation expects (N, C, H, W)
    vt = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
    vt = F.interpolate(vt, size=target_size, mode="bilinear", align_corners=False)
    video_np = vt.permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)

video_np = video_np / 255.0 * 2.0 - 1.0  # [0,255] → [-1, 1]
T_ = video_np.shape[0]

points_np = entry["points"]  # (N, T, 2) normalized [0,1] coords as (x, y)
occluded_np = entry["occluded"]  # (N, T) bool

# Convert GT to pixel coords at model resolution
gt_xy = points_np * np.array([target_size[1], target_size[0]])  # (N, T_orig, 2) = (x, y)
# If video was resized, interpolate GT visibility but keep original point count
if points_np.shape[1] != T_:
    # TAP-Vid DAVIS points cover full original video length.
    # After resize the number of frames stays the same (just spatial resize).
    pass  # T_ should match points_np.shape[1] since resize is spatial only

# First-frame query: all points visible at t=0
N = points_np.shape[0]
query_mask = ~occluded_np[:, 0]
if not query_mask.any():
    query_mask = np.ones(N, dtype=bool)

# Query points must be in PIXEL coordinates [t, y, x] — model divides by image_size internally
H256, W256 = target_size
y_px = points_np[:, 0, 1] * (H256 - 1)  # y pixel (0..255)
x_px = points_np[:, 0, 0] * (W256 - 1)  # x pixel (0..255)
query_pts = np.stack([
    np.zeros(N, dtype=np.float32),               # t = 0 (frame index)
    y_px,
    x_px,
], axis=-1)  # (N, 3) pixel coords

query_pts = query_pts[query_mask]   # (Q, 3)
gt_xy = gt_xy[query_mask]           # (Q, T, 2)
gt_occ = occluded_np[query_mask]    # (Q, T)
Q = query_pts.shape[0]
print(f"  Video: {T_} frames, {Q} valid queries", flush=True)

# ── 3. Online inference ──
print("Running online inference...", flush=True)
video = torch.from_numpy(video_np[None, ...]).to(device)  # (1, T, H, W, 3)
query_pts_t = torch.from_numpy(query_pts[None, ...]).to(device)  # (1, Q, 3)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()

with torch.no_grad():
    tracks, track_logits, visible_logits, state = model(
        video=video[:, :1], query_points=query_pts_t
    )
    pred_tracks_list = [tracks.cpu()]
    pred_vis_list = [(visible_logits > 0).cpu()]

    for f in range(1, T_):
        curr_tracks, curr_logits, curr_visible, state = model(
            video=video[:, f:f+1], state=state,
        )
        pred_tracks_list.append(curr_tracks.cpu())
        pred_vis_list.append((curr_visible > 0).cpu())

torch.cuda.synchronize()
infer_sec = time.time() - t0
peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

# Assemble: (1, T, Q, *) → squeeze batch
pred_tracks_256 = torch.cat(pred_tracks_list, dim=1).squeeze(0)  # (T, Q, 2)
pred_visible = torch.cat(pred_vis_list, dim=1).squeeze(0)        # (T, Q, 1)
pred_occ = ~pred_visible.squeeze(-1).bool()                        # (T, Q)

# Transpose to (Q, T, *) for metric APIs
pred_tracks_q = pred_tracks_256.permute(1, 0, 2)  # (Q, T, 2)
pred_occ_q = pred_occ.permute(1, 0)                # (Q, T)

summary["inference"] = {
    "time_sec": round(infer_sec, 3),
    "fps": round(T_ / infer_sec, 1),
    "frames": T_,
    "queries": Q,
    "peak_mem_mb": round(peak_mem_mb, 1),
}
print(f"  Inference: {infer_sec:.2f}s, {T_/infer_sec:.1f} fps, "
      f"{peak_mem_mb:.0f} MB peak", flush=True)

# ── 4. TAP-Vid metrics ──
print("Computing TAP-Vid metrics...", flush=True)
query_np = query_pts[None, ...]                    # (1, Q, 3)
gt_occ_np = gt_occ[None, ...].astype(bool)          # (1, Q, T)
gt_xy_np = np.asarray(gt_xy[None, ...])              # (1, Q, T, 2)
pred_occ_np = np.asarray(pred_occ_q[None, ...])      # (1, Q, T)
pred_xy_np = np.asarray(pred_tracks_q.numpy()[None, ..., ::-1])  # flip (x,y)→(y,x) per official colab

tapvid_results = compute_tapvid_metrics(
    query_points=query_np,
    gt_occluded=gt_occ_np,
    gt_tracks=gt_xy_np,
    pred_occluded=pred_occ_np,
    pred_tracks=pred_xy_np,
    query_mode="first",
)
summary["tapvid"] = {k: float(v) for k, v in tapvid_results.items()}
print(f"  AJ={summary['tapvid'].get('average_jaccard', 'N/A'):.4f}, "
      f"OA={summary['tapvid'].get('occlusion_accuracy', 'N/A'):.4f}", flush=True)

# ── 5. AJ_RD metrics ──
print("Computing AJ_RD metrics...", flush=True)
try:
    ajrd_result = compute_redetection_metrics(
        pred_tracks=torch.from_numpy(pred_xy_np.copy()),
        pred_visible=torch.from_numpy(~pred_occ_np).contiguous(),
        gt_tracks=torch.from_numpy(gt_xy_np),
        gt_visible=torch.from_numpy(~gt_occ_np.astype(bool)),
    )
    summary["aj_rd"] = {k: float(v) for k, v in ajrd_result.items()}
    print(f"  AJ_RD={summary['aj_rd'].get('AJ_RD', 'N/A'):.4f}", flush=True)
except Exception as e:
    summary["aj_rd"] = {"error": repr(e), "traceback": traceback.format_exc()}
    print(f"  AJ_RD failed: {e}", flush=True)

# ── 6. Save ──
out_json = OUT / f"tapnextpp_davis_{video_name}.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nResults saved to {out_json}")
print(json.dumps(summary, indent=2))
