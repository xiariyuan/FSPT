#!/usr/bin/env python3
"""Full 30-video DAVIS evaluation for TAPNext++."""
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


def compute_tapvid_metrics(
        query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks,
        query_mode, thresholds=(1, 2, 4, 8, 16),
):
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
        np.equal(pred_occluded, gt_occluded) & evaluation_points, axis=(1, 2)
    ) / np.maximum(np.sum(evaluation_points, axis=(1, 2)), 1e-8)
    metrics = {"occlusion_accuracy": float(occ_acc[0])}
    visible = ~gt_occluded
    pred_visible = ~pred_occluded
    all_j = []
    for thresh in thresholds:
        within = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < thresh ** 2
        is_correct = within & visible
        tp = np.sum(is_correct & pred_visible, axis=(1, 2))
        fp = np.sum((~is_correct) & pred_visible, axis=(1, 2)) + np.sum(
            (~visible) & pred_visible, axis=(1, 2))
        fn = np.sum(is_correct & (~pred_visible), axis=(1, 2))
        j = tp / np.maximum(tp + fp + fn, 1e-8)
        metrics[f"jaccard_{thresh}"] = float(j[0])
        all_j.append(j)
    metrics["average_jaccard"] = float(np.mean(np.stack(all_j), axis=0)[0])
    return metrics


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

for vi, video_name in enumerate(video_names):
    print(f"\n[{vi+1}/{len(video_names)}] {video_name}...", flush=True)
    entry = davis_data[video_name]
    video_np = entry["video"].astype(np.float32)
    target_size = (256, 256)

    # Resize
    if video_np.shape[1:3] != target_size:
        vt = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
        vt = F.interpolate(vt, size=target_size, mode="bilinear",
                           align_corners=False)
        video_np = vt.permute(0, 2, 3, 1).numpy()

    video_np = video_np / 255.0 * 2.0 - 1.0
    T_ = video_np.shape[0]
    points_np = entry["points"]
    occluded_np = entry["occluded"]

    # GT pixel coords
    gt_xy = points_np * np.array([target_size[1], target_size[0]])

    # Queries: first-frame
    N = points_np.shape[0]
    query_mask = ~occluded_np[:, 0]
    if not query_mask.any():
        query_mask = np.ones(N, dtype=bool)
    H256, W256 = target_size
    query_pts = np.stack([
        np.zeros(N, dtype=np.float32),
        points_np[:, 0, 1] * (H256 - 1),
        points_np[:, 0, 0] * (W256 - 1),
    ], axis=-1)[query_mask]
    gt_xy = gt_xy[query_mask]
    gt_occ = occluded_np[query_mask]
    Q = query_pts.shape[0]

    if Q == 0:
        print(f"  No valid queries, skip", flush=True)
        continue

    video = torch.from_numpy(video_np[None, ...]).to(device)
    query_t = torch.from_numpy(query_pts[None, ...]).to(device)

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

    # Assemble → (Q, T, *) then flip (y,x)→(x,y) for TAP-Vid metric
    tr_all = torch.cat(tr_list, dim=1).squeeze(0).permute(1, 0, 2)  # (Q,T,2)
    vl_all = torch.cat(vl_list, dim=1).squeeze(0).squeeze(-1).permute(1, 0)  # (T,Q)→(Q,T)
    occ_pred = ~vl_all.bool()

    pred_xy = np.asarray(tr_all.numpy()[..., ::-1])  # (y,x)→(x,y)
    pred_occ = np.asarray(occ_pred.numpy())

    total_frames += T_
    total_infer_sec += infer_sec

    # TAP-Vid metrics
    tapvid = compute_tapvid_metrics(
        query_points=query_pts[None, ...],
        gt_occluded=gt_occ[None, ...].astype(bool),
        gt_tracks=gt_xy[None, ...],
        pred_occluded=pred_occ[None, ...],
        pred_tracks=pred_xy[None, ...],
        query_mode="first",
    )

    # AJ_RD — expects (B,T,N,2) and (B,T,N)
    try:
        # Convert from (Q,T,2) → (1,T,Q,2) and (Q,T) → (1,T,Q)
        pred_tracks_b = torch.from_numpy(pred_xy.copy()).float().unsqueeze(0)  # (1,Q,T,2)
        pred_tracks_b = pred_tracks_b.permute(0, 2, 1, 3)  # (1,T,Q,2)
        pred_visible_b = torch.from_numpy(~pred_occ).bool().unsqueeze(0)  # (1,Q,T)
        pred_visible_b = pred_visible_b.permute(0, 2, 1)  # (1,T,Q)
        gt_tracks_b = torch.from_numpy(gt_xy.astype(np.float32).copy()).unsqueeze(0)  # (1,Q,T,2)
        gt_tracks_b = gt_tracks_b.permute(0, 2, 1, 3)  # (1,T,Q,2)
        gt_visible_b = torch.from_numpy(~gt_occ.astype(bool)).bool().unsqueeze(0)  # (1,Q,T)
        gt_visible_b = gt_visible_b.permute(0, 2, 1)  # (1,T,Q)

        ajrd = compute_redetection_metrics(
            pred_tracks=pred_tracks_b,
            pred_visible=pred_visible_b,
            gt_tracks=gt_tracks_b,
            gt_visible=gt_visible_b,
        )
        ajrd = {k: v.item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in ajrd.items()
                if not k.startswith("raw_stats/")}
    except Exception as e:
        import traceback
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

out_json = OUT / "tapnextpp_davis_full_eval.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nSaved: {out_json}")
