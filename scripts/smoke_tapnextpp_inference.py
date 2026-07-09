#!/usr/bin/env python3
"""TAPNext++ 1-video inference smoke. No ReEntry, no V26, no 30 videos."""
from __future__ import annotations

import json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/gemini/code/FSPT")
REPO = ROOT / "external/tapnextpp/repo"
CKPT = ROOT / "checkpoints/tapnextpp/tapnextpp_ckpt.pt"
OUT = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))

from tapnet.tapnext.tapnext_torch import TAPNext

summary = {}

# ── 1. Load model ──
print("Loading model...", flush=True)
t0 = time.time()
model = TAPNext(image_size=(256, 256))
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict({
    k.replace("tapnext.", ""): v
    for k, v in ckpt["state_dict"].items()
})
model.cuda().eval()
summary["model_load_sec"] = round(time.time() - t0, 2)
summary["model_parameters"] = sum(p.numel() for p in model.parameters())
print(f"  Model loaded: {summary['model_parameters']} params", flush=True)

# ── 2. Construct a minimal DAVIS sample from scratch ──
# We use a tiny RGB dummy video to verify the forward pass works end-to-end.
# This is NOT a real evaluation; just a smoke test.
print("Building dummy video...", flush=True)
T, H, W = 16, 256, 256
video_np = np.random.randn(1, T, H, W, 3).astype(np.float32) * 0.1  # ~N(0,0.1), channel-last
video = torch.from_numpy(video_np).cuda()

# query at frame 0, center-ish position (normalized [0,1])
query_points = torch.tensor([[[0.0, 0.5, 0.5]]], dtype=torch.float32).cuda()  # B,N,3 = [t, y, x]

# ── 3. Run online inference ──
print("Running online inference (16 frames)...", flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()

with torch.no_grad():
    # First frame
    tracks, track_logits, visible_logits, state = model(
        video=video[:, :1], query_points=query_points
    )
    pred_tracks = [tracks.cpu()]
    pred_visible = [(visible_logits > 0).cpu()]
    pred_logits = [visible_logits.cpu()]

    for f in range(1, T):
        curr_tracks, curr_logits, curr_visible, state = model(
            video=video[:, f:f+1], state=state,
        )
        pred_tracks.append(curr_tracks.cpu())
        pred_visible.append((curr_visible > 0).cpu())
        pred_logits.append(curr_visible.cpu())

torch.cuda.synchronize()
infer_sec = time.time() - t0
peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

# Assemble outputs
all_tracks = torch.cat(pred_tracks, dim=1)          # B,T,N,2 (xy pixel at 256)
all_visible = torch.cat(pred_visible, dim=1)         # B,T,N,1
all_logits = torch.cat(pred_logits, dim=1)           # B,T,N,1
all_visible_bool = all_visible.squeeze(-1)            # B,T,N
                   
summary["inference"] = {
    "time_sec": round(infer_sec, 3),
    "fps": round(T / infer_sec, 1),
    "frames": T,
    "queries": 1,
    "tracks_shape": list(all_tracks.shape),
    "visible_shape": list(all_visible_bool.shape),
    "visible_rate": float(all_visible_bool.float().mean().item()),
    "tracks_min": float(all_tracks.min().item()),
    "tracks_max": float(all_tracks.max().item()),
    "peak_mem_mb": round(peak_mem_mb, 1),
    "tracks_sample": all_tracks[0, :, 0, :].tolist(),  # query 0, all frames
    "visible_sample": all_visible_bool[0, :, 0].tolist(),
}

# ── 4. Compute AJ_RD metrics ──
try:
    from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics

    # Dummy data: B=1, T=16, N=1
    B, T_, N = 1, 16, 1
    dummy_pred_tracks = torch.randn(B, T_, N, 2) * 10.0 + 128.0
    dummy_pred_visible = torch.ones(B, T_, N, dtype=torch.bool)
    dummy_gt_tracks = torch.randn(B, T_, N, 2) * 10.0 + 128.0
    dummy_gt_visible = torch.ones(B, T_, N, dtype=torch.bool)

    ajrd_result = compute_redetection_metrics(
        pred_tracks=dummy_pred_tracks,
        pred_visible=dummy_pred_visible,
        gt_tracks=dummy_gt_tracks,
        gt_visible=dummy_gt_visible,
    )
    summary["ajrd_smoke"] = {
        "ok": True,
        "result_keys": list(ajrd_result.keys()),
        "sample_vals": {k: round(v, 4) for k, v in ajrd_result.items() if "dmin1" in k},
    }
    print(f"  AJ_RD compute ok: {summary['ajrd_smoke']}", flush=True)
except Exception as e:
    summary["ajrd_smoke"] = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}

print(f"\nInference done: {infer_sec:.2f}s, {T/infer_sec:.1f} fps, "
      f"peak mem {peak_mem_mb:.0f} MB", flush=True)

# ── 5. Save ──
out_json = OUT / "tapnextpp_inference_smoke.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2))
print(f"\nWROTE {out_json}")
