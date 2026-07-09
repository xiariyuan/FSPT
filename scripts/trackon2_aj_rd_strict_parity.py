#!/usr/bin/env python3
"""Strict apples-to-apples AJ_RD for Track-On2 using the SAME function as TAPNext++ v3.

- Loads Track-On2 cache (normalized [y,x] tracks, 256x256 input space)
- Converts to pixel [x,y] space (PIX=255, matching TAPNext++ v3)
- Calls `compute_redetection_metrics` from external/tapnextpp/repo (same as TAPNext++ v3)
- Aggregates per-video (video-weighted mean), same as TAPNext++ v3
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/gemini/code/FSPT")
REPO = ROOT / "external/tapnextpp/repo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics

CACHE = ROOT / "outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt"
OUT = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke/trackon2_aj_rd_strict_parity.json"

PIX = 255.0

payload = torch.load(str(CACHE), map_location="cpu", weights_only=False)
records = payload["records"]
print(f"Loaded {len(records)} records from {CACHE.name}")

per_video = {}
for r in records:
    vid = str(r["video_id"])
    pred_yx = np.asarray(r["pred_tracks"], dtype=np.float32)        # (N, T, 2) [y,x] norm
    gt_yx = np.asarray(r["gt_tracks"], dtype=np.float32)            # (N, T, 2) [y,x] norm
    pred_vis = np.asarray(r["pred_visibility"], dtype=bool)        # (N, T) True=visible
    gt_vis = np.asarray(r["gt_visibility"], dtype=bool)            # (N, T)
    qpts = np.asarray(r["query_points"], dtype=np.float32)         # (N, 3) [t, y, x] norm

    # Convert to pixel [x, y] space, matching TAPNext++ v3 (PIX=255)
    pred_xy_px = pred_yx[..., ::-1] * PIX       # (N, T, 2) [x, y] px
    gt_xy_px = gt_yx[..., ::-1] * PIX           # (N, T, 2) [x, y] px

    N, T = pred_xy_px.shape[:2]

    # compute_redetection_metrics expects (B, T, N, 2)
    pred_t_b = torch.from_numpy(pred_xy_px.copy()).float().unsqueeze(0).permute(0, 2, 1, 3)  # (1,T,N,2)
    pred_v_b = torch.from_numpy(pred_vis.copy()).bool().unsqueeze(0).permute(0, 2, 1)        # (1,T,N)
    gt_t_b = torch.from_numpy(gt_xy_px.copy()).float().unsqueeze(0).permute(0, 2, 1, 3)     # (1,T,N,2)
    gt_v_b = torch.from_numpy(gt_vis.copy()).bool().unsqueeze(0).permute(0, 2, 1)            # (1,T,N)

    try:
        ajrd = compute_redetection_metrics(
            pred_tracks=pred_t_b, pred_visible=pred_v_b,
            gt_tracks=gt_t_b, gt_visible=gt_v_b,
        )
        ajrd_clean = {k: float(v) if v == v else None
                      for k, v in ajrd.items() if not k.startswith("raw_stats/")}
    except Exception as e:
        ajrd_clean = {"error": repr(e)}

    per_video[vid] = {
        "frames": T, "queries": N,
        "aj_rd": ajrd_clean,
    }
    print(f"  {vid:30s}  N={N:3d}  AJ_RD={ajrd_clean.get('AJ_RD')}")

# Aggregate (video-weighted, matching TAPNext++ v3)
ajrds = [v["aj_rd"]["AJ_RD"] for v in per_video.values()
         if v["aj_rd"].get("AJ_RD") is not None]
summary = {
    "model_name": "trackon2_dinov3",
    "cache": str(CACHE),
    "metric_function": "compute_redetection_metrics (same as TAPNext++ v3)",
    "aggregation": "video-weighted mean (same as TAPNext++ v3)",
    "PIX": PIX,
    "n_videos_total": len(per_video),
    "n_videos_valid": len(ajrds),
    "AJ_RD_mean_video_weighted": round(float(np.mean(ajrds)), 4) if ajrds else None,
    "per_video": per_video,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print()
print("=" * 60)
print(f"Track-On2 AJ_RD (strict parity, video-weighted): {summary['AJ_RD_mean_video_weighted']}")
print(f"TAPNext++ v3 AJ_RD (video-weighted):              0.5961")
print(f"Saved: {OUT}")
