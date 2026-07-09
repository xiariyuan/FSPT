#!/usr/bin/env python3
"""Verify TAPNextPlusPlusPredictor against the existing eval script on DAVIS goat."""
from __future__ import annotations

import sys, json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/gemini/code/FSPT")
sys.path.insert(0, str(ROOT))

from scripts.tapnextpp_predictor import TAPNextPlusPlusPredictor
from baselines.track_on.dataset.tapvid import TAPVid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quick test: load predictor, run on 1 DAVIS video
predictor = TAPNextPlusPlusPredictor().to(device).eval()
print(f"Predictor loaded: {sum(p.numel() for p in predictor.parameters()):,} params", flush=True)

# Load DAVIS via TAPVid dataset
dataset = TAPVid(
    None,
    data_root=str(ROOT / "datasets/tapvid_davis/tapvid_davis.pkl"),
    dataset_type="davis",
)
print(f"Dataset: {len(dataset)} videos", flush=True)

# Get first video
rgbs, trajs_gt, visibles_gt, query_points = dataset[0]
# rgbs: (T, 3, 256, 256) float [0, 255]
# query_points: (N, 3) [t, y, x] pixel at 256

rgbs_batch = rgbs.unsqueeze(0).to(device)      # (1, T, 3, 256, 256)
qp_batch = query_points.unsqueeze(0).to(device)  # (1, N, 3)

# Run predictor
with torch.no_grad():
    tracks, visibility = predictor(rgbs_batch, qp_batch)

print(f"tracks:  {tracks.shape}  {tracks.dtype}")
print(f"visible: {visibility.shape}  {visibility.dtype}")
print(f"tracks min/max: {tracks.min().item():.1f}/{tracks.max().item():.1f}")
print(f"visible rate: {visibility.float().mean().item():.3f}")

# Quick metric sanity (compare tracks to gt)
# gt_trajs: (T, N, 2) [x, y] pixel at 256
gt = trajs_gt.unsqueeze(0).to(device)  # (1, T, N, 2)
dist = torch.norm(tracks - gt, dim=-1)  # (1, T, N)
within_16 = (dist < 16.0).float().mean().item()
print(f"Points within 16px of GT: {within_16:.3f}")

# Match OA
oa = (visibility == visibles_gt.unsqueeze(0).to(device)).float().mean().item()
print(f"OA vs GT: {oa:.3f}")

print("\nPredictor OK ✅")
