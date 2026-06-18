#!/usr/bin/env python3
"""Compute forward-backward consistency for pseudo-label verification.

For a point (x, y) at frame t:
  1. Forward: track (x,y) → (x',y') at frame t+1 via DINO template matching
  2. Backward: track (x',y') at t+1 → (x'',y'') at frame t
  3. FB error = L2 distance between (x,y) and (x'',y'')

Low FB error = temporally consistent tracking point.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from utils.crop_utils import extract_crop


def compute_fb_error(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
    x: float,
    y: float,
    dino_model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 64,
    search_radius: int = 16,
) -> float:
    """Compute forward-backward consistency error for a single point.

    Args:
        frame_t: (H, W, 3) uint8 frame at time t
        frame_t1: (H, W, 3) uint8 frame at time t+1
        x, y: point coordinates in pixels
        dino_model: DINOv2 model (frozen)
        device: torch device
        patch_size: template patch size
        search_radius: local search radius in pixels

    Returns:
        FB error in pixels, or -1.0 if tracking fails
    """
    H, W = frame_t.shape[:2]

    # Extract template patch at (x, y)
    template_crop = extract_crop(frame_t, np.array([x, y], dtype=np.float32), patch_size)
    t_t = torch.from_numpy(template_crop).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # Extract search crop from frame t+1 centered at (x, y) with search_radius
    search_crop = extract_crop(frame_t1, np.array([x, y], dtype=np.float32),
                                patch_size + search_radius * 2)
    t_t1 = torch.from_numpy(search_crop).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    sh, sw = search_crop.shape[:2]

    # Resize to DINO input size
    t_t = F.interpolate(t_t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
    t_t1 = F.interpolate(t_t1, size=(518, 518), mode="bilinear", align_corners=False).to(device)

    with torch.no_grad():
        feat_t = dino_model(t_t)[-1]  # (1, D, h_t, w_t)
        feat_t1 = dino_model(t_t1)[-1]  # (1, D, h_t1, w_t1)

    # Forward: template feat_t → search feat_t1
    h_t, w_t = feat_t.shape[-2:]
    h_t1, w_t1 = feat_t1.shape[-2:]

    # Resize template to match search feature grid
    feat_t_flat = F.normalize(feat_t.float(), dim=1)
    feat_t1_flat = F.normalize(feat_t1.float(), dim=1)

    # Compute similarity map: template patch as single descriptor
    # Use global pooling of template features as query
    template_desc = feat_t_flat.mean(dim=[-2, -1], keepdim=True)  # (1, D, 1, 1)
    sim_map = (feat_t1_flat * template_desc).sum(dim=1, keepdim=True)  # (1, 1, h_t1, w_t1)

    # Best match in forward direction
    sim_map_flat = sim_map.reshape(1, -1)
    best_idx_fwd = int(sim_map_flat.argmax().item())
    best_y_fwd = best_idx_fwd // w_t1
    best_x_fwd = best_idx_fwd % w_t1

    # Convert to pixel coords in search crop
    x1_px = (best_x_fwd + 0.5) * sw / float(w_t1)
    y1_px = (best_y_fwd + 0.5) * sh / float(h_t1)

    # Convert to global frame coords
    crop_center_x, crop_center_y = x, y
    x1_global = crop_center_x + (x1_px - sw / 2.0)
    y1_global = crop_center_y + (y1_px - sh / 2.0)

    # Clamp to image bounds
    x1_global = max(0, min(x1_global, W - 1))
    y1_global = max(0, min(y1_global, H - 1))

    # Backward: track from (x1_global, y1_global) at t+1 back to t
    backward_crop = extract_crop(frame_t, np.array([x1_global, y1_global], dtype=np.float32),
                                  patch_size + search_radius * 2)
    t_back = torch.from_numpy(backward_crop).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t_back = F.interpolate(t_back, size=(518, 518), mode="bilinear", align_corners=False).to(device)

    # Encode backward target (frame t+1 patch at x1,y1)
    fwd_patch = extract_crop(frame_t1, np.array([x1_global, y1_global], dtype=np.float32), patch_size)
    t_fwd = torch.from_numpy(fwd_patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t_fwd = F.interpolate(t_fwd, size=(518, 518), mode="bilinear", align_corners=False).to(device)

    with torch.no_grad():
        feat_fwd = dino_model(t_fwd)[-1]
        feat_back = dino_model(t_back)[-1]

    fwd_desc = F.normalize(feat_fwd.float(), dim=1).mean(dim=[-2, -1], keepdim=True)
    back_h, back_w = feat_back.shape[-2:]
    back_sim = (F.normalize(feat_back.float(), dim=1) * fwd_desc).sum(dim=1, keepdim=True)

    best_idx_bwd = int(back_sim.reshape(1, -1).argmax().item())
    best_y_bwd = best_idx_bwd // back_w
    best_x_bwd = best_idx_bwd % back_w

    back_crop_h, back_crop_w = backward_crop.shape[:2]
    x2_px = (best_x_bwd + 0.5) * back_crop_w / float(back_w)
    y2_px = (best_y_bwd + 0.5) * back_crop_h / float(back_h)

    x2_global = x1_global + (x2_px - back_crop_w / 2.0)
    y2_global = y1_global + (y2_px - back_crop_h / 2.0)

    # FB error = distance between original and back-tracked point
    fb_error = float(np.sqrt((x2_global - x)**2 + (y2_global - y)**2))

    # Sanity check: if tracking went out of bounds, return -1
    if fb_error > 500:
        return -1.0

    return fb_error


def compute_fb_batch(
    labels: list,
    video_frames: dict,
    dino_model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 64,
    search_radius: int = 16,
) -> list:
    """Compute FB errors for a batch of pseudo-labels.

    Each label must have: video_id, frame, xy ([x, y] pixel).

    Returns: list of labels with fb_error filled in.
    """
    for label in labels:
        vid = label["video_id"]
        t = label["frame"]
        x, y = label["xy"]
        frames = video_frames.get(vid)

        if frames is None or t + 1 >= len(frames):
            label["fb_error"] = -1.0
            continue

        frame_t = frames[t]
        frame_t1 = frames[t + 1]

        fb_err = compute_fb_error(
            frame_t, frame_t1, float(x), float(y),
            dino_model, device, patch_size, search_radius,
        )
        label["fb_error"] = round(fb_err, 2)

    return labels
