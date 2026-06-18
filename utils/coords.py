#!/usr/bin/env python3
"""
Canonical coordinate helper for FSPT re-detection experiments.

Single-source-of-truth for all coordinate conversions used across the
strided+original evaluation protocol.

All functions are pure: no global state, no implicit assumptions about
which protocol is active. Callers must pass explicit H/W/size parameters.

Coordinate conventions:
  - yx_norm:  [y, x] in [0, 1]², normalized by (H-1, W-1)
  - xy_norm:  [x, y] in [0, 1]², normalized by (W-1, H-1)
  - yx_pixel: [y, x] in [0, H-1] × [0, W-1]
  - xy_pixel: [x, y] in [0, W-1] × [0, H-1]
  - xy_256:   [x, y] in [0, 255]² (legacy first+input space, for compatibility)

Key: "normalized" always means denominator = (size - 1), not 255.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Tuple, Union


# ---------------------------------------------------------------------------
# Normalized <-> pixel conversions (size-minus-1 denominator)
# ---------------------------------------------------------------------------

def yx_norm_to_yx_pixel(
    yx: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Convert [..., y, x] from [0,1] normalized to [0, H-1]×[0, W-1] pixel."""
    scale_y = max(height - 1, 1)
    scale_x = max(width - 1, 1)
    out = yx.copy()
    out[..., 0] *= scale_y
    out[..., 1] *= scale_x
    return out


def yx_pixel_to_yx_norm(
    yx: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Convert [..., y, x] from pixel [0, H-1]×[0, W-1] to [0,1] normalized."""
    scale_y = max(height - 1, 1)
    scale_x = max(width - 1, 1)
    out = yx.astype(np.float32)
    out[..., 0] /= scale_y
    out[..., 1] /= scale_x
    return out


# ---------------------------------------------------------------------------
# yx <-> xy swaps
# ---------------------------------------------------------------------------

def yx_to_xy(arr: np.ndarray) -> np.ndarray:
    """Swap last-axis [..., y, x] to [..., x, y]."""
    return arr[..., [1, 0]]


def xy_to_yx(arr: np.ndarray) -> np.ndarray:
    """Swap last-axis [..., x, y] to [..., y, x]."""
    return arr[..., [1, 0]]


# ---------------------------------------------------------------------------
# Pixel conversions
# ---------------------------------------------------------------------------

def yx_norm_to_xy_pixel(
    yx: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Convert [..., y, x] normalized [0,1] to [..., x, y] pixel."""
    yx_px = yx_norm_to_yx_pixel(yx, height, width)
    return yx_to_xy(yx_px)


def xy_pixel_to_yx_norm(
    xy: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Convert [..., x, y] pixel to [..., y, x] normalized [0,1]."""
    yx_px = xy_to_yx(xy)
    return yx_pixel_to_yx_norm(yx_px, height, width)


# ---------------------------------------------------------------------------
# Legacy 256-space (first+input protocol, for compatibility only)
# ---------------------------------------------------------------------------

def yx_norm_to_yx_256(yx: np.ndarray) -> np.ndarray:
    """Convert [..., y, x] from [0,1] to legacy 256-space [0, 255]."""
    return (yx * 255.0).astype(np.float32)


def yx_256_to_yx_norm(yx_256: np.ndarray) -> np.ndarray:
    """Convert [..., y, x] from legacy 256-space [0, 255] to [0,1]."""
    return (yx_256.astype(np.float32) / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Assertions and validation
# ---------------------------------------------------------------------------

def assert_coord_range(
    arr: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    name: str = "coords",
    tol: float = 1e-4,
) -> None:
    """Assert that all values in arr are within [low - tol, high + tol]."""
    mn, mx = float(arr.min()), float(arr.max())
    if mn < low - tol or mx > high + tol:
        raise ValueError(
            f"{name}: expected range [{low}, {high}], got [{mn:.6f}, {mx:.6f}]"
        )


def assert_coord_shapes_match(
    arr1: np.ndarray, arr2: np.ndarray, name1: str = "arr1", name2: str = "arr2"
) -> None:
    """Assert that two coordinate arrays have matching shapes."""
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shape mismatch: {name1} {arr1.shape} vs {name2} {arr2.shape}"
        )


def roundtrip_coord_test(height: int = 480, width: int = 854, n: int = 100) -> dict:
    """Run roundtrip tests on random coordinates. Returns pass/fail dict."""
    rng = np.random.default_rng(42)
    results = {}

    # yx_norm -> yx_pixel -> yx_norm
    yx_norm = rng.random((n, 2)).astype(np.float32)
    yx_px = yx_norm_to_yx_pixel(yx_norm, height, width)
    yx_norm_rt = yx_pixel_to_yx_norm(yx_px, height, width)
    err = np.abs(yx_norm - yx_norm_rt).max()
    results["yx_norm_pixel_roundtrip_max_err"] = float(err)

    # yx_norm -> xy_pixel -> yx_norm
    xy_px = yx_norm_to_xy_pixel(yx_norm, height, width)
    yx_norm_rt2 = xy_pixel_to_yx_norm(xy_px, height, width)
    err2 = np.abs(yx_norm - yx_norm_rt2).max()
    results["yx_norm_xy_pixel_roundtrip_max_err"] = float(err2)

    # yx -> xy -> yx
    yx2 = yx_to_xy(xy_to_yx(yx_norm))
    err3 = np.abs(yx_norm - yx2).max()
    results["yx_xy_swap_roundtrip_max_err"] = float(err3)

    # Range assertions
    assert_coord_range(yx_norm, 0.0, 1.0, "yx_norm")
    assert_coord_range(yx_norm_rt, 0.0, 1.0, "yx_norm_roundtrip")
    results["range_assertions"] = "pass"

    return results


# ---------------------------------------------------------------------------
# Re-entry event detection
# ---------------------------------------------------------------------------

def find_reentry_events(
    gt_visibility: np.ndarray, query_t: int
) -> list:
    """Find all re-entry events for a single query track.

    A re-entry event is a frame where the point becomes visible after being
    occluded for at least one frame. Returns list of dicts:
        [{reentry_frame, occ_length, last_visible_t, occ_start_t}, ...]

    Args:
        gt_visibility: (T,) bool array (True = visible)
        query_t: query frame index

    Returns:
        list of re-entry event dicts
    """
    T = len(gt_visibility)
    events = []
    in_occ = False
    occ_start = -1
    last_visible_t = int(query_t)

    for t in range(int(query_t) + 1, T):
        visible = bool(gt_visibility[t])
        if not visible:
            if not in_occ:
                last_visible_t = t - 1
                occ_start = t
                in_occ = True
        else:
            if in_occ:
                occ_len = t - occ_start
                events.append({
                    "reentry_frame": t,
                    "occ_length": occ_len,
                    "last_visible_t": last_visible_t,
                    "occ_start_t": occ_start,
                })
                in_occ = False
    return events


def find_first_reentry(
    gt_visibility: np.ndarray, query_t: int
) -> dict | None:
    """Find the first re-entry event for a query track. Returns None if none."""
    events = find_reentry_events(gt_visibility, query_t)
    return events[0] if events else None


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------

def pixel_l2_error(
    pred: np.ndarray, gt: np.ndarray, height: int, width: int,
    pred_fmt: str = "yx_norm", gt_fmt: str = "yx_norm"
) -> np.ndarray:
    """Compute per-point L2 error in pixels.

    Args:
        pred, gt: coordinate arrays of shape (..., 2)
        height, width: original image dimensions
        pred_fmt, gt_fmt: one of 'yx_norm', 'xy_norm', 'yx_pixel', 'xy_pixel'

    Returns:
        Array of per-point errors in pixels, shape (...)
    """
    def to_xy_pixel(arr, fmt):
        if fmt == "yx_norm":
            return yx_norm_to_xy_pixel(arr, height, width)
        elif fmt == "xy_pixel":
            return arr
        elif fmt == "yx_pixel":
            return yx_to_xy(arr)
        elif fmt == "xy_norm":
            return yx_to_xy(yx_norm_to_yx_pixel(xy_to_yx(arr), height, width))
        else:
            raise ValueError(f"Unknown fmt: {fmt}")

    pred_px = to_xy_pixel(pred, pred_fmt)
    gt_px = to_xy_pixel(gt, gt_fmt)
    return np.sqrt(((pred_px - gt_px) ** 2).sum(axis=-1))


def torch_pixel_l2_error(
    pred: torch.Tensor, gt: torch.Tensor, height: int, width: int,
    pred_fmt: str = "yx_norm", gt_fmt: str = "yx_norm"
) -> torch.Tensor:
    """Torch version of pixel_l2_error."""
    scale_y = max(height - 1, 1)
    scale_x = max(width - 1, 1)

    def to_xy_pixel_t(t, fmt):
        if fmt == "yx_norm":
            x = t[..., 1] * scale_x
            y = t[..., 0] * scale_y
            return torch.stack([x, y], dim=-1)
        elif fmt == "xy_pixel":
            return t
        elif fmt == "yx_pixel":
            return torch.stack([t[..., 1], t[..., 0]], dim=-1)
        else:
            raise ValueError(f"Unknown fmt: {fmt}")

    pred_px = to_xy_pixel_t(pred, pred_fmt)
    gt_px = to_xy_pixel_t(gt, gt_fmt)
    return torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))


# ---------------------------------------------------------------------------
# Batch extraction from unified cache records
# ---------------------------------------------------------------------------

def extract_query_data(record: dict, query_idx: int) -> dict:
    """Extract all relevant data for a single query from a unified cache record.

    Returns dict with:
        query_t, query_yx, query_xy_px,
        pred_tracks (N,T,2 yx_norm), gt_tracks (N,T,2 yx_norm),
        pred_visibility (N,T bool), gt_visibility (N,T bool),
        height, width
    """
    height = int(record["original_size"][0])
    width = int(record["original_size"][1])
    return {
        "query_t": int(round(float(record["query_points"][query_idx, 0]))),
        "query_yx": record["query_points"][query_idx, 1:].copy(),
        "query_xy_px": yx_norm_to_xy_pixel(
            record["query_points"][query_idx, 1:], height, width
        ),
        "pred_tracks": np.asarray(record["pred_tracks"], dtype=np.float32),
        "gt_tracks": np.asarray(record["gt_tracks"], dtype=np.float32),
        "pred_visibility": np.asarray(record["pred_visibility"], dtype=bool),
        "gt_visibility": np.asarray(record["gt_visibility"], dtype=bool),
        "height": height,
        "width": width,
    }
