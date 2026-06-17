"""Shared image crop utilities for the recovery pipeline.

All scripts that crop patches must use these helpers to ensure pixel-identical behavior.
"""

from __future__ import annotations
import cv2
import numpy as np


def extract_crop(image: np.ndarray, center_xy: np.ndarray, crop_size: int) -> np.ndarray:
    """Extract a square crop centered on center_xy using cv2.getRectSubPix.

    This is the canonical crop implementation used throughout the codebase.
    cv2.getRectSubPix handles sub-pixel centers and border cases consistently.

    Args:
        image: (H, W, 3) uint8 RGB image
        center_xy: (2,) float [x, y] center in pixel coords
        crop_size: output crop side length

    Returns:
        (crop_size, crop_size, 3) uint8 RGB crop
    """
    center = (float(center_xy[0]), float(center_xy[1]))
    return cv2.getRectSubPix(image, (crop_size, crop_size), center)
