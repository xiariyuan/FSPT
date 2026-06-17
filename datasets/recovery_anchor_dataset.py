"""Dataset loader for offline recovery anchor training."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class RecoveryAnchorDataset(Dataset):
    """Load precomputed recovery-anchor samples for training the geometric head.

    Each sample contains:
        - support_descriptor: (D,) float32
        - search_feature_map: (D, H, W) float32
        - gt_xy: (2,) float32 pixel coords
        - base_xy: (2,) float32 pixel coords
        - tracker_visibility: float32
        - occ_length: int
        - search_crop_size: int
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        heatmap_size: int = 16,
        heatmap_sigma: float = 2.0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma

        index_path = self.dataset_dir / f"index_{split}.json"
        with open(index_path) as f:
            self.samples = json.load(f)

        # Filter to samples that have saved features
        self.valid_indices = []
        for i, s in enumerate(self.samples):
            npz_path = self.dataset_dir / s["file"]
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=False)
                if "support_descriptor" in data and "search_feature_map" in data:
                    self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[self.valid_indices[idx]]
        npz_path = self.dataset_dir / sample["file"]
        data = np.load(str(npz_path), allow_pickle=False)

        support_desc = torch.from_numpy(data["support_descriptor"]).float()  # (D,)
        search_fmap = torch.from_numpy(data["search_feature_map"]).float()   # (D, H, W)

        gt_xy = np.array(sample["gt_xy"], dtype=np.float32)    # pixel coords
        base_xy = np.array(sample["base_xy"], dtype=np.float32)
        search_size = sample["search_crop_size"]

        # Convert GT to search-crop-local coords [0, 1]
        half = search_size / 2.0
        gt_offset = gt_xy - base_xy  # offset from crop center
        gt_local_x = (gt_offset[0] + half) / search_size  # [0, 1] within crop
        gt_local_y = (gt_offset[1] + half) / search_size
        gt_local_xy = np.array([gt_local_x, gt_local_y], dtype=np.float32)

        # Check if GT is inside search crop
        gt_inside = bool(
            abs(gt_offset[0]) <= half and abs(gt_offset[1]) <= half
        )

        # Generate heatmap target
        D, H, W = search_fmap.shape
        heatmap_target = self._make_heatmap(gt_local_xy, H, W)

        # Tracker visibility
        tracker_vis = torch.tensor([sample.get("tracker_visibility_t0", 0.0)], dtype=torch.float32)

        # Base error for weighting
        base_error_px = sample["base_error_px"]

        return {
            "support_descriptor": support_desc,
            "search_feature_map": search_fmap,
            "tracker_vis": tracker_vis,
            "gt_local_xy": torch.from_numpy(gt_local_xy),
            "heatmap_target": heatmap_target,
            "gt_inside": torch.tensor(gt_inside),
            "base_error_px": torch.tensor(base_error_px),
            "occ_length": torch.tensor(sample["occ_length"]),
            "base_xy": torch.from_numpy(base_xy),
            "gt_xy": torch.from_numpy(gt_xy),
            "search_crop_size": torch.tensor(search_size),
        }

    def _make_heatmap(self, gt_local_xy: np.ndarray, H: int, W: int) -> torch.Tensor:
        """Generate Gaussian heatmap target."""
        heatmap = torch.zeros(H, W)
        cx = gt_local_xy[0] * W  # pixel coords in feature map
        cy = gt_local_xy[1] * H

        for h in range(H):
            for w in range(W):
                dist_sq = (h - cy) ** 2 + (w - cx) ** 2
                heatmap[h, w] = np.exp(-dist_sq / (2 * self.heatmap_sigma ** 2))

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap


def make_sample_weight(base_error_px: float) -> float:
    """Sample weight: upweight bad samples."""
    if base_error_px > 32:
        return 8.0
    elif base_error_px > 16:
        return 4.0
    else:
        return 1.0
