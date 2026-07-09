"""TAP-Vid RGB-Stacking dataset adapter.

This adapter mirrors the sample schema used by TAPVidDAVISDataset so existing
teacher exporters can be extended to RGB-Stacking with minimal changes.

Important protocol note:
- `start_index` / `num_videos` are supported for held-out evaluation.
- `video_name` and `sequence_index` always preserve the original RGB-Stacking
  dataset index, even when a subset is used. This prevents held-out videos
  10-19 from being mislabeled as 0-9.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .tapvid_official_eval import sample_queries_first, sample_queries_strided
except ImportError:
    from tapvid_official_eval import sample_queries_first, sample_queries_strided


class TAPVidRGBStackingDataset(Dataset):
    def __init__(
        self,
        root: str = "/gemini/code/datasets/tapvid_rgb_stacking",
        pkl_name: str = "tapvid_rgb_stacking.pkl",
        query_mode: str = "strided",
        query_stride: int = 5,
        max_videos: int = 0,
        num_points: Optional[int] = None,
        start_index: int = 0,
        num_videos: int = 0,
    ) -> None:
        self.root = Path(root)
        self.pkl_path = self.root / pkl_name
        if not self.pkl_path.exists():
            raise FileNotFoundError(str(self.pkl_path))
        with self.pkl_path.open("rb") as f:
            full_data = pickle.load(f)
        if not isinstance(full_data, list):
            raise ValueError(f"Expected RGB-Stacking pkl list, got {type(full_data)}")

        start = max(0, int(start_index))
        if start >= len(full_data):
            raise ValueError(f"start_index={start} out of range for dataset length {len(full_data)}")
        if num_videos and num_videos > 0:
            end = min(len(full_data), start + int(num_videos))
        elif max_videos and max_videos > 0:
            # Backward-compatible behavior: max_videos means "take this many
            # videos starting at start_index".
            end = min(len(full_data), start + int(max_videos))
        else:
            end = len(full_data)
        self.data = full_data[start:end]
        self.original_indices = list(range(start, end))

        self.query_mode = str(query_mode).lower().strip()
        if self.query_mode not in {"strided", "first"}:
            raise ValueError("query_mode must be strided or first")
        self.query_stride = max(1, int(query_stride))
        self.num_points = num_points
        self.start_index = start
        self.num_videos = len(self.data)
        self.dataset_length = len(full_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        idx = int(idx)
        entry = self.data[idx]
        original_idx = int(self.original_indices[idx])
        video = np.asarray(entry["video"])
        points_xy = np.asarray(entry["points"], dtype=np.float32)  # N,T,2 [x,y], normalized
        occluded = np.asarray(entry["occluded"], dtype=bool)
        if video.ndim != 4:
            raise ValueError(f"Expected video (T,H,W,3), got {video.shape}")
        if points_xy.ndim != 3 or points_xy.shape[-1] != 2:
            raise ValueError(f"Expected points (N,T,2), got {points_xy.shape}")
        if occluded.shape != points_xy.shape[:2]:
            raise ValueError(f"occluded shape {occluded.shape} does not match points {points_xy.shape[:2]}")
        T, H, W, _ = video.shape
        if points_xy.shape[1] != T:
            t = min(T, points_xy.shape[1])
            video = video[:t]
            points_xy = points_xy[:, :t]
            occluded = occluded[:, :t]
            T = t

        if self.query_mode == "first":
            sampled = sample_queries_first(occluded, points_xy, video)
        else:
            sampled = sample_queries_strided(occluded, points_xy, video, query_stride=self.query_stride)
        query_points = sampled["query_points"][0].astype(np.float32)  # [t,y,x]
        target_xy = sampled["target_points"][0].astype(np.float32)
        target_yx = target_xy[..., [1, 0]].astype(np.float32)
        target_occ = sampled["occluded"][0].astype(bool)

        if self.num_points is not None and self.num_points > 0 and target_yx.shape[0] > self.num_points:
            sel = np.arange(target_yx.shape[0])[: int(self.num_points)]
            target_yx = target_yx[sel]
            target_occ = target_occ[sel]
            query_points = query_points[sel]

        video_t = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        if video_t.max() > 1.5:
            video_t = video_t / 255.0
        query_t = torch.from_numpy(query_points).float()
        query_t[:, 0] = torch.clamp(query_t[:, 0], 0, T - 1)
        query_t[:, 1:3] = torch.clamp(query_t[:, 1:3], 0.0, 1.0)
        return {
            "video": video_t,
            "query_points": query_t,
            "target_points": torch.from_numpy(target_yx).float(),
            "occluded": torch.from_numpy(target_occ.astype(bool)),
            "video_name": f"rgb_stacking_{original_idx:06d}",
            "sequence_index": original_idx,
            "subset_position": idx,
            "original_size": torch.tensor([int(H), int(W)], dtype=torch.int32),
        }


def create_rgb_stacking_dataloader(root: str, batch_size: int = 1, num_workers: int = 0, **kwargs):
    from torch.utils.data import DataLoader

    ds = TAPVidRGBStackingDataset(root=root, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    ds = TAPVidRGBStackingDataset(max_videos=1)
    s = ds[0]
    print("len", len(ds))
    print("video", tuple(s["video"].shape), s["video"].dtype, float(s["video"].min()), float(s["video"].max()))
    print("query_points", tuple(s["query_points"].shape), float(s["query_points"].min()), float(s["query_points"].max()))
    print("target_points", tuple(s["target_points"].shape), float(s["target_points"].min()), float(s["target_points"].max()))
    print("occluded", tuple(s["occluded"].shape), float(s["occluded"].float().mean()))
    print("video_name", s["video_name"], "sequence_index", s["sequence_index"], "original_size", s["original_size"].tolist())
    ds2 = TAPVidRGBStackingDataset(start_index=10, num_videos=1)
    s2 = ds2[0]
    print("heldout smoke", s2["video_name"], s2["sequence_index"], tuple(s2["query_points"].shape))
