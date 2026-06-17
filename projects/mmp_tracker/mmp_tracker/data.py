from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


def _adapt_sample_to_first_frame_query(
    sample: Dict,
    max_points: Optional[int] = None,
    require_visible_at_query: bool = True,
    random_sample: bool = False,
) -> Optional[Dict]:
    if not isinstance(sample, dict):
        return None
    required = ("video", "target_points", "occluded")
    if any(key not in sample for key in required):
        return None

    video = sample["video"]
    target_points = sample["target_points"]
    occluded = sample["occluded"]
    if not isinstance(target_points, torch.Tensor) or not isinstance(occluded, torch.Tensor):
        return None
    if target_points.ndim != 3 or target_points.shape[-1] != 2:
        return None
    if occluded.ndim != 2:
        return None
    if target_points.shape[:2] != occluded.shape:
        return None
    if target_points.shape[0] <= 0 or target_points.shape[1] <= 0:
        return None

    visible0 = ~occluded[:, 0].bool()
    valid = visible0 if require_visible_at_query else torch.ones_like(visible0, dtype=torch.bool)
    if valid.sum().item() <= 0:
        return None

    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if max_points is not None and idx.numel() > int(max_points):
        if random_sample:
            perm = torch.randperm(idx.numel(), device=idx.device)
            idx = idx[perm[: int(max_points)]]
        else:
            idx = idx[: int(max_points)]
    if idx.numel() <= 0:
        return None

    target_points = target_points[idx].clone()
    occluded = occluded[idx].clone()
    query_points = torch.zeros((idx.numel(), 3), dtype=target_points.dtype)
    query_points[:, 1:3] = target_points[:, 0, :]

    adapted = dict(sample)
    adapted["video"] = video
    adapted["target_points"] = target_points
    adapted["occluded"] = occluded
    adapted["query_points"] = query_points
    adapted["visibility"] = (~occluded).to(dtype=target_points.dtype)
    adapted["query_mode"] = "first"
    return adapted


class FirstFrameQueryDataset(Dataset):
    def __init__(self, base_dataset: Dataset, max_points: Optional[int] = None, random_sample: bool = False):
        self.base_dataset = base_dataset
        self.max_points = max_points
        self.random_sample = random_sample

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        total = len(self.base_dataset)
        for offset in range(total):
            sample = self.base_dataset[(idx + offset) % total]
            adapted = _adapt_sample_to_first_frame_query(
                sample,
                max_points=self.max_points,
                random_sample=self.random_sample,
            )
            if adapted is not None:
                return adapted
        raise RuntimeError("Failed to fetch a valid first-frame-query sample from base dataset.")


class FirstFrameQueryIterableDataset(IterableDataset):
    def __init__(self, base_dataset: IterableDataset, max_points: Optional[int] = None, random_sample: bool = False):
        super().__init__()
        self.base_dataset = base_dataset
        self.max_points = max_points
        self.random_sample = random_sample

    def __len__(self) -> int:
        if hasattr(self.base_dataset, "__len__"):
            return len(self.base_dataset)  # type: ignore[arg-type]
        raise TypeError("Base iterable dataset does not provide __len__.")

    def __iter__(self) -> Iterator[Dict]:
        for sample in self.base_dataset:
            adapted = _adapt_sample_to_first_frame_query(
                sample,
                max_points=self.max_points,
                random_sample=self.random_sample,
            )
            if adapted is not None:
                yield adapted


def wrap_first_frame_query_dataset(dataset, max_points: Optional[int] = None, random_sample: bool = False):
    if isinstance(dataset, IterableDataset):
        return FirstFrameQueryIterableDataset(dataset, max_points=max_points, random_sample=random_sample)
    return FirstFrameQueryDataset(dataset, max_points=max_points, random_sample=random_sample)
