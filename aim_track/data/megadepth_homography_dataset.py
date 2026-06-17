from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _split_paths(paths: Sequence[Path], split: str, val_fraction: float) -> List[Path]:
    split = str(split or "train").strip().lower()
    val_count = max(1, int(round(len(paths) * float(val_fraction)))) if paths else 0
    if split in {"val", "valid", "validation"}:
        return list(paths[-val_count:]) if val_count > 0 else []
    if split in {"train", "training"}:
        return list(paths[:-val_count]) if val_count > 0 else list(paths)
    raise ValueError(f"Unsupported split: {split}")


def _resize_shorter_side(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_size[0]), int(target_size[1])
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _sample_homography(
    width: int,
    height: int,
    jitter_ratio: float,
    rng: random.Random,
) -> np.ndarray:
    src = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )
    jitter_x = float(width - 1) * float(jitter_ratio)
    jitter_y = float(height - 1) * float(jitter_ratio)
    dst = src.copy()
    for idx in range(4):
        dst[idx, 0] += rng.uniform(-jitter_x, jitter_x)
        dst[idx, 1] += rng.uniform(-jitter_y, jitter_y)
    dst[:, 0] = np.clip(dst[:, 0], 0.0, float(width - 1))
    dst[:, 1] = np.clip(dst[:, 1], 0.0, float(height - 1))
    homography = cv2.getPerspectiveTransform(src, dst)
    return homography.astype(np.float32)


def _apply_homography(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points_xy.astype(np.float32), ones], axis=1)
    warped = homo @ homography.T
    warped = warped[:, :2] / np.clip(warped[:, 2:3], 1.0e-6, None)
    return warped


def _sample_valid_points(
    homography: np.ndarray,
    width: int,
    height: int,
    num_points: int,
    rng: random.Random,
    max_trials: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(max_trials):
        src_xy = np.stack(
            [
                np.array([rng.uniform(0.0, float(width - 1)) for _ in range(num_points)], dtype=np.float32),
                np.array([rng.uniform(0.0, float(height - 1)) for _ in range(num_points)], dtype=np.float32),
            ],
            axis=1,
        )
        tgt_xy = _apply_homography(src_xy, homography)
        valid = (
            (tgt_xy[:, 0] >= 0.0)
            & (tgt_xy[:, 0] <= float(width - 1))
            & (tgt_xy[:, 1] >= 0.0)
            & (tgt_xy[:, 1] <= float(height - 1))
        )
        if int(valid.sum()) >= num_points:
            return src_xy[:num_points], tgt_xy[:num_points]
        if int(valid.sum()) > 0:
            src_xy = src_xy[valid]
            tgt_xy = tgt_xy[valid]
            if src_xy.shape[0] >= num_points:
                return src_xy[:num_points], tgt_xy[:num_points]
    raise RuntimeError("Failed to sample enough valid points for homography pair.")


@dataclass
class HomographySample:
    source_image: torch.Tensor
    target_image: torch.Tensor
    source_points_yx: torch.Tensor
    target_points_yx: torch.Tensor
    image_path: str


class MegaDepthHomographyDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        num_points: int = 32,
        jitter_ratio: float = 0.20,
        val_fraction: float = 0.1,
        photometric_jitter: float = 0.2,
        seed: int = 42,
        repeat_factor: int = 64,
    ):
        super().__init__()
        self.image_root = Path(image_root).resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")
        all_paths = _list_images(self.image_root)
        if not all_paths:
            raise FileNotFoundError(f"No images found under {self.image_root}")
        self.paths = _split_paths(all_paths, split=split, val_fraction=val_fraction)
        if not self.paths:
            raise RuntimeError(f"No images assigned to split={split} under {self.image_root}")
        self.split = str(split)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.num_points = int(num_points)
        self.jitter_ratio = float(jitter_ratio)
        self.repeat_factor = max(1, int(repeat_factor))
        self.base_seed = int(seed)
        self.color_jitter = ColorJitter(
            brightness=float(photometric_jitter),
            contrast=float(photometric_jitter),
            saturation=float(photometric_jitter),
            hue=min(float(photometric_jitter) * 0.5, 0.5),
        )

    def __len__(self) -> int:
        return len(self.paths) * self.repeat_factor if self.split.lower().startswith("train") else len(self.paths)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.base_seed + int(index))

    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        array = _resize_shorter_side(array, self.image_size)
        return array

    def __getitem__(self, index: int) -> HomographySample:
        path = self.paths[index % len(self.paths)]
        rng = self._rng(index)
        source = self._load_image(path)
        height, width = int(source.shape[0]), int(source.shape[1])
        homography = _sample_homography(width=width, height=height, jitter_ratio=self.jitter_ratio, rng=rng)
        target = cv2.warpPerspective(source, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        src_xy, tgt_xy = _sample_valid_points(homography, width=width, height=height, num_points=self.num_points, rng=rng)

        source_image = Image.fromarray(source)
        target_image = Image.fromarray(target)
        if self.split.lower().startswith("train"):
            source_image = self.color_jitter(source_image)
            target_image = self.color_jitter(target_image)
        source_tensor = torch.from_numpy(np.array(source_image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(np.array(target_image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float() / 255.0

        src_yx = np.stack(
            [
                src_xy[:, 1] / max(height - 1, 1),
                src_xy[:, 0] / max(width - 1, 1),
            ],
            axis=1,
        )
        tgt_yx = np.stack(
            [
                tgt_xy[:, 1] / max(height - 1, 1),
                tgt_xy[:, 0] / max(width - 1, 1),
            ],
            axis=1,
        )

        return HomographySample(
            source_image=source_tensor,
            target_image=target_tensor,
            source_points_yx=torch.from_numpy(src_yx.astype(np.float32)),
            target_points_yx=torch.from_numpy(tgt_yx.astype(np.float32)),
            image_path=str(path),
        )

