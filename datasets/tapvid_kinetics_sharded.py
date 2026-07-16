"""
Streaming TAP-Vid Kinetics dataset backed by sharded pickle files.

Why this exists
---------------
The official TAP-Vid Kinetics dumps are often distributed as large sharded
pickles (e.g. 0000_of_0010.pkl ... 0009_of_0010.pkl). Loading all shards into
memory (the default `TAPVidKineticsDataset`) can easily OOM on servers.

This IterableDataset loads **one shard at a time** and yields prepared samples,
mirroring the low-memory Kubric sharded dataset design.
"""

from __future__ import annotations

import ctypes
import glob
import io
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from .augmentation import PointTrackingAugmentation, TemporalAugmentation
from .coord_utils import normalize_points_yx, normalize_query_points_tyx
from .tapvid_official_eval import sample_queries_first, sample_queries_strided

logger = logging.getLogger(__name__)

try:
    from omegaconf import OmegaConf

    OMEGACONF_AVAILABLE = True
except Exception:
    OMEGACONF_AVAILABLE = False
    OmegaConf = None  # type: ignore[assignment]

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None

_POSIX_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def _best_effort_drop_file_cache(fd: int) -> None:
    """Hint the kernel that shard pages can be reclaimed after use."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        os.posix_fadvise(int(fd), 0, 0, int(_POSIX_FADV_DONTNEED))
    except Exception:
        return


def _best_effort_malloc_trim() -> None:
    """Return freed heap pages to the OS when glibc is available."""
    if _LIBC is None:
        return
    try:
        _LIBC.malloc_trim(0)
    except Exception:
        return


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class _SkipSample(Exception):
    """Signal that a sample should be skipped without counting as a failure."""


class TAPVidKineticsShardedIterableDataset(IterableDataset):
    """
    Stream Kinetics samples from sharded pickle files described by a manifest.

    Manifest format (JSON):
    {
      "split": "train",
      "num_samples": 1144,
      "shards": [
        {"path": "0000_of_0010.pkl", "num_samples": 115},
        ...
      ]
    }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: Optional[Tuple[int, int]] = (256, 256),
        num_points: Optional[int] = 256,
        max_frames: Optional[int] = 24,
        augmentation: Optional[object] = None,
        annotation_file: Optional[str] = None,
        max_retries: int = 5,
        query_mode: str = "strided",
        query_stride: int = 5,
        points_order: str = "xy",
        deterministic_sampling: bool = False,
        deterministic_seed: Optional[int] = None,
        sampling: Optional[object] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.root = Path(root)
        self.split = str(split)
        self.resolution = resolution
        self.num_points = int(num_points) if num_points is not None else None
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.max_retries = max(0, int(max_retries))
        self.query_mode = self._normalize_query_mode(query_mode)
        self.query_stride = max(1, int(query_stride))
        self.points_order = self._normalize_points_order(points_order)

        self.deterministic_sampling = bool(deterministic_sampling)
        self.deterministic_seed = int(deterministic_seed) if deterministic_seed is not None else None
        self.sampling_cfg = self._normalize_sampling_config(sampling)
        self.sampling_strategy = str(self.sampling_cfg.get("strategy", "uniform")).lower().strip()
        self.sampling_hard_fraction = float(self.sampling_cfg.get("hard_fraction", 0.5) or 0.5)
        self.sampling_hard_fraction = max(0.0, min(1.0, self.sampling_hard_fraction))
        self.sampling_min_occlusion_len = max(
            1, int(self.sampling_cfg.get("min_occlusion_len", 10) or 10)
        )
        self.sampling_frames_after = max(1, int(self.sampling_cfg.get("frames_after", 4) or 4))

        self.aug_cfg = self._normalize_aug_config(augmentation)
        self.augmentation = bool(self.aug_cfg.get("enabled", False))
        self.spatial_aug = None
        self.temporal_aug = None
        if self.augmentation:
            cfg = self.aug_cfg
            random_scale = cfg.get("random_scale", None)
            if isinstance(random_scale, list):
                random_scale = tuple(random_scale)
            crop_size = cfg.get("crop_size", self.resolution or (256, 256))
            if isinstance(crop_size, list):
                crop_size = tuple(crop_size)
            random_rotation = float(cfg.get("random_rotation", 0.0) or 0.0)
            self.spatial_aug = PointTrackingAugmentation(
                random_crop=bool(cfg.get("random_crop", True)),
                random_flip=bool(cfg.get("random_flip", True)),
                color_jitter=float(cfg.get("color_jitter", 0.4) or 0.0),
                random_scale=random_scale,
                random_rotation=random_rotation,
                crop_size=crop_size or (256, 256),
            )

            temporal_cfg = cfg.get("temporal", {})
            if not isinstance(temporal_cfg, dict):
                temporal_cfg = {}
            temporal_enabled = bool(temporal_cfg.get("enabled", False))
            if any(k in cfg for k in ["random_reverse", "random_speed", "random_start"]):
                temporal_enabled = True
            if temporal_enabled:
                random_speed = temporal_cfg.get("random_speed", cfg.get("random_speed", (0.5, 2.0)))
                if isinstance(random_speed, list):
                    random_speed = tuple(random_speed)
                self.temporal_aug = TemporalAugmentation(
                    random_reverse=bool(temporal_cfg.get("random_reverse", cfg.get("random_reverse", True))),
                    random_speed=random_speed,
                    random_start=bool(temporal_cfg.get("random_start", cfg.get("random_start", True))),
                )

        self.manifest_path = self._resolve_manifest_path(annotation_file)
        self._manifest = self._load_manifest(self.manifest_path)
        self._shards = self._resolve_shards(self._manifest)
        if not self._shards:
            raise RuntimeError(f"No shards found in manifest: {self.manifest_path}")

        num_samples = int(self._manifest.get("num_samples", 0) or 0)
        if num_samples <= 0:
            shard_counts = [int(s.get("num_samples", 0) or 0) for s in self._shards]
            if all(v > 0 for v in shard_counts):
                num_samples = int(sum(shard_counts))
        self._num_examples = max(0, int(num_samples))

    def _normalize_aug_config(self, augmentation) -> Dict:
        if augmentation is None:
            return {"enabled": False}
        if isinstance(augmentation, bool):
            return {"enabled": augmentation}
        if OMEGACONF_AVAILABLE and OmegaConf is not None and OmegaConf.is_config(augmentation):
            cfg = OmegaConf.to_container(augmentation, resolve=True)
        elif isinstance(augmentation, dict):
            cfg = dict(augmentation)
        else:
            try:
                cfg = dict(augmentation)
            except Exception:
                cfg = {"enabled": True}
        cfg.setdefault("enabled", True)
        return cfg

    def _normalize_sampling_config(self, sampling) -> Dict:
        if sampling is None:
            return {"strategy": "uniform"}
        if isinstance(sampling, str):
            return {"strategy": sampling}
        if OMEGACONF_AVAILABLE and OmegaConf is not None and OmegaConf.is_config(sampling):
            cfg = OmegaConf.to_container(sampling, resolve=True)
        elif isinstance(sampling, dict):
            cfg = dict(sampling)
        else:
            try:
                cfg = dict(sampling)
            except Exception:
                cfg = {"strategy": "uniform"}
        cfg.setdefault("strategy", "uniform")
        return cfg

    def _compute_reappearance_event_times(
        self,
        query_points: np.ndarray,
        occluded: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return points that reappear after a long occlusion following the query frame.

        Returns:
            hard_mask: points whose query is followed by a long occlusion and a visible reappearance.
            reappear_t: the first visible frame after that long occlusion, or -1 when absent.
        """
        n = int(query_points.shape[0])
        if n <= 0 or occluded.ndim != 2 or occluded.shape[1] <= 0:
            return np.zeros((n,), dtype=bool), np.full((n,), -1, dtype=np.int64)

        q_t = np.round(query_points[:, 0]).astype(np.int64)
        q_t = np.clip(q_t, 0, occluded.shape[1] - 1)
        hard_mask = np.zeros((n,), dtype=bool)
        reappear_t = np.full((n,), -1, dtype=np.int64)
        min_len = max(int(self.sampling_min_occlusion_len), 1)

        for i in range(n):
            run = 0
            for t in range(int(q_t[i]), occluded.shape[1]):
                if occluded[i, t]:
                    run += 1
                else:
                    if run >= min_len:
                        hard_mask[i] = True
                        reappear_t[i] = int(t)
                        break
                    run = 0

        return hard_mask, reappear_t

    def _compute_reappearance_hard_mask(
        self,
        query_points: np.ndarray,
        occluded: np.ndarray,
    ) -> np.ndarray:
        hard_mask, _ = self._compute_reappearance_event_times(query_points, occluded)
        return hard_mask

    def _normalize_points_order(self, order: Optional[str]) -> str:
        text = str(order).strip().lower() if order is not None else "xy"
        if text in ("", "none", "null"):
            return "xy"
        if text not in ("xy", "yx", "auto"):
            logger.warning(f"Unknown points_order={order}, fallback to 'xy'")
            return "xy"
        return text

    def _normalize_query_mode(self, query_mode: Optional[str]) -> str:
        mode = str(query_mode).strip().lower() if query_mode is not None else "strided"
        if mode in ("", "none", "null"):
            mode = "strided"
        if mode not in ("first", "strided"):
            logger.warning(f"Unknown query_mode={query_mode}, fallback to 'strided'")
            mode = "strided"
        return mode

    def _pick_sample_indices(
        self,
        query_points: np.ndarray,
        occluded: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        n = int(query_points.shape[0])
        if self.num_points is None or self.num_points <= 0 or n <= 0:
            return np.arange(n, dtype=np.int64)

        k = int(self.num_points)
        if self.sampling_strategy in ("uniform", "random", "", None):
            if n >= k:
                return rng.choice(n, k, replace=False).astype(np.int64)
            return rng.choice(n, k, replace=True).astype(np.int64)

        if self.sampling_strategy in ("occlusion_balanced", "occ_balanced", "balanced_occlusion"):
            q_t = np.round(query_points[:, 0]).astype(np.int64)
            q_t = np.clip(q_t, 0, occluded.shape[1] - 1)
            point_idx = np.arange(n, dtype=np.int64)
            visible_at_query = ~occluded[point_idx, q_t]
            occluded_any = occluded.any(axis=1)

            hard_mask = visible_at_query & occluded_any
            easy_mask = visible_at_query & (~occluded_any)
            hard_idx = np.nonzero(hard_mask)[0]
            easy_idx = np.nonzero(easy_mask)[0]
            vis_idx = np.nonzero(visible_at_query)[0]

            hard_fraction = float(self.sampling_hard_fraction)
            k_hard = int(round(k * hard_fraction))
            k_easy = int(k - k_hard)

            chosen = []
            if hard_idx.size > 0 and k_hard > 0:
                chosen.append(rng.choice(hard_idx, k_hard, replace=(hard_idx.size < k_hard)))
            else:
                k_easy = k
            if k_easy > 0:
                pool = easy_idx if easy_idx.size > 0 else (vis_idx if vis_idx.size > 0 else np.arange(n))
                chosen.append(rng.choice(pool, k_easy, replace=(pool.size < k_easy)))

            if chosen:
                idx = np.concatenate(chosen, axis=0).astype(np.int64)
                if idx.size > 1:
                    rng.shuffle(idx)
                return idx

        if self.sampling_strategy in (
            "reappearance_focused",
            "reappearance_focus",
            "reappear",
            "long_occ_reappearance",
            "relocalization_reappearance",
        ):
            q_t = np.round(query_points[:, 0]).astype(np.int64)
            q_t = np.clip(q_t, 0, occluded.shape[1] - 1)
            point_idx = np.arange(n, dtype=np.int64)
            visible_at_query = ~occluded[point_idx, q_t]
            hard_mask = visible_at_query & self._compute_reappearance_hard_mask(query_points, occluded)
            hard_idx = np.nonzero(hard_mask)[0]
            easy_idx = np.nonzero(visible_at_query & (~hard_mask))[0]
            vis_idx = np.nonzero(visible_at_query)[0]

            hard_fraction = float(self.sampling_hard_fraction)
            k_hard = int(round(k * hard_fraction))
            k_easy = int(k - k_hard)

            chosen = []
            if hard_idx.size > 0 and k_hard > 0:
                chosen.append(rng.choice(hard_idx, k_hard, replace=(hard_idx.size < k_hard)))
            else:
                k_easy = k
            if k_easy > 0:
                pool = easy_idx if easy_idx.size > 0 else (vis_idx if vis_idx.size > 0 else np.arange(n))
                chosen.append(rng.choice(pool, k_easy, replace=(pool.size < k_easy)))

            if chosen:
                idx = np.concatenate(chosen, axis=0).astype(np.int64)
                if idx.size > 1:
                    rng.shuffle(idx)
                return idx

        if n >= k:
            return rng.choice(n, k, replace=False).astype(np.int64)
        return rng.choice(n, k, replace=True).astype(np.int64)

    def _select_reappearance_temporal_window(
        self,
        query_points: np.ndarray,
        occluded: np.ndarray,
        total_frames: int,
        rng: np.random.RandomState,
    ) -> int:
        """
        Pick a crop that keeps a hard reappearance query and enough future context.

        This prevents the 24-frame crop from randomly missing the hard visible
        frames immediately after a long occlusion.
        """
        if self.max_frames is None or self.max_frames <= 0:
            raise _SkipSample("Reappearance-focused sampling requires a positive crop length.")
        if query_points.ndim != 2 or occluded.ndim != 2 or query_points.shape[0] == 0:
            raise _SkipSample("Empty query set for reappearance-focused sampling.")

        q_t = np.round(query_points[:, 0]).astype(np.int64)
        q_t = np.clip(q_t, 0, max(int(occluded.shape[1]) - 1, 0))
        point_idx = np.arange(int(query_points.shape[0]), dtype=np.int64)
        visible_at_query = ~occluded[point_idx, q_t]
        hard_mask, reappear_t = self._compute_reappearance_event_times(query_points, occluded)
        hard_mask = visible_at_query & hard_mask
        hard_idx = np.nonzero(hard_mask)[0]
        if hard_idx.size == 0:
            raise _SkipSample("No hard reappearance query in sample; skipping easy fallback.")

        max_lag = int(self.max_frames) - int(self.sampling_frames_after)
        if max_lag <= 0:
            raise _SkipSample(
                f"Crop length ({self.max_frames}) is too short for frames_after={self.sampling_frames_after}."
            )

        eligible = hard_idx[
            (reappear_t[hard_idx] >= 0)
            & ((reappear_t[hard_idx] - q_t[hard_idx]) <= max_lag)
            & ((reappear_t[hard_idx] + int(self.sampling_frames_after)) <= int(total_frames))
        ]
        if eligible.size == 0:
            raise _SkipSample("No hard query leaves enough future context inside the source video.")

        anchor_idx = int(rng.choice(eligible))
        anchor_t = int(q_t[anchor_idx])
        reappear_anchor_t = int(reappear_t[anchor_idx])
        min_start = max(0, reappear_anchor_t + int(self.sampling_frames_after) - int(self.max_frames))
        max_start = min(anchor_t, int(total_frames) - int(self.max_frames))
        if max_start < min_start:
            raise _SkipSample("No valid temporal window for the selected hard query.")
        return int(rng.randint(min_start, max_start + 1))

    def _resolve_manifest_path(self, annotation_file: Optional[str]) -> Path:
        if annotation_file:
            p = Path(str(annotation_file))
            return p if p.is_absolute() else (self.root / p)

        split = self.split.lower()
        if split in ("val", "valid"):
            split = "validation"
        candidates = [
            f"{split}.index.json",
            f"tapvid_kinetics_{split}.index.json",
            "train.index.json",
            "tapvid_kinetics_train.index.json",
        ]
        for name in candidates:
            p = self.root / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Sharded Kinetics manifest not found under {self.root}. "
            "Create a *.index.json manifest (e.g. train.index.json) describing the shards."
        )

    def _load_manifest(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest must be a JSON object: {path}")
        return data

    def _resolve_shards(self, manifest: Dict) -> List[Dict]:
        raw = manifest.get("shards", None)
        if not isinstance(raw, list):
            raise ValueError(f"Manifest missing 'shards' list: {self.manifest_path}")

        shards: List[Dict] = []
        for item in raw:
            if isinstance(item, str):
                rel = item
                num = 0
            elif isinstance(item, dict):
                rel = item.get("path", None) or item.get("file", None)
                if rel is None:
                    continue
                num = int(item.get("num_samples", 0) or 0)
            else:
                continue
            p = Path(rel)
            if not p.is_absolute():
                p = self.root / p
            shards.append({"path": p, "num_samples": num})
        return shards

    def __len__(self) -> int:
        world = 1
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world = int(torch.distributed.get_world_size())
        except Exception:
            world = 1
        world = max(1, world)

        if self._num_examples > 0:
            return int((self._num_examples + world - 1) // world)
        return int((len(self._shards) + world - 1) // world)

    def _iter_assigned_shards(self):
        rank = 0
        world = 1
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = int(torch.distributed.get_rank())
                world = int(torch.distributed.get_world_size())
        except Exception:
            rank, world = 0, 1
        rank = max(0, rank)
        world = max(1, world)

        worker = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker is not None:
            worker_id = int(worker.id)
            num_workers = int(worker.num_workers)
        num_workers = max(1, num_workers)

        shard_count = world * num_workers
        shard_index = rank * num_workers + worker_id

        for i, shard in enumerate(self._shards):
            if i % shard_count == shard_index:
                yield i, shard["path"]

    def _rng_for_sample(self, sample_idx: int) -> np.random.RandomState:
        if self.deterministic_sampling:
            base = int(self.deterministic_seed) if self.deterministic_seed is not None else 0
            mixed = (base * 1000003 + int(sample_idx) * 9176) % (2**32)
            return np.random.RandomState(int(mixed))
        seed = int((torch.initial_seed() + int(sample_idx) * 1013) % (2**32))
        return np.random.RandomState(seed)

    def _infer_points_order(
        self,
        points: np.ndarray,
        query_points: Optional[np.ndarray],
        height: int,
        width: int,
    ) -> str:
        if points is None or points.size == 0:
            return "xy"
        pts = np.asarray(points)
        if pts.shape[-1] != 2:
            return "xy"

        if float(np.nanmax(np.abs(pts))) <= 1.5:
            invalid_if_xy = (
                (pts[..., 0] < 0) | (pts[..., 0] > 1) | (pts[..., 1] < 0) | (pts[..., 1] > 1)
            ).mean()
            invalid_if_yx = (
                (pts[..., 0] < 0) | (pts[..., 0] > 1) | (pts[..., 1] < 0) | (pts[..., 1] > 1)
            ).mean()
            if abs(float(invalid_if_xy) - float(invalid_if_yx)) > 1e-6:
                return "xy" if invalid_if_xy < invalid_if_yx else "yx"
            return "xy"

        invalid_if_xy = (
            (pts[..., 0] < 0) | (pts[..., 0] > width) | (pts[..., 1] < 0) | (pts[..., 1] > height)
        ).mean()
        invalid_if_yx = (
            (pts[..., 0] < 0) | (pts[..., 0] > height) | (pts[..., 1] < 0) | (pts[..., 1] > width)
        ).mean()
        if abs(float(invalid_if_xy) - float(invalid_if_yx)) > 1e-6:
            return "xy" if invalid_if_xy < invalid_if_yx else "yx"
        return "xy"

    def _convert_points_to_internal_yx(
        self,
        points: np.ndarray,
        query_points: Optional[np.ndarray],
        height: int,
        width: int,
    ) -> np.ndarray:
        order = self.points_order
        if order == "auto":
            order = self._infer_points_order(points, query_points, height, width)
        if order == "xy":
            return points[..., [1, 0]]
        return points

    def _decode_video_if_needed(self, video: np.ndarray) -> np.ndarray:
        if not isinstance(video, np.ndarray):
            video = np.asarray(video)
        if video.ndim != 1 or video.size == 0:
            return video

        first = video[0]
        is_encoded_bytes = isinstance(first, (bytes, np.bytes_))
        is_encoded_uint8 = isinstance(first, np.ndarray) and first.dtype == np.uint8 and first.ndim == 1
        if not (is_encoded_bytes or is_encoded_uint8):
            return video

        decoded_frames: List[np.ndarray] = []
        for frame_idx, item in enumerate(video):
            if isinstance(item, np.ndarray):
                frame_bytes = item.tobytes()
            else:
                frame_bytes = bytes(item)

            if CV2_AVAILABLE and cv2 is not None:
                frame_buf = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_buf, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise ValueError(f"Failed to decode encoded frame #{frame_idx} with OpenCV.")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                decoded_frames.append(frame_rgb)
            else:
                from PIL import Image  # type: ignore

                with Image.open(io.BytesIO(frame_bytes)) as img:
                    decoded_frames.append(np.asarray(img.convert("RGB")))

        return np.stack(decoded_frames, axis=0)

    def _generate_queries_from_tracks(
        self,
        points_yx: np.ndarray,
        occluded: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mode = self._normalize_query_mode(mode or self.query_mode)
        points_yx = np.asarray(points_yx, dtype=np.float32)
        occluded = np.asarray(occluded, dtype=bool)
        num_tracks, num_frames = points_yx.shape[:2]
        if num_tracks == 0 or num_frames == 0:
            return points_yx, occluded, np.zeros((0, 3), dtype=np.float32)

        points_xy = points_yx[..., [1, 0]].astype(np.float32)
        use_pixel_scale = float(np.nanmax(np.abs(points_xy))) > 1.5 if points_xy.size > 0 else False
        if use_pixel_scale:
            h_est = max(1.0, float(np.nanmax(points_yx[..., 0])) if points_yx.size > 0 else 1.0)
            w_est = max(1.0, float(np.nanmax(points_yx[..., 1])) if points_yx.size > 0 else 1.0)
            points_xy_norm = points_xy / np.array([max(w_est - 1.0, 1.0), max(h_est - 1.0, 1.0)], dtype=np.float32)
        else:
            h_est, w_est = 1.0, 1.0
            points_xy_norm = points_xy

        dummy_frames = np.zeros((num_frames, 1, 1, 3), dtype=np.float32)
        visible = ~occluded

        try:
            if mode == "first":
                if not np.any(visible.any(axis=1)):
                    return (
                        np.zeros((0, num_frames, 2), dtype=np.float32),
                        np.zeros((0, num_frames), dtype=bool),
                        np.zeros((0, 3), dtype=np.float32),
                    )
                converted = sample_queries_first(occluded, points_xy_norm, dummy_frames)
            else:
                has_strided = any(np.any(visible[:, t]) for t in range(0, num_frames, self.query_stride))
                if not has_strided:
                    return self._generate_queries_from_tracks(points_yx, occluded, rng=rng, mode="first")
                converted = sample_queries_strided(
                    occluded, points_xy_norm, dummy_frames, query_stride=self.query_stride
                )
        except Exception:
            if mode != "first":
                return self._generate_queries_from_tracks(points_yx, occluded, rng=rng, mode="first")
            return (
                np.zeros((0, num_frames, 2), dtype=np.float32),
                np.zeros((0, num_frames), dtype=bool),
                np.zeros((0, 3), dtype=np.float32),
            )

        target_xy_norm = converted["target_points"][0].astype(np.float32)  # (M,T,2) [x,y]
        target_occ = converted["occluded"][0].astype(bool)
        query_tyx = converted["query_points"][0].astype(np.float32)  # (M,3) [t,y,x] normalized

        target_yx = target_xy_norm[..., [1, 0]]
        query_yx = query_tyx[:, 1:3]

        if use_pixel_scale:
            target_yx = target_yx * np.array([max(h_est - 1.0, 1.0), max(w_est - 1.0, 1.0)], dtype=np.float32)
            query_yx = query_yx * np.array([max(h_est - 1.0, 1.0), max(w_est - 1.0, 1.0)], dtype=np.float32)

        queries = np.concatenate([query_tyx[:, :1], query_yx], axis=1).astype(np.float32)

        # Optional point subsampling here (keeps query-visible by construction).
        if self.num_points is not None and self.num_points > 0 and target_yx.shape[0] > self.num_points:
            if rng is None:
                rng = np.random.RandomState()
            idx = rng.choice(int(target_yx.shape[0]), int(self.num_points), replace=False)
            target_yx = target_yx[idx]
            target_occ = target_occ[idx]
            queries = queries[idx]

        return target_yx.astype(np.float32), target_occ, queries

    def _ensure_query_visible(
        self,
        query_points: torch.Tensor,
        target_points: torch.Tensor,
        occluded: torch.Tensor,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_points.numel() == 0 or target_points.numel() == 0 or occluded.numel() == 0:
            return query_points, target_points, occluded
        if occluded.dim() != 2:
            return query_points, target_points, occluded
        T = occluded.shape[1]
        query_t = query_points[:, 0].round().long().clamp(0, T - 1)
        point_idx = torch.arange(occluded.shape[0], device=occluded.device)
        visible_mask = ~occluded[point_idx, query_t]
        if visible_mask.all():
            return query_points, target_points, occluded
        valid_idx = torch.nonzero(visible_mask, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            return query_points, target_points, occluded

        if self.num_points is not None and self.num_points > 0:
            k = int(self.num_points)
            if valid_idx.numel() >= k:
                if rng is not None:
                    local = rng.choice(int(valid_idx.numel()), k, replace=False)
                    perm = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:k]
                chosen = valid_idx[perm]
            else:
                if rng is not None:
                    local = rng.randint(0, int(valid_idx.numel()), size=(k,))
                    rand = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    rand = torch.randint(0, valid_idx.numel(), (k,), device=valid_idx.device)
                chosen = valid_idx[rand]
            return query_points[chosen], target_points[chosen], occluded[chosen]
        return query_points[valid_idx], target_points[valid_idx], occluded[valid_idx]

    def _prepare_sample(
        self,
        sample: Dict,
        sample_idx: int,
        shard_idx: int,
    ) -> Dict:
        rng = self._rng_for_sample(sample_idx)

        video = sample.get("video", None)
        if video is None:
            raise KeyError("Missing 'video' in kinetics shard sample.")
        video = _as_numpy(video)

        points = sample.get("points", None)
        if points is None:
            raise KeyError("Missing 'points' in kinetics shard sample.")
        points = _as_numpy(points).astype(np.float32)

        occluded = sample.get("occluded", None)
        if occluded is None:
            visibility = sample.get("visibility", None) or sample.get("vis", None)
            if visibility is not None:
                occluded = ~_as_numpy(visibility).astype(bool)
        if occluded is None:
            raise KeyError("Missing 'occluded'/'visibility' in kinetics shard sample.")
        occluded = _as_numpy(occluded).astype(bool)

        # Ensure points are (N,T,2) and occluded is (N,T).
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"Invalid points shape: {points.shape}")
        if occluded.ndim != 2:
            raise ValueError(f"Invalid occluded shape: {occluded.shape}")

        # Align temporal dimension with the raw video length (works for encoded bytes too).
        t_video = int(video.shape[0])
        if points.shape[1] != t_video:
            min_t = min(int(points.shape[1]), t_video)
            if min_t <= 0:
                raise ValueError("Invalid temporal alignment (empty after min_t).")
            video = video[:min_t]
            points = points[:, :min_t]
            occluded = occluded[:, :min_t]
            t_video = min_t

        # Generate query points if not provided (official protocol).
        query_points = sample.get("query_points", None)
        if query_points is None:
            query_points = sample.get("queries", None)
        if query_points is not None:
            query_points = _as_numpy(query_points).astype(np.float32)
            if query_points.ndim != 2:
                raise ValueError(f"Invalid query_points shape: {query_points.shape}")
            if query_points.shape[1] == 2:
                query_points = np.concatenate(
                    [np.zeros((query_points.shape[0], 1), dtype=np.float32), query_points], axis=1
                )
            if query_points.shape[1] != 3:
                raise ValueError(f"query_points must be (N,3), got {query_points.shape}")
        else:
            query_points = None

        # We can infer size only after decoding, but points are usually normalized already.
        # Use dummy height/width when inferring; if points are pixel-scale, we will normalize later.
        points_yx = self._convert_points_to_internal_yx(points, query_points, height=1, width=1)

        if query_points is None:
            points_yx, occluded, query_points = self._generate_queries_from_tracks(points_yx, occluded, rng=rng)

        if points_yx.shape[0] == 0:
            raise ValueError("No points available after query generation.")

        # Optional temporal window: slice video/points/occluded/query_points BEFORE decoding.
        if self.max_frames is not None and self.max_frames > 0 and t_video > self.max_frames:
            if self.sampling_strategy in (
                "reappearance_focused",
                "reappearance_focus",
                "reappear",
                "long_occ_reappearance",
                "relocalization_reappearance",
            ):
                start_idx = self._select_reappearance_temporal_window(query_points, occluded, t_video, rng)
            else:
                anchor_idx = int(rng.randint(0, query_points.shape[0])) if query_points.shape[0] > 0 else 0
                anchor_t = int(np.round(query_points[anchor_idx, 0]))
                anchor_t = int(np.clip(anchor_t, 0, t_video - 1))
                min_start = max(0, anchor_t - self.max_frames + 1)
                max_start = min(anchor_t, t_video - self.max_frames)
                if max_start < min_start:
                    min_start = max_start = max(0, min(anchor_t, t_video - self.max_frames))
                start_idx = int(rng.randint(min_start, max_start + 1))
            frame_indices = np.arange(start_idx, start_idx + self.max_frames)
            video = video[frame_indices]
            points_yx = points_yx[:, frame_indices]
            occluded = occluded[:, frame_indices]

            new_query_t = query_points[:, 0] - start_idx
            valid_mask = (new_query_t >= 0) & (new_query_t < len(frame_indices))
            if valid_mask.any():
                query_points = query_points[valid_mask]
                points_yx = points_yx[valid_mask]
                occluded = occluded[valid_mask]
                query_points[:, 0] = new_query_t[valid_mask]
            else:
                query_points[:, 0] = np.clip(new_query_t, 0, len(frame_indices) - 1)
                t_idx = query_points[:, 0].astype(np.int64)
                query_points[:, 1] = points_yx[np.arange(points_yx.shape[0]), t_idx, 0]
                query_points[:, 2] = points_yx[np.arange(points_yx.shape[0]), t_idx, 1]

        if query_points.shape[0] == 0:
            raise ValueError("No query points left after temporal processing.")

        idx = self._pick_sample_indices(query_points, occluded, rng=rng)
        if idx.size > 0:
            query_points = query_points[idx]
            points_yx = points_yx[idx]
            occluded = occluded[idx]

        # Decode and infer original size.
        video = self._decode_video_if_needed(video)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Decoded video must be (T,H,W,3), got {video.shape}")
        T, H, W, _ = video.shape
        # Use a tensor so default_collate stacks it as (B,2), which is compatible
        # with _resolve_resolution_from_batch implementations across train/eval.
        original_size = torch.tensor([int(H), int(W)], dtype=torch.int32)

        # Normalize points/query to [0,1] if needed (pixel-scale detection).
        points_norm = normalize_points_yx(points_yx, H, W)
        query_norm = normalize_query_points_tyx(query_points, H, W)

        # Convert to tensors.
        video_t = torch.from_numpy(np.asarray(video)).permute(0, 3, 1, 2).float()
        if video_t.numel() > 0 and float(video_t.max()) > 1.5:
            video_t = video_t / 255.0
        target_t = torch.from_numpy(np.asarray(points_norm, dtype=np.float32)).float()
        occluded_t = torch.from_numpy(np.asarray(occluded, dtype=bool))
        query_t = torch.from_numpy(np.asarray(query_norm, dtype=np.float32)).float()

        query_t[:, 0] = torch.clamp(query_t[:, 0], 0, video_t.shape[0] - 1)
        query_t[:, 1:3] = torch.clamp(query_t[:, 1:3], 0.0, 1.0)

        if self.augmentation and self.spatial_aug is not None:
            outputs = self.spatial_aug(video_t, target_t, occluded_t, query_t)
            if len(outputs) == 4:
                video_t, target_t, occluded_t, query_t = outputs
            else:
                video_t, target_t, occluded_t = outputs
        if self.augmentation and self.temporal_aug is not None:
            video_t, target_t, occluded_t, query_t = self.temporal_aug(video_t, target_t, occluded_t, query_t)

        query_t[:, 0] = torch.clamp(query_t[:, 0], 0, video_t.shape[0] - 1)
        if self.augmentation:
            query_t, target_t, occluded_t = self._ensure_query_visible(query_t, target_t, occluded_t, rng=rng)

        if self.resolution is not None:
            new_h, new_w = self.resolution
            if video_t.shape[-2:] != (new_h, new_w):
                video_t = F.interpolate(video_t, size=(new_h, new_w), mode="bilinear", align_corners=False)

        video_name = sample.get("video_name", None) or sample.get("name", None) or sample.get("id", None)
        if video_name is None or str(video_name).strip() == "":
            video_name = f"kinetics_s{shard_idx:03d}_{sample_idx:06d}"
        else:
            video_name = str(video_name)

        return {
            "video": video_t,
            "query_points": query_t,
            "target_points": target_t,
            "occluded": occluded_t,
            "video_name": video_name,
            "original_size": original_size,
        }

    def __iter__(self):
        import gc

        # Keep a local reference so a partially consumed generator can close
        # safely during interpreter teardown, when module globals may already
        # have been cleared.
        skip_sample_error = _SkipSample
        failures = 0
        global_idx = 0
        for shard_idx, shard_path in self._iter_assigned_shards():
            if not Path(shard_path).exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            shard_fd = None
            with open(shard_path, "rb") as f:
                shard_fd = f.fileno()
                shard_data = pickle.load(f)
            if isinstance(shard_data, dict):
                sample_keys = list(shard_data.keys())
                for sample_key in sample_keys:
                    out = None
                    sample = shard_data.pop(sample_key)
                    try:
                        out = self._prepare_sample(sample, sample_idx=global_idx, shard_idx=shard_idx)
                        failures = 0
                        yield out
                    except skip_sample_error:
                        pass
                    except Exception:
                        failures += 1
                        if failures > self.max_retries:
                            raise
                    finally:
                        del sample
                        if out is not None:
                            del out
                        global_idx += 1
                del sample_keys
            elif isinstance(shard_data, list):
                for sample_pos in range(len(shard_data)):
                    out = None
                    sample = shard_data[sample_pos]
                    shard_data[sample_pos] = None
                    try:
                        out = self._prepare_sample(sample, sample_idx=global_idx, shard_idx=shard_idx)
                        failures = 0
                        yield out
                    except skip_sample_error:
                        pass
                    except Exception:
                        failures += 1
                        if failures > self.max_retries:
                            raise
                    finally:
                        del sample
                        if out is not None:
                            del out
                        global_idx += 1
            else:
                raise ValueError(f"Unsupported shard content in {shard_path}: {type(shard_data)}")

            del shard_data
            if shard_fd is not None:
                _best_effort_drop_file_cache(shard_fd)
            gc.collect()
            _best_effort_malloc_trim()
