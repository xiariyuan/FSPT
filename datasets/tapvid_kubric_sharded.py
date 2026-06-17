"""
Streaming Kubric dataset backed by sharded pickle files.

This dataset is designed for low-memory training environments:
- Preprocess TFDS MOVi-E once into shard files.
- Stream one shard at a time during training.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from .augmentation import PointTrackingAugmentation, TemporalAugmentation
from .coord_utils import normalize_points_yx, normalize_query_points_tyx

try:
    from omegaconf import OmegaConf

    OMEGACONF_AVAILABLE = True
except Exception:
    OMEGACONF_AVAILABLE = False
    OmegaConf = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class _SkipSample(Exception):
    """Signal that a sample should be skipped without counting as a failure."""


class TAPVidKubricShardedIterableDataset(IterableDataset):
    """
    Stream Kubric samples from sharded pickle files described by a manifest.

    Expected manifest format (JSON):
    {
      "split": "train",
      "num_samples": 9749,
      "shards": [
        {"path": "train_00000.pkl", "num_samples": 64},
        ...
      ]
    }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 24,
        num_points: int = 256,
        resolution: Tuple[int, int] = (256, 256),
        augmentation: bool = True,
        annotation_file: Optional[str] = None,
        max_retries: int = 5,
        deterministic_sampling: bool = False,
        deterministic_seed: Optional[int] = None,
        sampling: Optional[object] = None,
        **kwargs,
    ):
        super().__init__()
        self.shuffle_shards = bool(kwargs.pop("shuffle_shards", str(split).lower().strip() == "train"))
        self.shuffle_samples = bool(kwargs.pop("shuffle_samples", str(split).lower().strip() == "train"))
        del kwargs

        self.root = Path(root)
        self.split = str(split)
        self.num_frames = int(num_frames) if num_frames is not None else -1
        self.num_points = int(num_points) if num_points is not None else 0
        self.resolution = resolution
        self.max_retries = max(0, int(max_retries))
        self.deterministic_sampling = bool(deterministic_sampling)
        self.deterministic_seed = int(deterministic_seed) if deterministic_seed is not None else None
        self.sampling_cfg = self._normalize_sampling_config(sampling)
        self.sampling_strategy = str(self.sampling_cfg.get("strategy", "uniform")).lower().strip()
        self.sampling_hard_fraction = float(self.sampling_cfg.get("hard_fraction", 0.5) or 0.5)
        self.sampling_hard_fraction = max(0.0, min(1.0, self.sampling_hard_fraction))
        self.sampling_min_occlusion_len = max(
            1, int(self.sampling_cfg.get("min_occlusion_len", 10) or 10)
        )
        self._iter_counter = 0

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
            random_rotation = float(cfg.get("random_rotation", 0.0) or 0.0)
            self.spatial_aug = PointTrackingAugmentation(
                random_crop=cfg.get("random_crop", True),
                random_flip=cfg.get("random_flip", True),
                color_jitter=cfg.get("color_jitter", 0.4),
                random_scale=random_scale,
                random_rotation=random_rotation,
                crop_size=crop_size,
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
                    random_reverse=temporal_cfg.get("random_reverse", cfg.get("random_reverse", True)),
                    random_speed=random_speed,
                    random_start=temporal_cfg.get("random_start", cfg.get("random_start", True)),
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
        if OMEGACONF_AVAILABLE and OmegaConf is not None:
            try:
                if OmegaConf.is_config(sampling):
                    cfg = OmegaConf.to_container(sampling, resolve=True)
                else:
                    cfg = sampling
            except Exception:
                cfg = sampling
        else:
            cfg = sampling
        if isinstance(cfg, dict):
            cfg = dict(cfg)
        else:
            try:
                cfg = dict(cfg)
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

    def _resolve_manifest_path(self, annotation_file: Optional[str]) -> Path:
        if annotation_file:
            p = Path(annotation_file)
            return p if p.is_absolute() else (self.root / p)

        split = self.split.lower()
        if split in ("val", "valid"):
            split = "validation"
        candidates = [
            f"{split}.index.json",
            f"tapvid_kubric_{split}.index.json",
            "train.index.json",
            "tapvid_kubric_train.index.json",
        ]
        for name in candidates:
            p = self.root / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Sharded Kubric manifest not found under {self.root}. "
            "Set data.train.annotation_file to a *.index.json manifest."
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
                p = (self.root / p)
            shards.append({"path": p, "num_samples": num})
        return shards

    def __len__(self):
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

        shard_entries = list(enumerate(self._shards))
        if self.shuffle_shards and not self.deterministic_sampling and len(shard_entries) > 1:
            seed = int((torch.initial_seed() + self._iter_counter * 104729) % (2**32))
            rng = np.random.RandomState(seed)
            rng.shuffle(shard_entries)
        for i, shard in shard_entries:
            if i % shard_count == shard_index:
                yield i, shard["path"]

    def _rng_for_sample(self, sample_idx: int):
        if self.deterministic_sampling:
            base = int(self.deterministic_seed) if self.deterministic_seed is not None else 0
            mixed = (base * 1000003 + int(sample_idx) * 9176) % (2**32)
            return np.random.RandomState(int(mixed))
        seed = int((torch.initial_seed() + int(sample_idx) * 1013) % (2**32))
        return np.random.RandomState(seed)

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
            if valid_idx.numel() >= self.num_points:
                if rng is not None:
                    local = rng.choice(int(valid_idx.numel()), int(self.num_points), replace=False)
                    perm = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[: self.num_points]
                chosen = valid_idx[perm]
            else:
                if rng is not None:
                    local = rng.randint(0, int(valid_idx.numel()), size=(int(self.num_points),))
                    rand = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    rand = torch.randint(0, valid_idx.numel(), (self.num_points,), device=valid_idx.device)
                chosen = valid_idx[rand]
            return query_points[chosen], target_points[chosen], occluded[chosen]
        return query_points[valid_idx], target_points[valid_idx], occluded[valid_idx]

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
        """
        if self.num_frames is None or self.num_frames <= 0:
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

        max_lag = int(self.num_frames) - int(self.sampling_frames_after)
        if max_lag <= 0:
            raise _SkipSample(
                f"Crop length ({self.num_frames}) is too short for frames_after={self.sampling_frames_after}."
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
        min_start = max(0, reappear_anchor_t + int(self.sampling_frames_after) - int(self.num_frames))
        max_start = min(anchor_t, int(total_frames) - int(self.num_frames))
        if max_start < min_start:
            raise _SkipSample("No valid temporal window for the selected hard query.")
        return int(rng.randint(min_start, max_start + 1))

    def _prepare_sample(self, sample: Dict, sample_idx: int, shard_idx: int) -> Dict:
        rng = self._rng_for_sample(sample_idx)

        video = sample.get("video", None)
        if video is None:
            raise KeyError("Missing 'video' in sharded sample.")
        video = _as_numpy(video)
        if video.ndim != 4:
            raise ValueError(f"Invalid video shape: {video.shape}")
        if video.shape[-1] != 3 and video.shape[1] == 3:
            video = np.transpose(video, (0, 2, 3, 1))
        if video.shape[-1] != 3:
            raise ValueError(f"Expected video shape (T,H,W,3), got {video.shape}")

        query_points = sample.get("query_points", None)
        if query_points is None:
            query_points = sample.get("queries", None)
        target_points = sample.get("target_points", sample.get("points", sample.get("tracks", None)))
        occluded = sample.get("occluded", sample.get("occlusions", sample.get("occ", None)))
        if target_points is None:
            raise KeyError("Missing target_points/points/tracks in sharded sample.")
        target_points = _as_numpy(target_points).astype(np.float32)
        if target_points.ndim != 3:
            raise ValueError(f"Invalid target_points shape: {target_points.shape}")
        if target_points.shape[-1] != 2:
            raise ValueError(f"target_points last dim must be 2, got {target_points.shape}")
        if target_points.shape[0] == video.shape[0] and target_points.shape[1] != video.shape[0]:
            target_points = np.transpose(target_points, (1, 0, 2))

        if occluded is None:
            occluded = np.zeros(target_points.shape[:2], dtype=bool)
        else:
            occluded = _as_numpy(occluded).astype(bool)
            if occluded.shape != target_points.shape[:2]:
                if occluded.shape == (target_points.shape[1], target_points.shape[0]):
                    occluded = np.transpose(occluded, (1, 0))
                else:
                    raise ValueError(
                        f"Invalid occluded shape {occluded.shape}; expected {target_points.shape[:2]}"
                    )

        query_generated = False
        if query_points is None:
            query_generated = True
            n = target_points.shape[0]
            query_points = np.zeros((n, 3), dtype=np.float32)
            query_points[:, 0] = 0.0
            query_points[:, 1] = target_points[:, 0, 0]
            query_points[:, 2] = target_points[:, 0, 1]
        else:
            query_points = _as_numpy(query_points).astype(np.float32)
            if query_points.ndim != 2:
                raise ValueError(f"Invalid query_points shape: {query_points.shape}")
            if query_points.shape[1] == 2:
                query_points = np.concatenate(
                    [np.zeros((query_points.shape[0], 1), dtype=np.float32), query_points], axis=1
                )
            if query_points.shape[1] != 3:
                raise ValueError(f"query_points must be (N,3), got {query_points.shape}")

        # Align sample lengths.
        n = min(target_points.shape[0], occluded.shape[0], query_points.shape[0])
        if n <= 0:
            raise ValueError("Empty point set after alignment.")
        target_points = target_points[:n]
        occluded = occluded[:n]
        query_points = query_points[:n]

        # Optional temporal window.
        t_total = int(video.shape[0])
        if self.num_frames > 0 and t_total > self.num_frames:
            if self.sampling_strategy in (
                "reappearance_focused",
                "reappearance_focus",
                "reappear",
                "long_occ_reappearance",
                "relocalization_reappearance",
            ):
                start_idx = self._select_reappearance_temporal_window(query_points, occluded, t_total, rng)
            elif query_points.shape[0] > 0:
                anchor_idx = int(rng.randint(0, query_points.shape[0]))
                anchor_t = int(np.round(query_points[anchor_idx, 0]))
                anchor_t = int(np.clip(anchor_t, 0, t_total - 1))
                min_start = max(0, anchor_t - self.num_frames + 1)
                max_start = min(anchor_t, t_total - self.num_frames)
                if max_start < min_start:
                    min_start = max_start = max(0, min(anchor_t, t_total - self.num_frames))
                start_idx = int(rng.randint(min_start, max_start + 1))
            else:
                start_idx = int(rng.randint(0, t_total - self.num_frames + 1))
            frame_indices = np.arange(start_idx, start_idx + self.num_frames)
            video = video[frame_indices]
            target_points = target_points[:, frame_indices]
            occluded = occluded[:, frame_indices]

            new_query_t = query_points[:, 0] - start_idx
            valid_mask = (new_query_t >= 0) & (new_query_t < len(frame_indices))
            if valid_mask.any():
                query_points = query_points[valid_mask]
                target_points = target_points[valid_mask]
                occluded = occluded[valid_mask]
                query_points[:, 0] = new_query_t[valid_mask]
            else:
                if target_points.shape[0] <= 0:
                    raise ValueError("No valid query points after temporal cropping.")
                query_points[:, 0] = np.clip(new_query_t, 0, len(frame_indices) - 1)
                t_idx = query_points[:, 0].astype(np.int64)
                query_points[:, 1] = target_points[np.arange(target_points.shape[0]), t_idx, 0]
                query_points[:, 2] = target_points[np.arange(target_points.shape[0]), t_idx, 1]

        if target_points.shape[0] <= 0:
            raise ValueError("No points left after temporal processing.")

        # Optional point subsampling.
        idx = self._pick_sample_indices(query_points, occluded, rng=rng)
        if idx.size > 0:
            query_points = query_points[idx]
            target_points = target_points[idx]
            occluded = occluded[idx]

        # Keep original resolution as a tensor so DataLoader collate produces (B,2).
        original_size = torch.tensor(
            [int(video.shape[1]), int(video.shape[2])], dtype=torch.int32
        )
        video_t = torch.from_numpy(np.asarray(video)).permute(0, 3, 1, 2).float()
        if video_t.numel() > 0 and float(video_t.max()) > 1.5:
            video_t = video_t / 255.0
        target_points = normalize_points_yx(target_points, int(video.shape[1]), int(video.shape[2]))
        query_points = normalize_query_points_tyx(query_points, int(video.shape[1]), int(video.shape[2]))

        query_t = torch.from_numpy(np.asarray(query_points, dtype=np.float32)).float()
        target_t = torch.from_numpy(np.asarray(target_points, dtype=np.float32)).float()
        occluded_t = torch.from_numpy(np.asarray(occluded, dtype=bool))

        query_t[:, 0] = torch.clamp(query_t[:, 0], 0, video_t.shape[0] - 1)
        query_t[:, 1:3] = torch.clamp(query_t[:, 1:3], 0.0, 1.0)

        if self.augmentation and self.spatial_aug is not None:
            outputs = self.spatial_aug(video_t, target_t, occluded_t, query_t)
            if len(outputs) == 4:
                video_t, target_t, occluded_t, query_t = outputs
            else:
                video_t, target_t, occluded_t = outputs
        if self.augmentation and self.temporal_aug is not None:
            video_t, target_t, occluded_t, query_t = self.temporal_aug(
                video_t, target_t, occluded_t, query_t
            )

        query_t[:, 0] = torch.clamp(query_t[:, 0], 0, video_t.shape[0] - 1)
        if self.augmentation or query_generated:
            query_t, target_t, occluded_t = self._ensure_query_visible(
                query_t,
                target_t,
                occluded_t,
                rng=rng if self.deterministic_sampling else None,
            )

        if self.resolution is not None:
            new_h, new_w = self.resolution
            if video_t.shape[-2:] != (new_h, new_w):
                video_t = F.interpolate(video_t, size=(new_h, new_w), mode="bilinear", align_corners=False)

        video_name = sample.get("video_name", f"shard{shard_idx:05d}_sample{sample_idx:07d}")
        return {
            "video": video_t,
            "query_points": query_t,
            "target_points": target_t,
            "occluded": occluded_t,
            "video_name": str(video_name),
            "original_size": original_size,
        }

    def __iter__(self):
        failures = 0
        global_idx = 0
        self._iter_counter += 1
        for shard_idx, shard_path in self._iter_assigned_shards():
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)
            if isinstance(shard_data, dict):
                samples = list(shard_data.values())
            elif isinstance(shard_data, list):
                samples = shard_data
            else:
                raise ValueError(f"Unsupported shard content in {shard_path}: {type(shard_data)}")

            sample_order = np.arange(len(samples), dtype=np.int64)
            if self.shuffle_samples and not self.deterministic_sampling and sample_order.size > 1:
                seed = int((torch.initial_seed() + self._iter_counter * 130363 + shard_idx * 8191) % (2**32))
                rng = np.random.RandomState(seed)
                rng.shuffle(sample_order)

            for sample_local_idx in sample_order:
                sample = samples[int(sample_local_idx)]
                try:
                    out = self._prepare_sample(sample, sample_idx=global_idx, shard_idx=shard_idx)
                    failures = 0
                    yield out
                except _SkipSample:
                    pass
                except Exception:
                    failures += 1
                    if failures > self.max_retries:
                        raise
                finally:
                    global_idx += 1
