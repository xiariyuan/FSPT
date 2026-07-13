"""
FSPT Datasets Module

支持的数据集:
- TAP-Vid-Kubric: 合成数据，用于训练
- TAP-Vid-DAVIS: 真实视频，用于测试
- TAP-Vid-Kinetics: 真实视频，用于测试
- TAP-Vid-RGB-Stacking: 真实视频，用于验证/测试
"""

from __future__ import annotations

import glob
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, Subset

from .augmentation import PointTrackingAugmentation
from .metrics import compute_dataset_metrics, compute_tapvid_metrics
from .tapvid_davis import TAPVidDAVISDataset

logger = logging.getLogger(__name__)

# Optional datasets (may fail to import in minimal environments).
try:
    from .tapvid_kubric import (
        TAPVidKubricDataset,
        TAPVidKubricTFDSIterableDataset,
        create_kubric_dataloader,
    )
except ImportError:
    TAPVidKubricDataset = None
    TAPVidKubricTFDSIterableDataset = None
    create_kubric_dataloader = None

try:
    from .tapvid_kubric_pickle import TAPVidKubricPickleDataset
except ImportError:
    TAPVidKubricPickleDataset = None

try:
    from .tapvid_kubric_sharded import TAPVidKubricShardedIterableDataset
except ImportError:
    TAPVidKubricShardedIterableDataset = None

try:
    from .tapvid_kinetics import TAPVidKineticsDataset
except ImportError:
    TAPVidKineticsDataset = None

try:
    from .tapvid_kinetics_sharded import TAPVidKineticsShardedIterableDataset
except ImportError:
    TAPVidKineticsShardedIterableDataset = None

try:
    from .tapvid_rgb_stacking import TAPVidRGBStackingDataset
except ImportError:
    TAPVidRGBStackingDataset = None

try:
    from .megadepth_pairs import MegaDepthPairDataset
except ImportError:
    MegaDepthPairDataset = None


def normalize_dataset_name(name: str) -> str:
    """规范化数据集名称（支持别名）"""
    if not name:
        raise ValueError("Dataset name cannot be empty")
    normalized = name.lower().replace("-", "_").strip()
    if normalized.startswith("tapvid_"):
        normalized = normalized[len("tapvid_") :]
    return normalized


def _seed_worker(worker_id: int) -> None:
    """确保多进程数据加载的随机性可复现"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(name: str, root: str, **kwargs):
    """
    获取数据集

    Args:
        name: 数据集名称（支持: kubric/davis/kinetics 或 tapvid_kubric/tapvid_davis/tapvid_kinetics）
        root: 数据集根目录
        split: 数据集划分（部分数据集支持）
        **kwargs: 其他参数
    """
    name = normalize_dataset_name(name)

    # Extract shared args.
    split = kwargs.pop("split", None)
    backend_raw = kwargs.pop("backend", "auto")
    backend = str(backend_raw).lower().strip() if backend_raw is not None else "auto"
    backend_alias = {
        "auto": "auto",
        "tfds": "tfds",
        "pickle": "pickle",
        "pkl": "pickle",
        "sharded": "sharded_pkl",
        "sharded_pkl": "sharded_pkl",
        "sharded-pkl": "sharded_pkl",
        "index_json": "sharded_pkl",
    }
    backend = backend_alias.get(backend, backend)

    if name == "kubric":
        if backend not in {"auto", "tfds", "pickle", "sharded_pkl"}:
            raise ValueError(
                f"Unsupported kubric backend={backend_raw}. Expected one of: auto|tfds|pickle|sharded_pkl"
            )

        # Kubric supports split.
        if split is not None:
            kwargs["split"] = split

        use_tfds = bool(kwargs.get("use_tfds", False))
        annotation_file = kwargs.get("annotation_file", None)
        if backend == "tfds":
            use_tfds = True
        elif backend in {"pickle", "sharded_pkl"}:
            use_tfds = False

        if backend == "sharded_pkl":
            if annotation_file is None:
                split_name = str(split).lower().strip() if split is not None else "train"
                if split_name in ("val", "validation"):
                    split_name = "validation"
                kwargs["annotation_file"] = f"{split_name}.index.json"
            if TAPVidKubricShardedIterableDataset is None:
                raise ImportError("TAPVidKubricShardedIterableDataset not available")
            return TAPVidKubricShardedIterableDataset(root, **kwargs)

        if backend == "pickle":
            if annotation_file is None:
                split_name = str(split).lower().strip() if split is not None else "train"
                if split_name == "val":
                    split_name = "validation"
                kwargs["annotation_file"] = f"{split_name}.pkl"
            if TAPVidKubricPickleDataset is None:
                raise ImportError("TAPVidKubricPickleDataset not available")
            return TAPVidKubricPickleDataset(root, **kwargs)

        if not use_tfds and annotation_file:
            annotation_name = str(annotation_file).lower()
            if annotation_name.endswith(".index.json"):
                if TAPVidKubricShardedIterableDataset is None:
                    raise ImportError("TAPVidKubricShardedIterableDataset not available")
                return TAPVidKubricShardedIterableDataset(root, **kwargs)
            if TAPVidKubricPickleDataset is None:
                raise ImportError("TAPVidKubricPickleDataset not available")
            return TAPVidKubricPickleDataset(root, **kwargs)

        if use_tfds:
            if TAPVidKubricDataset is None:
                raise ImportError("TAPVidKubricDataset not available")
            if TAPVidKubricTFDSIterableDataset is not None:
                return TAPVidKubricTFDSIterableDataset(root, **kwargs)
            return TAPVidKubricDataset(root, **kwargs)

        if TAPVidKubricDataset is None:
            raise ImportError("TAPVidKubricDataset not available")
        return TAPVidKubricDataset(root, **kwargs)

    if name == "davis":
        return TAPVidDAVISDataset(root, **kwargs)

    if name in {"rgb_stacking", "rgbstacking", "stacking"}:
        if TAPVidRGBStackingDataset is None:
            raise ImportError("TAPVidRGBStackingDataset not available")
        # RGB-Stacking is a single map-style pickle dataset. Preserve an
        # explicit annotation filename for exact cache provenance.
        annotation_file = kwargs.pop("annotation_file", None)
        if annotation_file is not None:
            kwargs.setdefault("pkl_name", str(annotation_file))
        return TAPVidRGBStackingDataset(root, **kwargs)

    if name == "kinetics":
        # Avoid OOM: if shards exist, default to streaming dataset in auto mode.
        annotation_file = kwargs.get("annotation_file", None)
        annotation_name = str(annotation_file).lower() if annotation_file is not None else ""
        custom_annotation_path = None
        if annotation_file is not None:
            custom_annotation_path = (
                str(annotation_file)
                if os.path.isabs(str(annotation_file))
                else os.path.join(str(root), str(annotation_file))
            )

        has_single = os.path.exists(os.path.join(str(root), "tapvid_kinetics.pkl"))
        has_csv = os.path.exists(os.path.join(str(root), "tapvid_kinetics.csv"))
        if custom_annotation_path is not None:
            if annotation_name.endswith(".pkl"):
                has_single = os.path.exists(custom_annotation_path)
            elif annotation_name.endswith(".csv"):
                has_csv = os.path.exists(custom_annotation_path)
        has_shards = bool(glob.glob(os.path.join(str(root), "*_of_*.pkl")))
        prefer_sharded = (
            backend == "sharded_pkl"
            or annotation_name.endswith(".index.json")
            or (backend == "auto" and has_shards and not has_single and not has_csv)
        )

        if prefer_sharded:
            if TAPVidKineticsShardedIterableDataset is None:
                raise ImportError("TAPVidKineticsShardedIterableDataset not available")
            sharded_kwargs = dict(kwargs)
            sharded_kwargs.pop("annotation_file", None)
            return TAPVidKineticsShardedIterableDataset(
                root,
                split=str(split) if split is not None else "train",
                annotation_file=annotation_file,
                **sharded_kwargs,
            )

        if TAPVidKineticsDataset is None:
            raise ImportError("TAPVidKineticsDataset not available")
        return TAPVidKineticsDataset(root, **kwargs)

    if name in {"megadepth_pairs", "megadepth_pair", "megadepth_two_view"}:
        if MegaDepthPairDataset is None:
            raise ImportError("MegaDepthPairDataset not available")
        if split is not None:
            kwargs["split"] = split
        return MegaDepthPairDataset(root, **kwargs)

    raise ValueError(f"Unknown dataset: {name}")


def _limit_dataset_samples(dataset, max_samples):
    """
    Best-effort sample limiter for short-run experiments.

    For map-style datasets we use `Subset`. For iterable datasets we wrap the
    iterator and stop after a per-worker quota so the total cap stays bounded
    even when DataLoader uses multiple workers.
    """
    if max_samples is None:
        return dataset
    try:
        max_samples = int(max_samples)
    except (TypeError, ValueError):
        return dataset
    if max_samples <= 0:
        return dataset

    if isinstance(dataset, IterableDataset):
        class _LimitedIterableDataset(IterableDataset):
            def __init__(self, inner, limit):
                super().__init__()
                self.inner = inner
                self.limit = int(limit)

            def _local_limit(self) -> int:
                total = max(0, int(self.limit))
                if total <= 0:
                    return 0
                world = 1
                rank = 0
                num_workers = 1
                worker_id = 0
                try:
                    if dist.is_available() and dist.is_initialized():
                        world = max(1, int(dist.get_world_size()))
                        rank = max(0, int(dist.get_rank()))
                except Exception:
                    world = 1
                    rank = 0
                worker = torch.utils.data.get_worker_info()
                if worker is not None:
                    worker_id = max(0, int(worker.id))
                    num_workers = max(1, int(worker.num_workers))
                shard_count = max(1, world * num_workers)
                shard_index = rank * num_workers + worker_id
                base = total // shard_count
                extra = total % shard_count
                return base + (1 if shard_index < extra else 0)

            def __iter__(self):
                local_limit = self._local_limit()
                if local_limit <= 0:
                    return
                for idx, sample in enumerate(self.inner):
                    if idx >= local_limit:
                        break
                    yield sample

            def __len__(self):
                try:
                    return min(int(len(self.inner)), int(self.limit))
                except Exception:
                    return int(self.limit)

        return _LimitedIterableDataset(dataset, max_samples)

    try:
        dataset_len = len(dataset)
        limit = min(int(dataset_len), int(max_samples))
        return Subset(dataset, range(limit))
    except Exception:
        return dataset


def get_dataloader(name: str, root: str, batch_size: int = 1, **kwargs):
    """
    获取数据加载器

    Args:
        name: 数据集名称
        root: 数据集根目录
        batch_size: batch大小
        split: 数据集划分 ('train', 'val', 'test')
        **kwargs: 其他参数
    """
    from torch.utils.data import DataLoader, DistributedSampler

    seed = kwargs.pop("seed", None)
    distributed = bool(kwargs.pop("distributed", False))
    rank = int(kwargs.pop("rank", 0))
    world_size = int(kwargs.pop("world_size", 1))
    num_workers = int(kwargs.pop("num_workers", 4))
    pin_memory = bool(kwargs.pop("pin_memory", True))
    persistent_workers = bool(kwargs.pop("persistent_workers", False))
    prefetch_factor = kwargs.pop("prefetch_factor", None)
    multiprocessing_context = kwargs.pop("multiprocessing_context", None)
    base_tracks_dir = kwargs.pop("base_tracks_dir", None)
    base_tracks_strict = bool(kwargs.pop("base_tracks_strict", False))
    base_tracks_query_tol = float(kwargs.pop("base_tracks_query_tol", 1e-4) or 1e-4)
    max_samples = kwargs.pop("max_samples", None)
    split = kwargs.pop("split", None)

    normalized_name = normalize_dataset_name(name)

    # If base_tracks are injected, dataset-level random augmentation would make cached base_tracks/base_visibility
    # inconsistent with the returned sample. To keep them aligned, disable dataset-level augmentation and re-apply it
    # after cache injection.
    augmentation_cfg = kwargs.get("augmentation", None)

    def _aug_enabled(cfg) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, bool):
            return bool(cfg)
        try:
            enabled_attr = getattr(cfg, "enabled", None)
            if enabled_attr is not None:
                return bool(enabled_attr)
        except Exception:
            pass
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled", True))
        return True

    aug_is_enabled = _aug_enabled(augmentation_cfg)
    apply_aug_after_cache = base_tracks_dir is not None and aug_is_enabled
    if apply_aug_after_cache:
        kwargs["augmentation"] = {"enabled": False}

    dataset = get_dataset(normalized_name, root, split=split, **kwargs)
    if base_tracks_dir is not None:
        try:
            from .base_tracks import BaseTracksCacheWrapper

            dataset = BaseTracksCacheWrapper(
                dataset,
                base_tracks_dir=str(base_tracks_dir),
                dataset_name=normalized_name,
                strict=base_tracks_strict,
                query_tol=base_tracks_query_tol,
            )
            if apply_aug_after_cache:
                from .augmented_base_tracks import AugmentationAfterBaseTracksWrapper

                dataset = AugmentationAfterBaseTracksWrapper(dataset, augmentation=augmentation_cfg)
                logger.info(
                    "base_tracks_dir enabled with augmentation: applying augmentation after cache injection."
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to enable base_tracks_dir={base_tracks_dir}: {exc}") from exc

    dataset = _limit_dataset_samples(dataset, max_samples)

    is_iterable_dataset = isinstance(dataset, torch.utils.data.IterableDataset)

    # TAP-Vid eval-style datasets run with batch_size=1.
    if normalized_name in ["davis", "kinetics"]:
        batch_size = 1

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        base_seed = int(seed)
        if distributed and world_size > 1:
            base_seed += rank
        generator.manual_seed(base_seed)
        worker_init_fn = _seed_worker

    split_name = str(split).lower() if split is not None else None
    is_train_split = split_name in (None, "", "train", "training")
    shuffle = normalized_name == "kubric" and is_train_split and not is_iterable_dataset
    drop_last = normalized_name == "kubric" and is_train_split and not is_iterable_dataset
    if drop_last:
        try:
            dataset_len = len(dataset)
            if dataset_len < batch_size:
                drop_last = False
            if distributed and world_size > 1:
                if dataset_len < world_size:
                    drop_last = False
                else:
                    samples_per_rank = dataset_len // world_size
                    if samples_per_rank < batch_size:
                        drop_last = False
        except TypeError:
            pass

    sampler = None
    if distributed and world_size > 1 and not is_iterable_dataset:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=int(seed) if seed is not None else 0,
        )
        shuffle = False

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            try:
                dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
            except (TypeError, ValueError):
                pass
        if multiprocessing_context is not None:
            dataloader_kwargs["multiprocessing_context"] = multiprocessing_context

    return DataLoader(**dataloader_kwargs)


__all__ = [
    "TAPVidKubricDataset",
    "TAPVidDAVISDataset",
    "TAPVidKineticsDataset",
    "TAPVidKineticsShardedIterableDataset",
    "MegaDepthPairDataset",
    "create_kubric_dataloader",
    "compute_tapvid_metrics",
    "compute_dataset_metrics",
    "PointTrackingAugmentation",
    "normalize_dataset_name",
    "get_dataset",
    "get_dataloader",
]
