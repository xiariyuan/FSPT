"""
FSPT Datasets Module

支持的数据集:
- TAP-Vid-Kubric: 合成数据，用于训练
- TAP-Vid-DAVIS: 真实视频，用于测试
- TAP-Vid-Kinetics: 真实视频，用于测试
"""

import logging
import random
import numpy as np
import torch

from .tapvid_davis import TAPVidDAVISDataset
from .metrics import compute_tapvid_metrics, compute_dataset_metrics
from .augmentation import PointTrackingAugmentation

logger = logging.getLogger(__name__)

# 尝试导入可选数据集
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


def normalize_dataset_name(name: str) -> str:
    """规范化数据集名称（支持别名）"""
    if not name:
        raise ValueError("Dataset name cannot be empty")
    normalized = name.lower().replace("-", "_").strip()
    if normalized.startswith("tapvid_"):
        normalized = normalized[len("tapvid_") :]
    return normalized


def _seed_worker(worker_id: int):
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
        
    Returns:
        dataset: PyTorch Dataset
    """
    name = normalize_dataset_name(name)
    
    # 提取split参数，只有Kubric支持
    split = kwargs.pop('split', None)
    
    if name == 'kubric':
        # Kubric支持split参数
        if split is not None:
            kwargs['split'] = split

        # 检查是否使用pickle格式
        use_tfds = bool(kwargs.get("use_tfds", False))
        annotation_file = kwargs.get("annotation_file", None)

        if not use_tfds and annotation_file:
            annotation_name = str(annotation_file).lower()
            if annotation_name.endswith(".index.json"):
                if TAPVidKubricShardedIterableDataset is None:
                    raise ImportError("TAPVidKubricShardedIterableDataset not available")
                return TAPVidKubricShardedIterableDataset(root, **kwargs)
            # 使用pickle数据集
            if TAPVidKubricPickleDataset is None:
                raise ImportError("TAPVidKubricPickleDataset not available")
            return TAPVidKubricPickleDataset(root, **kwargs)
        elif use_tfds:
            # 使用TFDS数据集
            if TAPVidKubricDataset is None:
                raise ImportError("TAPVidKubricDataset not available")
            if TAPVidKubricTFDSIterableDataset is not None:
                return TAPVidKubricTFDSIterableDataset(root, **kwargs)
            return TAPVidKubricDataset(root, **kwargs)
        else:
            # 默认使用TFDS
            if TAPVidKubricDataset is None:
                raise ImportError("TAPVidKubricDataset not available")
            return TAPVidKubricDataset(root, **kwargs)
    elif name == 'davis':
        # DAVIS只用于测试，不需要split参数
        return TAPVidDAVISDataset(root, **kwargs)
    elif name == 'kinetics':
        if TAPVidKineticsDataset is None:
            raise ImportError("TAPVidKineticsDataset not available")
        # Kinetics只用于测试，不需要split参数
        return TAPVidKineticsDataset(root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataloader(name: str, root: str, batch_size: int = 1, **kwargs):
    """
    获取数据加载器
    
    Args:
        name: 数据集名称
        root: 数据集根目录
        batch_size: 批大小
        split: 数据集划分 ('train', 'val', 'test')
        **kwargs: 其他参数
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader, DistributedSampler
    
    seed = kwargs.pop('seed', None)
    distributed = bool(kwargs.pop('distributed', False))
    rank = int(kwargs.pop('rank', 0))
    world_size = int(kwargs.pop('world_size', 1))
    num_workers = int(kwargs.pop('num_workers', 4))
    pin_memory = bool(kwargs.pop('pin_memory', True))
    base_tracks_dir = kwargs.pop('base_tracks_dir', None)
    base_tracks_strict = bool(kwargs.pop('base_tracks_strict', False))
    base_tracks_query_tol = float(kwargs.pop('base_tracks_query_tol', 1e-4) or 1e-4)
    split = kwargs.pop('split', None)  # 提取split参数，传递给数据集
    
    normalized_name = normalize_dataset_name(name)
    
    # 将split参数传递给get_dataset（内部会处理）
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

    # When base_tracks are injected, dataset-level random augmentation would make the cached
    # base_tracks/base_visibility inconsistent with the returned sample. To keep them aligned,
    # we disable dataset-level augmentation and re-apply it after cache injection.
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

    is_iterable_dataset = isinstance(dataset, torch.utils.data.IterableDataset)
    
    # 测试集batch_size=1
    if normalized_name in ['davis', 'kinetics']:
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
    is_train_split = split_name in (None, '', 'train', 'training')
    shuffle = (normalized_name == 'kubric' and is_train_split and not is_iterable_dataset)
    drop_last = (normalized_name == 'kubric' and is_train_split and not is_iterable_dataset)
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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    
    return dataloader


__all__ = [
    'TAPVidKubricDataset',
    'TAPVidDAVISDataset',
    'TAPVidKineticsDataset',
    'create_kubric_dataloader',
    'compute_tapvid_metrics',
    'compute_dataset_metrics',
    'PointTrackingAugmentation',
    'normalize_dataset_name',
    'get_dataset',
    'get_dataloader',
]
