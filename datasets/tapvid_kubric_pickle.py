"""
TAP-Vid Kubric Pickle Dataset

简化的数据集类，用于加载预处理的pickle格式数据
内存消耗: 5-10GB (vs TFDS的29GB)
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class TAPVidKubricPickleDataset(Dataset):
    """
    TAP-Vid Kubric数据集 - Pickle格式

    使用预处理的pickle文件，大幅降低内存消耗

    Args:
        root: 数据集根目录
        annotation_file: pickle文件名 (e.g., 'train.pkl')
        split: 数据集分割 ('train', 'val', 'test')
        num_frames: 帧数 (用于验证)
        num_points: 点数 (用于验证)
        resolution: 分辨率 [H, W] (用于验证)
        augmentation: 数据增强配置 (dict)
    """

    def __init__(
        self,
        root,
        annotation_file='train.pkl',
        split='train',
        num_frames=24,
        num_points=256,
        resolution=(256, 256),
        augmentation=None,
        **kwargs
    ):
        self.root = root
        self.annotation_file = annotation_file
        self.split = split
        self.num_frames = num_frames
        self.num_points = num_points
        self.resolution = resolution
        self.augmentation = augmentation or {}

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载pickle数据"""
        pkl_path = os.path.join(self.root, self.annotation_file)

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Pickle file not found: {pkl_path}\n"
                f"Please run preprocess_kubric_tfds.py first."
            )

        logger.info(f"Loading data from pickle: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            self.samples = pickle.load(f)

        logger.info(f"Loaded {len(self.samples)} samples from pickle")

        # 验证数据格式
        if len(self.samples) > 0:
            sample = self.samples[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Video shape: {sample['video'].shape}")
            logger.info(f"Query points shape: {sample['query_points'].shape}")
            logger.info(f"Target points shape: {sample['target_points'].shape}")
            logger.info(f"Occluded shape: {sample['occluded'].shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取样本

        Returns:
            dict: {
                'video': (T, H, W, 3) float32 [0, 1]
                'query_points': (N, 3) float32 [t, y, x]
                'target_points': (N, T, 2) float32 [x, y]
                'occluded': (N, T) bool
            }
        """
        sample = self.samples[idx]

        # 转换为tensor
        video = torch.from_numpy(sample['video']).float() / 255.0  # (T, H, W, 3)
        query_points = torch.from_numpy(sample['query_points']).float()  # (N, 3)
        target_points = torch.from_numpy(sample['target_points']).float()  # (N, T, 2)
        occluded = torch.from_numpy(sample['occluded'])  # (N, T)

        # 应用数据增强（如果启用）
        if self.augmentation.get('enabled', False) and self.split == 'train':
            video, query_points, target_points, occluded = self._apply_augmentation(
                video, query_points, target_points, occluded
            )

        return {
            'video': video,
            'query_points': query_points,
            'target_points': target_points,
            'occluded': occluded,
        }

    def _apply_augmentation(self, video, query_points, target_points, occluded):
        """
        应用数据增强

        支持的增强:
        - random_flip: 随机水平翻转
        - random_crop: 随机裁剪
        - color_jitter: 颜色抖动
        """
        T, H, W, C = video.shape
        N = query_points.shape[0]

        # 随机水平翻转
        if self.augmentation.get('random_flip', False):
            if torch.rand(1).item() > 0.5:
                # 翻转视频
                video = torch.flip(video, dims=[2])  # 沿宽度翻转

                # 翻转点坐标
                query_points[:, 2] = W - 1 - query_points[:, 2]  # x坐标
                target_points[:, :, 0] = W - 1 - target_points[:, :, 0]  # x坐标

        # 颜色抖动
        if self.augmentation.get('color_jitter', 0) > 0:
            jitter = self.augmentation['color_jitter']

            # 亮度
            brightness_factor = 1.0 + torch.rand(1).item() * jitter * 2 - jitter
            video = video * brightness_factor

            # 对比度
            contrast_factor = 1.0 + torch.rand(1).item() * jitter * 2 - jitter
            mean = video.mean(dim=(0, 1, 2), keepdim=True)
            video = (video - mean) * contrast_factor + mean

            # 裁剪到[0, 1]
            video = torch.clamp(video, 0, 1)

        return video, query_points, target_points, occluded


class TAPVidKubricPickleIterableDataset(Dataset):
    """
    可迭代版本的Pickle数据集

    用于非常大的数据集，避免一次性加载所有数据到内存
    """

    def __init__(self, root, annotation_file='train.pkl', **kwargs):
        self.root = root
        self.annotation_file = annotation_file

        pkl_path = os.path.join(root, annotation_file)

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

        # 只加载元数据，不加载实际数据
        metadata_file = pkl_path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
                self.num_samples = self.metadata['num_samples']
        else:
            # 如果没有元数据，需要加载一次来获取长度
            with open(pkl_path, 'rb') as f:
                samples = pickle.load(f)
                self.num_samples = len(samples)

        self.pkl_path = pkl_path
        logger.info(f"Initialized iterable dataset with {self.num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 每次访问时重新加载（内存友好，但速度慢）
        with open(self.pkl_path, 'rb') as f:
            samples = pickle.load(f)
            sample = samples[idx]

        # 转换为tensor
        return {
            'video': torch.from_numpy(sample['video']).float() / 255.0,
            'query_points': torch.from_numpy(sample['query_points']).float(),
            'target_points': torch.from_numpy(sample['target_points']).float(),
            'occluded': torch.from_numpy(sample['occluded']),
        }


# 便捷函数
def load_kubric_pickle(root, split='train', **kwargs):
    """
    便捷函数：加载Kubric pickle数据集

    Args:
        root: 数据集根目录
        split: 'train', 'val', 'test'
        **kwargs: 其他参数

    Returns:
        TAPVidKubricPickleDataset
    """
    annotation_file = f"{split}.pkl"
    return TAPVidKubricPickleDataset(
        root=root,
        annotation_file=annotation_file,
        split=split,
        **kwargs
    )


if __name__ == '__main__':
    # 测试代码
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tapvid_kubric_pickle.py <data_root>")
        sys.exit(1)

    root = sys.argv[1]

    print("Testing TAPVidKubricPickleDataset...")

    # 加载数据集
    dataset = load_kubric_pickle(root, split='train')

    print(f"\nDataset size: {len(dataset)}")

    # 测试加载样本
    sample = dataset[0]

    print(f"\nSample 0:")
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  Video dtype: {sample['video'].dtype}")
    print(f"  Video range: [{sample['video'].min():.3f}, {sample['video'].max():.3f}]")
    print(f"  Query points shape: {sample['query_points'].shape}")
    print(f"  Target points shape: {sample['target_points'].shape}")
    print(f"  Occluded shape: {sample['occluded'].shape}")

    print("\n✓ Dataset loaded successfully!")
