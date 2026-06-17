"""
TAP-Vid DAVIS Dataset

DAVIS是一个经典的视频分割数据集，TAP-Vid在其上标注了点追踪数据。
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    OmegaConf = None
from .augmentation import PointTrackingAugmentation, TemporalAugmentation

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TAPVidDAVISDataset(Dataset):
    """
    TAP-Vid DAVIS测试数据集
    
    数据格式:
    - video: (T, H, W, 3) uint8
    - points: (N, T, 2) float32, 像素坐标
    - occluded: (N, T) bool
    - query_points: (N, 3) float32, [t, y, x] 像素坐标
    
    Args:
        root: 数据集根目录，应包含 tapvid_davis.pkl
        resolution: 目标分辨率 (H, W)，None表示使用原始分辨率
        augmentation: 数据增强配置或开关（默认关闭）
        num_points: 可选点数上限（默认使用全部点）
        max_retries: 单个样本失败时的最大重试次数
    """
    
    def __init__(
        self,
        root: str,
        resolution: Optional[Tuple[int, int]] = None,
        augmentation: Optional[object] = None,
        num_points: Optional[int] = None,
        max_retries: int = 5,
    ):
        self.root = root
        self.resolution = resolution
        self.num_points = num_points
        self.max_retries = max(0, int(max_retries))
        self.aug_cfg = self._normalize_aug_config(augmentation)
        self.augmentation = self.aug_cfg.get('enabled', False)
        self.spatial_aug = None
        self.temporal_aug = None
        if self.augmentation:
            cfg = self.aug_cfg
            random_scale = cfg.get('random_scale', None)
            if isinstance(random_scale, list):
                random_scale = tuple(random_scale)
            crop_size = cfg.get('crop_size', self.resolution)
            random_rotation = float(cfg.get('random_rotation', 0.0) or 0.0)
            self.spatial_aug = PointTrackingAugmentation(
                random_crop=cfg.get('random_crop', True),
                random_flip=cfg.get('random_flip', True),
                color_jitter=cfg.get('color_jitter', 0.4),
                random_scale=random_scale,
                random_rotation=random_rotation,
                crop_size=crop_size or (256, 256),
            )

            temporal_cfg = cfg.get('temporal', {})
            if not isinstance(temporal_cfg, dict):
                temporal_cfg = {}
            temporal_enabled = bool(temporal_cfg.get('enabled', False))
            if any(k in cfg for k in ['random_reverse', 'random_speed', 'random_start']):
                temporal_enabled = True
            if temporal_enabled:
                random_speed = temporal_cfg.get('random_speed', cfg.get('random_speed', (0.5, 2.0)))
                if isinstance(random_speed, list):
                    random_speed = tuple(random_speed)
                self.temporal_aug = TemporalAugmentation(
                    random_reverse=temporal_cfg.get('random_reverse', cfg.get('random_reverse', True)),
                    random_speed=random_speed,
                    random_start=temporal_cfg.get('random_start', cfg.get('random_start', True)),
                )
        
        # 加载标注
        anno_path = os.path.join(root, 'tapvid_davis.pkl')
        if not os.path.exists(anno_path):
            raise FileNotFoundError(
                f"Annotation file not found at {anno_path}. "
                "Please download TAP-Vid DAVIS dataset first."
            )
        
        with open(anno_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.video_names = sorted(list(self.data.keys()))
        self.use_video_files = False
        if self.video_names:
            sample = self.data[self.video_names[0]]
            self.use_video_files = 'video' not in sample
        if self.use_video_files:
            if not CV2_AVAILABLE:
                raise ImportError("cv2 is required to load DAVIS videos from files.")
            filtered = []
            for name in self.video_names:
                sample = self.data.get(name, {})
                video_path = sample.get('video_path')
                if video_path is not None:
                    if not os.path.isabs(video_path):
                        video_path = os.path.join(self.root, video_path)
                    if os.path.exists(video_path):
                        filtered.append(name)
                        continue
                video_dir = os.path.join(self.root, 'DAVIS', 'JPEGImages', '480p', name)
                if os.path.isdir(video_dir):
                    frame_files = [
                        f for f in os.listdir(video_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    if frame_files:
                        filtered.append(name)
            self.video_names = filtered
            if not self.video_names:
                raise FileNotFoundError(
                    f"No DAVIS frames found under {os.path.join(self.root, 'DAVIS', 'JPEGImages', '480p')}."
                )
        
        print(f"Loaded TAP-Vid DAVIS dataset with {len(self.video_names)} videos")
        
    def _normalize_aug_config(self, augmentation) -> Dict:
        """统一增强配置"""
        if augmentation is None:
            return {'enabled': False}
        if isinstance(augmentation, bool):
            return {'enabled': augmentation}
        if OMEGACONF_AVAILABLE and OmegaConf.is_config(augmentation):
            cfg = OmegaConf.to_container(augmentation, resolve=True)
        elif isinstance(augmentation, dict):
            cfg = dict(augmentation)
        else:
            try:
                cfg = dict(augmentation)
            except Exception:
                cfg = {'enabled': True}
        cfg.setdefault('enabled', True)
        return cfg

    def _ensure_query_visible(
        self,
        query_points: torch.Tensor,
        target_points: torch.Tensor,
        occluded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """确保查询帧点可见，避免增强后出现不可见查询点。"""
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
                perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:self.num_points]
                chosen = valid_idx[perm]
            else:
                rand = torch.randint(0, valid_idx.numel(), (self.num_points,), device=valid_idx.device)
                chosen = valid_idx[rand]
            return query_points[chosen], target_points[chosen], occluded[chosen]
        return query_points[valid_idx], target_points[valid_idx], occluded[valid_idx]
        
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx, retry: int = 0):
        video_name = self.video_names[idx]
        video_data = self.data[video_name]
        query_generated = False
        
        # 获取数据
        if 'video' in video_data:
            video = video_data['video']  # (T, H, W, 3) numpy array
        else:
            video_dir = video_data.get('video_path')
            if video_dir is None:
                video_dir = os.path.join(self.root, 'DAVIS', 'JPEGImages', '480p', video_name)
            elif not os.path.isabs(video_dir):
                video_dir = os.path.join(self.root, video_dir)
            try:
                video = self._load_video_from_frames(video_dir)
            except Exception as e:
                if retry < self.max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise e

        if video is None:
            if retry < self.max_retries:
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("Empty video sample encountered.")
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        video = np.asarray(video)
        if video.ndim == 1 and video.size > 0 and isinstance(video[0], np.ndarray):
            video = np.stack(video)
        if video.size == 0 or video.shape[0] == 0:
            if retry < self.max_retries:
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("Empty video sample encountered.")
        T, H, W, _ = video.shape
        points = video_data.get('points')  # (N, T, 2) 像素坐标
        if points is None:
            raise KeyError("Missing 'points' in DAVIS sample.")
        occluded = video_data.get('occluded')  # (N, T)
        if occluded is None:
            visibility = video_data.get('visibility') or video_data.get('vis')
            if visibility is not None:
                occluded = ~np.array(visibility, dtype=bool)
        if occluded is None:
            raise KeyError("Missing 'occluded' or 'visibility' in DAVIS sample.")
        query_points = video_data.get('query_points')  # (N, 3) [t, y, x]
        if query_points is None:
            query_points = video_data.get('queries')
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if isinstance(occluded, torch.Tensor):
            occluded = occluded.detach().cpu().numpy()
        if isinstance(query_points, torch.Tensor):
            query_points = query_points.detach().cpu().numpy()
        points = np.array(points, dtype=np.float32)
        occluded = np.array(occluded, dtype=bool)
        query_points = np.array(query_points, dtype=np.float32) if query_points is not None else None
        if points.ndim == 3 and points.shape[0] == T and points.shape[1] != T:
            points = np.transpose(points, (1, 0, 2))
        if occluded.ndim == 2 and occluded.shape[0] == T and occluded.shape[1] != T:
            occluded = np.transpose(occluded, (1, 0))
        if points.shape[1] != T:
            min_t = min(points.shape[1], T)
            if min_t <= 0:
                if retry < self.max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("Invalid temporal alignment.")
            video = video[:min_t]
            points = points[:, :min_t]
            occluded = occluded[:, :min_t]
            T = min_t
        if query_points is None or query_points.shape[0] == 0:
            query_points = np.stack(
                [
                    np.zeros(points.shape[0], dtype=np.float32),
                    points[:, 0, 0],
                    points[:, 0, 1],
                ],
                axis=1,
            )
            query_generated = True
        if points.shape[0] != query_points.shape[0] or occluded.shape[0] != points.shape[0]:
            min_len = min(points.shape[0], query_points.shape[0], occluded.shape[0])
            if min_len == 0:
                if retry < self.max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("No valid points remain after alignment.")
            points = points[:min_len]
            occluded = occluded[:min_len]
            query_points = query_points[:min_len]
        if points.shape[0] == 0:
            if retry < self.max_retries:
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("No points available in DAVIS sample.")
        N = points.shape[0]
        
        # 保存原始尺寸
        original_size = (H, W)
        
        # 归一化点坐标到[0, 1]（若已归一化则保持）
        points_normalized = points.copy().astype(np.float32)
        if points_normalized.size > 0 and points_normalized.max() > 1.5:
            points_normalized[:, :, 0] /= H  # y
            points_normalized[:, :, 1] /= W  # x
        
        query_points_normalized = query_points.copy().astype(np.float32)
        query_points_normalized[:, 0] = np.clip(query_points_normalized[:, 0], 0, T - 1)
        if query_points_normalized.size > 0 and query_points_normalized[:, 1:].max() > 1.5:
            query_points_normalized[:, 1] /= H  # y
            query_points_normalized[:, 2] /= W  # x

        # 可选采样点数
        if self.num_points is not None and self.num_points > 0 and points_normalized.shape[0] > self.num_points:
            indices = np.random.choice(points_normalized.shape[0], self.num_points, replace=False)
            points_normalized = points_normalized[indices]
            occluded = occluded[indices]
            query_points_normalized = query_points_normalized[indices]
        
        # 转换为PyTorch tensor
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        if video.max() > 1.5:
            video = video / 255.0
        target_points = torch.from_numpy(points_normalized).float()
        occluded = torch.from_numpy(occluded.astype(bool))
        query_points = torch.from_numpy(query_points_normalized).float()
        query_points[:, 1:3] = torch.clamp(query_points[:, 1:3], 0.0, 1.0)

        # 数据增强
        if self.augmentation and self.spatial_aug is not None:
            outputs = self.spatial_aug(video, target_points, occluded, query_points)
            if len(outputs) == 4:
                video, target_points, occluded, query_points = outputs
            else:
                video, target_points, occluded = outputs
        if self.augmentation and self.temporal_aug is not None:
            video, target_points, occluded, query_points = self.temporal_aug(
                video, target_points, occluded, query_points
            )
        query_points[:, 0] = torch.clamp(query_points[:, 0], 0, video.shape[0] - 1)
        if self.augmentation or query_generated:
            query_points, target_points, occluded = self._ensure_query_visible(
                query_points, target_points, occluded
            )

        # 调整分辨率
        if self.resolution is not None:
            new_H, new_W = self.resolution
            if video.shape[-2:] != (new_H, new_W):
                video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        return {
            'video': video,  # (T, 3, H, W)
            'query_points': query_points,  # (N, 3) [t, y, x] 归一化
            'target_points': target_points,  # (N, T, 2) [y, x] 归一化
            'occluded': occluded,  # (N, T)
            'video_name': video_name,
            'original_size': original_size,
        }
    
    def _resize_video(self, video: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
        """调整视频分辨率（保留兼容接口）"""
        T, H, W, C = video.shape
        new_H, new_W = resolution
        video_t = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        video_t = F.interpolate(video_t, size=(new_H, new_W), mode='bilinear', align_corners=False)
        video_t = video_t.permute(0, 2, 3, 1).clamp(0, 255)
        if np.issubdtype(video.dtype, np.integer):
            return video_t.round().to(torch.uint8).cpu().numpy()
        return video_t.cpu().numpy().astype(video.dtype)

    def _load_video_from_frames(self, video_dir: str) -> np.ndarray:
        """从帧目录加载视频"""
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required to load DAVIS videos from files.")
        if os.path.isfile(video_dir):
            cap = cv2.VideoCapture(video_dir)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if not frames:
                raise FileNotFoundError(f"Failed to read video: {video_dir}")
            return np.stack(frames)
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        )
        if not frame_files:
            raise FileNotFoundError(f"No frames found in: {video_dir}")
        frames = []
        for name in frame_files:
            path = os.path.join(video_dir, name)
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"Failed to read frames from: {video_dir}")
        return np.stack(frames)
    
    def get_video_info(self, idx):
        """获取视频信息"""
        video_name = self.video_names[idx]
        video_data = self.data[video_name]
        
        video = video_data['video']
        T, H, W, _ = video.shape
        N = video_data['points'].shape[0]
        
        return {
            'video_name': video_name,
            'num_frames': T,
            'resolution': (H, W),
            'num_points': N,
        }


def create_davis_dataloader(
    root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    **kwargs
):
    """创建DAVIS数据加载器"""
    from torch.utils.data import DataLoader
    
    dataset = TAPVidDAVISDataset(root, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试数据集
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='datasets/tapvid_davis')
    args = parser.parse_args()
    
    dataset = TAPVidDAVISDataset(args.root)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 测试第一个样本
    sample = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Video name: {sample['video_name']}")
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  Query points shape: {sample['query_points'].shape}")
    print(f"  Target points shape: {sample['target_points'].shape}")
    print(f"  Occluded shape: {sample['occluded'].shape}")
    print(f"  Original size: {sample['original_size']}")
    
    # 统计
    print(f"\n  Query points range: [{sample['query_points'].min():.3f}, {sample['query_points'].max():.3f}]")
    print(f"  Target points range: [{sample['target_points'].min():.3f}, {sample['target_points'].max():.3f}]")
    print(f"  Occluded ratio: {sample['occluded'].float().mean():.3f}")
