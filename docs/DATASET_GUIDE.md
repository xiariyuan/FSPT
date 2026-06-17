# FSPT 数据集完整指南

## 一、数据集概览

### 1.1 TAP-Vid Benchmark 数据集族

| 数据集 | 类型 | 视频数 | 帧数/视频 | 分辨率 | 点数/视频 | 用途 | 磁盘大小 |
|--------|------|--------|-----------|--------|-----------|------|----------|
| **TAP-Vid-Kubric** | 合成 | 38,325+799 | 24 | 256×256 | ~20-30 | 训练 | **~150GB** |
| **TAP-Vid-DAVIS** | 真实 | 30 | 34-104 | 480p | ~20 | 测试 | **~500MB** (pkl) |
| **TAP-Vid-Kinetics** | 真实 | 1,189 | 250 | 可变→256 | ~20 | 测试 | **~30-50GB** (视频) |

### 1.2 详细存储需求

```
总存储需求估算:
├── TAP-Vid-Kubric (训练)
│   ├── MOVi-E TFDS格式: ~150GB
│   └── 预处理pickle: ~80GB (可选，加速加载)
│
├── TAP-Vid-DAVIS (测试)
│   └── tapvid_davis.pkl: ~500MB
│
├── TAP-Vid-Kinetics (测试)
│   ├── 原始视频: ~30-50GB (需要从Kinetics-700下载)
│   └── 预处理pkl: ~15GB
│
最小需求 (仅测试): ~1GB
完整需求 (训练+测试): ~200GB
```

跨平台一键下载（推荐）：
```bash
cd "$PROJECT_DIR"
python scripts/download_all_datasets.py --root datasets
# 可选：下载Kubric TFDS调试数据（不含点标注）
# python scripts/download_all_datasets.py --root datasets --kubric-tfds
```

### 1.3 扩展数据集

| 数据集 | 类型 | 视频数 | 描述 | 磁盘大小 | 用途 |
|--------|------|--------|------|----------|------|
| **DriveTrack** | 真实 | 24小时 | 自动驾驶，10亿点轨迹 | **~500GB** | 训练+测试 |
| **PointOdyssey** | 合成 | 2,000+ | 长视频(2000帧)，复杂遮挡 | **~300GB** | 训练 |
| **TAPVid-3D** | 真实 | 4,000+ | 3D点追踪，带深度 | **~50GB** | 测试 |

---

## 二、TAP-Vid-Kubric (训练数据)

### 2.1 数据集描述

Kubric是一个合成数据生成框架，可以生成具有精确标注的点追踪数据。

**特点**:
- 完美的ground truth标注
- 可控的遮挡和运动
- 多样的物体和场景
- 适合预训练

### 2.2 下载方式

```bash
# 方式1: 使用TAP-Vid官方Kubric标注 pkl (推荐)
mkdir -p datasets/tapvid_kubric
# 将 tapvid_kubric_train.pkl 放到 datasets/tapvid_kubric/

# 方式2: 使用TensorFlow Datasets (仅调试，无官方轨迹)
pip install tensorflow tensorflow-datasets

python << 'EOF'
import tensorflow_datasets as tfds
ds = tfds.load('movi_e/256x256', split='train', data_dir='datasets/tapvid_kubric')
print(f"Downloaded {len(ds)} videos")
EOF

# 方式3: 使用gsutil从Google Cloud Storage下载TFDS
pip install gsutil
mkdir -p datasets/tapvid_kubric/movi_e
gsutil -m cp -r "gs://kubric-public/tfds/movi_e/256x256" datasets/tapvid_kubric/movi_e/
```

### 2.3 数据格式

```python
# Kubric数据格式
{
    'video': tf.Tensor,           # (T, H, W, 3) uint8, RGB视频帧
    'segmentations': tf.Tensor,   # (T, H, W, 1) uint8, 实例分割
    'depth': tf.Tensor,           # (T, H, W, 1) float32, 深度图
    'forward_flow': tf.Tensor,    # (T-1, H, W, 2) float32, 前向光流
    'backward_flow': tf.Tensor,   # (T-1, H, W, 2) float32, 后向光流
    'metadata': {
        'num_instances': int,
        'video_name': str,
        ...
    }
}
```

### 2.4 TAP-Vid格式转换

```python
import numpy as np
import tensorflow as tf

def convert_kubric_to_tapvid(kubric_sample):
    """
    将Kubric数据转换为TAP-Vid格式
    
    TAP-Vid格式:
    - video: (T, H, W, 3) uint8
    - query_points: (N, 3) float32, [t, y, x] 归一化坐标
    - target_points: (N, T, 2) float32, [y, x] 归一化坐标
    - occluded: (N, T) bool, True表示被遮挡
    """
    video = kubric_sample['video'].numpy()
    T, H, W, _ = video.shape
    
    # 从分割图中采样查询点
    segmentations = kubric_sample['segmentations'].numpy()
    
    query_points = []
    target_points = []
    occluded = []
    
    # 对每个实例采样点
    for instance_id in range(1, kubric_sample['metadata']['num_instances'] + 1):
        # 找到实例在每帧的位置
        for t in range(T):
            mask = segmentations[t, :, :, 0] == instance_id
            if mask.sum() > 0:
                # 随机采样一个点
                ys, xs = np.where(mask)
                idx = np.random.randint(len(ys))
                y, x = ys[idx] / H, xs[idx] / W  # 归一化
                
                # 追踪这个点到所有帧
                track_y, track_x = [], []
                track_occ = []
                
                # 使用光流追踪
                for t2 in range(T):
                    if t2 == t:
                        track_y.append(y)
                        track_x.append(x)
                        track_occ.append(False)
                    else:
                        # 使用光流计算位置
                        # (简化实现，实际应该累积光流)
                        mask_t2 = segmentations[t2, :, :, 0] == instance_id
                        if mask_t2.sum() > 0:
                            ys2, xs2 = np.where(mask_t2)
                            # 找最近的点
                            y2, x2 = ys2.mean() / H, xs2.mean() / W
                            track_y.append(y2)
                            track_x.append(x2)
                            track_occ.append(False)
                        else:
                            track_y.append(0)
                            track_x.append(0)
                            track_occ.append(True)
                
                query_points.append([t, y, x])
                target_points.append(list(zip(track_y, track_x)))
                occluded.append(track_occ)
                break  # 每个实例只采样一个点
    
    return {
        'video': video,
        'query_points': np.array(query_points, dtype=np.float32),
        'target_points': np.array(target_points, dtype=np.float32),
        'occluded': np.array(occluded, dtype=bool),
    }
```

### 2.5 数据加载器

```python
from datasets.tapvid_kubric import TAPVidKubricDataset

dataset = TAPVidKubricDataset(
    root="datasets/tapvid_kubric",
    split="train",
    num_frames=24,
    num_points=256,
    resolution=(256, 256),
    annotation_file="tapvid_kubric_train.pkl",
    use_tfds=False,
)
```

如需使用 TFDS 调试，请设置 `use_tfds=True` 且 `allow_synthetic_tracks=True`。

---

## 三、TAP-Vid-DAVIS (测试数据)

### 3.1 数据集描述

DAVIS是一个经典的视频分割数据集，TAP-Vid在其上标注了点追踪数据。

**特点**:
- 真实视频，多样场景
- 高质量标注
- 包含遮挡和快速运动
- 标准测试集

### 3.2 下载方式

```bash
DATA_DIR="${DATA_DIR:-datasets}"
cd "$DATA_DIR"
mkdir -p tapvid_davis && cd tapvid_davis

# 1. 下载DAVIS 2017视频
wget https://data.vision.ee.ethz.ch/cserber/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-480p.zip

# 2. 下载TAP-Vid DAVIS标注
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl

# 目录结构
# tapvid_davis/
# ├── DAVIS/
# │   └── JPEGImages/
# │       └── 480p/
# │           ├── bear/
# │           ├── blackswan/
# │           └── ...
# └── tapvid_davis.pkl  # 标注文件
```

### 3.3 数据格式

```python
import pickle

# 加载标注
with open('tapvid_davis/tapvid_davis.pkl', 'rb') as f:
    data = pickle.load(f)

# 数据结构
# data是一个字典，key是视频名称
for video_name, video_data in data.items():
    print(f"Video: {video_name}")
    print(f"  video shape: {video_data['video'].shape}")  # (T, H, W, 3)
    print(f"  points shape: {video_data['points'].shape}")  # (N, T, 2)
    print(f"  occluded shape: {video_data['occluded'].shape}")  # (N, T)
    print(f"  query_points shape: {video_data['query_points'].shape}")  # (N, 3)
```

### 3.4 数据加载器

```python
# datasets/tapvid_davis.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class TAPVidDAVISDataset(Dataset):
    """
    TAP-Vid DAVIS测试数据集
    """
    
    def __init__(
        self,
        root: str,
        resolution: tuple = None,  # None表示使用原始分辨率
    ):
        self.root = root
        self.resolution = resolution
        
        # 加载标注
        anno_path = os.path.join(root, 'tapvid_davis.pkl')
        with open(anno_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.video_names = list(self.data.keys())
        
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_data = self.data[video_name]
        
        # 获取数据
        video = video_data['video']  # (T, H, W, 3)
        points = video_data['points']  # (N, T, 2) 像素坐标
        occluded = video_data['occluded']  # (N, T)
        query_points = video_data['query_points']  # (N, 3) [t, y, x]
        
        T, H, W, _ = video.shape
        
        # 归一化点坐标到[0, 1]
        points_normalized = points.copy()
        points_normalized[:, :, 0] /= H
        points_normalized[:, :, 1] /= W
        
        query_points_normalized = query_points.copy()
        query_points_normalized[:, 1] /= H
        query_points_normalized[:, 2] /= W
        
        # 调整分辨率
        if self.resolution is not None:
            video = self._resize_video(video, self.resolution)
        
        # 转换为PyTorch tensor
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        target_points = torch.from_numpy(points_normalized).float()
        occluded = torch.from_numpy(occluded).bool()
        query_points = torch.from_numpy(query_points_normalized).float()
        
        return {
            'video': video,  # (T, 3, H, W)
            'query_points': query_points,  # (N, 3)
            'target_points': target_points,  # (N, T, 2)
            'occluded': occluded,  # (N, T)
            'video_name': video_name,
            'original_size': (H, W),
        }
    
    def _resize_video(self, video, resolution):
        """调整视频分辨率"""
        T, H, W, C = video.shape
        new_H, new_W = resolution
        
        resized = np.zeros((T, new_H, new_W, C), dtype=video.dtype)
        for t in range(T):
            img = Image.fromarray(video[t])
            img = img.resize((new_W, new_H), Image.BILINEAR)
            resized[t] = np.array(img)
        
        return resized
```

---

## 四、TAP-Vid-Kinetics (测试数据)

### 4.1 数据集描述

Kinetics是大规模动作识别数据集，TAP-Vid从中选取视频并标注点追踪。

**特点**:
- 大规模真实视频
- 多样的动作和场景
- 更具挑战性
- 用于泛化能力测试

### 4.2 下载方式

```bash
DATA_DIR="${DATA_DIR:-datasets}"
cd "$DATA_DIR"
mkdir -p tapvid_kinetics && cd tapvid_kinetics

# 1. 下载标注文件
wget https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip
unzip tapvid_kinetics.zip
rm tapvid_kinetics.zip

# 说明:
# - 官方tapvid_kinetics.pkl通常包含视频帧（较大，30~50GB）
# - 如果你使用的是“轻量标注”（不含视频帧），请将视频放入 videos/ 目录
#   并确保安装 opencv-python 以便读取 mp4
```

### 4.3 数据加载器

```python
from datasets.tapvid_kinetics import TAPVidKineticsDataset

dataset = TAPVidKineticsDataset(
    root="datasets/tapvid_kinetics",
    resolution=(256, 256),
    max_frames=250,
)
```

如果 `tapvid_kinetics.pkl` 不包含 `video` 字段，请将 `mp4` 放入 `datasets/tapvid_kinetics/videos/` 并安装 `opencv-python`。

---

## 五、评估指标计算

### 5.1 TAP-Vid官方指标

```python
from datasets.metrics import compute_tapvid_metrics

metrics = compute_tapvid_metrics(
    pred_tracks,
    gt_tracks,
    pred_visibility,
    gt_visibility,
    query_points,
    resolution=tuple(video.shape[-2:]),  # (H, W) 或单一整数
    exclude_query_frame=True,
)
```

### 5.2 数据集级别评估

```python
from datasets.metrics import compute_dataset_metrics

dataset_metrics = compute_dataset_metrics(
    model,
    dataloader,
    device='cuda',
    exclude_query_frame=True,
)
```

---

## 六、数据增强策略

### 6.1 标准增强

```yaml
# configs/fspt_base.yaml
data:
  augmentation:
    random_crop: true
    random_flip: true
    color_jitter: 0.4
    random_scale: [0.8, 1.2]
    random_rotation: 0.0
    temporal:
      enabled: false
      random_reverse: true
      random_speed: [0.5, 2.0]
      random_start: true

  val:
    augmentation:
      enabled: false
```

说明：
- 训练增强会同步变换 `query_points` 与 `occluded`
- 验证/评估默认关闭增强，避免影响指标
- `data.val.num_points` 仅用于上限采样（默认使用全部点）

---

## 七、完整数据Pipeline

### 7.1 统一数据接口

```python
from datasets import get_dataloader

train_loader = get_dataloader(
    name="tapvid_kubric",
    root="datasets/tapvid_kubric",
    batch_size=8,
    split="train",
    num_frames=24,
    num_points=256,
    resolution=(256, 256),
    augmentation=config.data.augmentation,
)

val_loader = get_dataloader(
    name="tapvid_davis",
    root="datasets/tapvid_davis",
    resolution=None,
    augmentation=config.data.val.augmentation,
)
```

---

## 八、使用示例

### 8.1 完整训练流程

```bash
# 使用统一配置进行训练/评估
python train.py --config configs/fspt_base.yaml
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis --config configs/fspt_base.yaml
```
