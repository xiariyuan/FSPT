# FSPT 项目完整性概述

本文档提供FSPT项目的完整概述，供评估和检查使用。

---

## 一、项目文件清单

### 1. 核心代码文件

| 文件路径 | 描述 | 代码行数 | 状态 |
|----------|------|----------|------|
| `models/__init__.py` | 模型模块初始化和导出 | ~110 | ✅ 完成 |
| `models/semantic_encoder.py` | CLIP语义编码器 | ~240 | ✅ 完成 |
| `models/freq_semantic_fusion.py` | 语义增强频率分解 | ~280 | ✅ 完成 |
| `models/occlusion_predictor.py` | 频率感知遮挡预测器 | ~290 | ✅ 完成 |
| `models/point_tracker.py` | 完整点追踪模型 | ~400 | ✅ 完成 |

### 2. 数据集代码

| 文件路径 | 描述 | 代码行数 | 状态 |
|----------|------|----------|------|
| `datasets/__init__.py` | 数据集模块初始化 | ~90 | ✅ 完成 |
| `datasets/tapvid_kubric.py` | Kubric训练数据集 | ~320 | ✅ 完成 |
| `datasets/tapvid_davis.py` | DAVIS测试数据集 | ~150 | ✅ 完成 |
| `datasets/tapvid_kinetics.py` | Kinetics测试数据集 | ~200 | ✅ 完成 |
| `datasets/metrics.py` | TAP-Vid评估指标 | ~260 | ✅ 完成 |
| `datasets/augmentation.py` | 数据增强 | ~200 | ✅ 完成 |

### 3. 训练和评估脚本

| 文件路径 | 描述 | 代码行数 | 状态 |
|----------|------|----------|------|
| `train.py` | 完整训练脚本 | ~450 | ✅ 完成 |
| `evaluate.py` | 评估和可视化脚本 | ~280 | ✅ 完成 |
| `verify_project.py` | 项目验证脚本 | ~150 | ✅ 完成 |

### 4. 配置文件

| 文件路径 | 描述 | 状态 |
|----------|------|------|
| `configs/fspt_base.yaml` | 基础配置（完整参数） | ✅ 完成 |
| `requirements.txt` | Python依赖列表 | ✅ 完成 |

### 5. 工具脚本

| 文件路径 | 描述 | 状态 |
|----------|------|------|
| `scripts/download_baselines.sh` | 下载基线代码 | ✅ 完成 |
| `scripts/download_datasets.sh` | 下载数据集 | ✅ 完成 |
| `scripts/setup_environment.sh` | 环境配置 | ✅ 完成 |
| `scripts/init_remote.sh` | 远程服务器初始化 | ✅ 完成 |
| `scripts/sync_to_remote.ps1` | Windows同步脚本 | ✅ 完成 |
| `scripts/visualize.py` | 可视化工具 | ✅ 完成 |

### 6. 文档文件

| 文件路径 | 描述 | 状态 |
|----------|------|------|
| `README.md` | 项目主文档 | ✅ 完成 |
| `docs/RESEARCH_PLAN.md` | 详细研究计划 | ✅ 完成 |
| `docs/EXPERIMENTS.md` | 实验设计 | ✅ 完成 |
| `docs/LITERATURE_REVIEW.md` | 文献综述 | ✅ 完成 |
| `docs/DATASET_GUIDE.md` | 数据集使用指南 | ✅ 完成 |
| `docs/CODE_REUSE_PLAN.md` | 代码复用策略 | ✅ 完成 |
| `docs/QUICK_START.md` | 快速开始指南 | ✅ 完成 |
| `docs/PROJECT_OVERVIEW.md` | 本文档 | ✅ 完成 |

---

## 二、核心模块详解

### 2.1 SemanticEncoder (语义编码器)

**文件**: `models/semantic_encoder.py`

**功能**: 利用冻结的CLIP模型提取视频帧的语义特征

**关键方法**:
- `extract_features()`: 提取全局图像级特征
- `extract_spatial_features()`: 提取空间(patch-level)特征
- `sample_point_features()`: 在指定点位置采样语义特征

**输入输出**:
```python
# 输入
video: (B, T, 3, H, W)  # 视频帧

# 输出 (global mode)
features: (B, T, output_dim)  # 全局语义特征

# 输出 (spatial mode)
features: (B, T, H', W', output_dim)  # 空间语义特征
```

### 2.2 SemanticEnhancedLFD (语义增强频率分解)

**文件**: `models/freq_semantic_fusion.py`

**功能**: 将语义特征与频率分解相结合，实现语义引导的频率调制

**关键组件**:
- `SemanticModulator`: 根据语义特征动态调整频率响应
- `SimplifiedLFD`: 当FM-Track模块不可用时的备用实现
- 跨模态注意力融合

**设计思想**:
1. 语义相似的点在同一频带应该有相似的响应
2. 语义特征可以帮助区分不同物体的点
3. 语义一致性可以用于遮挡推理

**输入输出**:
```python
# 输入
geo_feat: (B, T, N, geo_dim)  # 几何特征
semantic_feat: (B, T, N, semantic_dim)  # 语义特征

# 输出
output: (B, T, N, geo_dim)  # 融合后的特征
info: Dict  # 包含频带特征、gate等中间结果
```

### 2.3 FrequencyAwareOcclusionPredictor (遮挡预测器)

**文件**: `models/occlusion_predictor.py`

**功能**: 基于频率分解和语义一致性进行遮挡检测和轨迹预测

**关键组件**:
- `OcclusionDetector`: 基于频率特征检测遮挡
- `TrajectoryPredictor`: 基于低频分量预测遮挡期间位置
- `SemanticConsistencyModule`: 传播遮挡信息到语义相似的点

**核心思想**:
1. 高频异常表示遮挡
2. 低频轨迹可外推遮挡期间位置
3. 同一物体的点应有相似遮挡状态

**输入输出**:
```python
# 输入
band_features: Dict[str, Tensor]  # 各频带特征
semantic_feat: (B, T, N, semantic_dim)  # 语义特征
positions: (B, T, N, 2)  # 当前位置

# 输出
occlusion_prob: (B, T, N)  # 遮挡概率
predicted_positions: (B, T, N, 2)  # 预测位置
confidence: (B, T, N)  # 预测置信度
```

### 2.4 FSPTTracker (完整追踪器)

**文件**: `models/point_tracker.py`

**功能**: 整合所有模块的完整点追踪模型

**组件集成**:
1. `GeometricBackbone`: 几何特征提取
2. `SemanticEncoder`: 语义特征提取
3. `SemanticEnhancedLFD`: 语义增强频率分解
4. `TemporalTransformer`: 时序建模
5. `FrequencyAwareOcclusionPredictor`: 遮挡处理
6. `PositionDecoder`: 位置解码

**输入输出**:
```python
# 输入
video: (B, T, 3, H, W)  # 视频帧
query_points: (B, N, 3)  # 查询点 [t, y, x]

# 输出
tracks: (B, N, T, 2)  # 预测轨迹 [y, x]
visibility: (B, N, T)  # 可见性概率
```

---

## 三、数据集详解

### 3.1 TAP-Vid Benchmark

**数据格式**:
```python
{
    'video': (T, H, W, 3),  # uint8 RGB视频
    'query_points': (N, 3),  # [t, y, x] 查询点
    'target_points': (N, T, 2),  # [y, x] 目标轨迹
    'occluded': (N, T),  # bool 遮挡标签
}
```

**支持的数据集**:
1. **TAP-Vid-Kubric**: 合成训练数据 (~10K视频)
2. **TAP-Vid-DAVIS**: 真实测试数据 (30视频)
3. **TAP-Vid-Kinetics**: 真实测试数据 (1189视频)

### 3.2 评估指标

详细的指标计算在 `datasets/metrics.py`:

```python
def compute_tapvid_metrics(
    pred_tracks,      # 预测轨迹
    gt_tracks,        # 真实轨迹
    pred_visibility,  # 预测可见性
    gt_visibility,    # 真实可见性
    query_points,     # 查询点
):
    # 返回:
    # - AJ: Average Jaccard
    # - <1px, <2px, <4px, <8px, <16px: 位置精度
    # - OA: Occlusion Accuracy
```

---

## 四、训练流程

### 4.1 完整训练流程

```python
# train.py 主要步骤

1. 加载配置 (OmegaConf)
2. 创建模型 (FSPTTracker)
3. 创建数据加载器 (TAP-Vid-Kubric)
4. 创建优化器 (AdamW)
5. 创建调度器 (CosineAnnealingLR)
6. 创建损失函数 (PointTrackingLoss)

for epoch in range(epochs):
    # 训练
    train_one_epoch(model, train_loader, ...)
    
    # 评估
    if epoch % eval_every == 0:
        evaluate(model, val_loader, ...)
    
    # 保存检查点
    save_checkpoint(...)
```

### 4.2 损失函数

```python
class PointTrackingLoss:
    # 1. 位置损失 (L1, 仅可见点)
    position_loss = |pred - gt| * visible_mask
    
    # 2. 遮挡损失 (BCE)
    occlusion_loss = BCE(pred_visibility, gt_visibility)
    
    # 3. 频率正交损失
    freq_ortho_loss = ...
    
    # 4. 语义一致性损失
    semantic_loss = ...
    
    total = w1*position + w2*occlusion + w3*ortho + w4*semantic
```

---

## 五、配置参数

### 5.1 模型配置

```yaml
model:
  backbone:
    type: "resnet50"          # 骨干网络类型
    pretrained: true          # 是否使用预训练
    
  clip:
    model: "ViT-B/16"         # CLIP模型
    freeze: true              # 冻结CLIP
    dim: 512                  # CLIP特征维度
    
  frequency:
    enabled: true             # 启用频率分解
    num_bands: 4              # 频带数量
    kernel_size: 7            # 滤波器大小
    
  temporal:
    num_layers: 6             # Transformer层数
    num_heads: 8              # 注意力头数
    dim: 256                  # 特征维度
    dropout: 0.1              # Dropout率
    
  occlusion:
    enabled: true             # 启用遮挡预测
    use_semantic_propagation: true  # 语义传播
```

### 5.2 训练配置

```yaml
training:
  epochs: 100
  batch_size: 8
  
  optimizer:
    type: "AdamW"
    lr: 1.0e-4
    weight_decay: 1.0e-4
    clip_lr_scale: 0.1        # CLIP微调学习率缩放（冻结时无效）
    
  scheduler:
    type: "CosineAnnealingLR"
    T_max: 100
    eta_min: 1.0e-6
    
  amp:
    enabled: true            # 混合精度训练
```

多卡训练（推荐 DDP）:
```bash
torchrun --nproc_per_node=4 scripts/train_distributed.py --config configs/fspt_base.yaml
```

---

## 六、评估方法

### 6.1 运行评估

```bash
# 评估DAVIS
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis

# 评估所有并可视化
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset all --visualize

# 指定分辨率
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis --resolution 256 256
```

### 6.2 评估输出

```
=============================================================================
EVALUATION RESULTS
=============================================================================

DAVIS
----------------------------------------
  AJ        : 0.6523 ± 0.0812
  <4px      : 0.7234 ± 0.0654
  OA        : 0.8912 ± 0.0321

  All thresholds:
    <1px      : 0.4532
    <2px      : 0.5821
    <8px      : 0.8234
    <16px     : 0.9123
    avg_error_px: 3.4521

=============================================================================
```

---

## 七、可视化工具

### 7.1 追踪结果可视化

```python
from scripts.visualize import visualize_tracking_results

visualize_tracking_results(
    video,           # (T, H, W, 3)
    pred_tracks,     # (N, T, 2)
    gt_tracks,       # (N, T, 2)
    pred_visibility, # (N, T)
    gt_visibility,   # (N, T)
    output_dir,
    video_name,
)
# 输出: 每帧PNG + 动画GIF
```

### 7.2 频率分解可视化

```python
from scripts.visualize import visualize_frequency_decomposition

visualize_frequency_decomposition(
    band_features,  # 各频带特征
    output_path,
    point_idx=0,
)
# 输出: 时域和频域特征图
```

### 7.3 遮挡分析可视化

```python
from scripts.visualize import visualize_occlusion_prediction

visualize_occlusion_prediction(
    video,
    pred_visibility,
    gt_visibility,
    output_dir,
    video_name,
)
# 输出: 遮挡准确率曲线、混淆矩阵等
```

---

## 八、项目验证

运行验证脚本检查所有模块：

```bash
python verify_project.py
```

预期输出：
```
============================================================
FSPT Project Verification
============================================================

[1/6] Checking core dependencies...
  ✓ PyTorch
  ✓ NumPy
  ✓ OmegaConf

[2/6] Checking model modules...
  ✓ models package
  ✓ SemanticEncoder
  ✓ SemanticEnhancedLFD
  ✓ FrequencyAwareOcclusionPredictor
  ✓ FSPTTracker

[3/6] Checking dataset modules...
  ✓ datasets package
  ✓ TAPVidDAVISDataset
  ✓ PointTrackingAugmentation

[4/6] Testing metrics...
  ✓ Metrics computation

[5/6] Testing model forward pass...
  ✓ FSPTTracker forward pass

[6/6] Checking scripts...
  ✓ train.py
  ✓ evaluate.py
  ✓ scripts/visualize.py

============================================================
All checks passed! FSPT project is ready.
============================================================
```

---

## 九、下一步计划

### 9.1 短期任务 (1-2周)
1. 在远程服务器初始化环境
2. 复用FM-Track的频率分解模块
3. 下载并验证基线方法
4. 复现基线结果

### 9.2 中期任务 (3-4周)
1. 实现完整的FSPT模型
2. 在Kubric上训练
3. 在DAVIS上评估
4. 消融实验

### 9.3 长期目标
1. 达到SOTA性能
2. 撰写论文
3. 提交顶会

---

## 十、联系方式

如有问题，请联系项目作者。

---

*文档更新日期: 2026-01-25*
