# FSPT: Frequency-Semantic Point Tracking

**频率-语义联合点追踪框架**

> 首个将可学习频率分解与视觉语言模型语义理解相结合的点追踪方法

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 项目概述

本项目旨在解决当前点追踪(Point Tracking / Tracking Any Point)领域的核心挑战：

| 问题 | 现有方法的局限 | 我们的解决方案 |
|------|---------------|--------------|
| 遮挡处理能力不足 | 依赖可见性插值或简单预测 | 频率感知遮挡推理 |
| 缺乏长程时序一致性 | 逐帧匹配导致漂移 | 低频轨迹建模保持一致性 |
| 缺失语义理解能力 | 纯几何特征匹配 | CLIP语义特征增强 |
| 多尺度运动处理不灵活 | 固定窗口或尺度 | 可学习频率分解自适应 |
| 合成-真实域差距 | 域特定训练 | 语义一致性伪标签跨域训练 |

## 核心创新

| 创新点 | 描述 |
|--------|------|
| 频率自适应点追踪 | 将点轨迹按运动频率分解，低频做长程关联，高频做精确定位 |
| 语义增强点匹配 | 利用CLIP语义特征引导点匹配 |
| 频率感知遮挡推理 | 用低频轨迹预测遮挡期间的点位置 |
| 跨域自适应训练 | 利用语义一致性做伪标签 |
| 多尺度几何特征融合 | FPN风格融合多尺度几何特征（可选） |
| 迭代精化追踪 | 多次位置更新提升细节一致性（可选） |

## 目录结构

```
FSPT-FrequencySemanticPointTracking/
├── docs/                    # 文档
│   ├── RESEARCH_PLAN.md     # 研究计划
│   ├── EXPERIMENTS.md       # 实验设计
│   └── LITERATURE_REVIEW.md # 文献综述
├── models/                  # 模型代码
│   ├── __init__.py
│   ├── semantic_encoder.py  # 语义编码器(CLIP)
│   ├── freq_semantic_fusion.py # 频率-语义融合
│   ├── occlusion_predictor.py  # 遮挡预测器
│   └── point_tracker.py     # 主追踪器
├── datasets/                # 数据集处理
│   ├── tapvid_kubric.py
│   ├── tapvid_davis.py
│   ├── tapvid_kinetics.py
│   ├── metrics.py
│   └── augmentation.py
├── configs/                 # 配置文件
├── scripts/                 # 工具脚本
│   ├── download_baselines.sh
│   ├── download_datasets.sh
│   ├── setup_environment.sh
│   └── visualize.py
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── verify_project.py        # 项目验证
└── requirements.txt         # 依赖列表
```

## 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n fspt python=3.10
conda activate fspt

# 安装PyTorch (CUDA 12.1)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git
```

### 2. 下载基线代码

```bash
bash scripts/download_baselines.sh
```

### 3. 下载数据集

```bash
# 下载TAP-Vid基准数据集
bash scripts/download_datasets.sh
# 可选：下载Kubric TFDS调试数据（不含点标注）
# DOWNLOAD_KUBRIC_TFDS=1 bash scripts/download_datasets.sh

# 如果没有bash环境（如Windows），可使用Python脚本：
# python scripts/download_all_datasets.py --root /gemini/code/datasets
# （可选，调试用）下载Kubric TFDS数据（不含点标注）：
# python scripts/download_all_datasets.py --root /gemini/code/datasets --kubric-tfds

# 或者手动下载
# DAVIS: https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl
# Kinetics: https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip
```

**Kubric 训练数据说明**
- 推荐使用 TAP-Vid 官方 Kubric 标注 `tapvid_kubric_train.pkl` 放置在 `/gemini/code/datasets/tapvid_kubric/`
- 若文件名不同，请在 `configs/fspt_base.yaml` 中设置 `data.train.annotation_file`
- `data.train.use_tfds` 仅用于调试（需要 `allow_synthetic_tracks=true`）

### 4. 验证安装

```bash
# 验证所有模块是否正常
python verify_project.py
```

### 5. 训练模型

```bash
# 使用基础配置训练
python train.py --config configs/fspt_base.yaml

# 从检查点恢复训练
python train.py --config configs/fspt_base.yaml --resume checkpoints/fspt_base/latest.pth

# Debug模式（减少数据量）
python train.py --config configs/fspt_base.yaml --debug

# 多卡训练（DDP，推荐脚本）
torchrun --nproc_per_node=2 scripts/train_distributed.py --config configs/fspt_base.yaml

# 如需直接用 train.py，需要启用 distributed 配置：
# --hardware.distributed.enabled true
torchrun --nproc_per_node=2 train.py --config configs/fspt_base.yaml --hardware.distributed.enabled true
```

### 5.1 Route A（推荐）：CoTracker3 作为 base + FSPT 作为 refinement

该路线用于顶会实验闭环：先用强基线（CoTracker3）产生粗轨迹，再用本项目的频率/语义/遮挡模块做残差精化。

**云服务器准备**
```bash
# 1) 下载并安装 CoTracker（以及权重）
bash scripts/download_baselines.sh

# 2) 下载数据集
bash scripts/download_datasets.sh

# (Optional) Precompute CoTracker base tracks for faster evaluation/validation
# This is most useful for DAVIS/Kinetics where query_points are deterministic.
python scripts/precompute_base_tracks.py \
  --dataset davis \
  --root /gemini/code/datasets/tapvid_davis \
  --checkpoint baselines/cotracker/checkpoints/scaled_offline.pth \
  --output-dir outputs/base_tracks

# (Optional) Kubric: enable deterministic sampling when you need cached query_points to match.
# Note: if you enable `data.*.base_tracks_dir` together with `data.augmentation`, the dataloader
# will disable dataset-level augmentation and re-apply it *after* cache injection so cached
# `base_tracks/base_visibility` stay consistent (safe for training too).
# For Kubric training, cached query_points must also match: set `data.train.deterministic_sampling=true`.
# python scripts/precompute_base_tracks.py \
#   --dataset kubric --root /gemini/code/datasets/tapvid_kubric --checkpoint baselines/cotracker/checkpoints/scaled_offline.pth \
#   --output-dir outputs/base_tracks --deterministic --deterministic-seed 42 --num-frames 24 --num-points 256
# If you enable occlusion-balanced point sampling in training (data.train.sampling.strategy=occlusion_balanced),
# make sure to match it here too:
#   --sampling-strategy occlusion_balanced --sampling-hard-fraction 0.5

# (Optional) Precompute / warm up CLIP semantic cache for repeated evaluation
# This writes into model.semantic.cache.root_dir (or --cache-dir override) and speeds up repeated eval runs.
python scripts/precompute_semantic_cache.py --config configs/fspt_cotracker_refine.yaml --dataset all
# Or use a checkpoint's embedded config:
# python scripts/precompute_semantic_cache.py --checkpoint checkpoints/fspt_cotracker_refine/best.pth --dataset all
```

**训练（Hybrid Refiner）**
```bash
python train.py --config configs/fspt_cotracker_refine.yaml

# DDP
torchrun --nproc_per_node=8 scripts/train_distributed.py --config configs/fspt_cotracker_refine.yaml

# (Optional) Stage schedule logs (Route A refiner) are saved under:
#   outputs/<experiment_name>/refiner_stage_log.jsonl
#   outputs/<experiment_name>/refiner_stage_transitions.jsonl
# Per-epoch train/val metrics are also appended to:
#   outputs/<experiment_name>/epoch_metrics.jsonl
# You can summarize best epochs (overall / per stage) with:
#   python scripts/summarize_epoch_metrics.py --exp-dir outputs/<experiment_name> --metric AJ --mode max \
#     --save-csv stage_best.csv --save-md stage_best.md
#
 # (Optional) Fuse base tracker visibility as a prior (Route A ablation):
 #   Edit configs/fspt_cotracker_refine.yaml:
 #     model.refiner.visibility_prior.mode: blend
 #     model.refiner.visibility_prior.alpha: 0.2
 #
 # (Optional) Distill base tracker tracks (stabilize early training / ablation):
 #   Edit configs/fspt_base.yaml (loss.base_track_consistency.*):
 #     loss.base_track_consistency.weight: 0.05
 #     loss.base_track_consistency.type: l2
 #     loss.base_track_consistency.mask: gt_visible
 #
 # (Optional) Distillation weight schedule (ramp up):
 #   loss.base_track_consistency.schedule.type: linear
 #   loss.base_track_consistency.schedule.start_epoch: 0
 #   loss.base_track_consistency.schedule.end_epoch: 10
 #   loss.base_track_consistency.schedule.start_weight: 0.0
 #   loss.base_track_consistency.schedule.end_weight: 0.05
 #
 # (Optional) Prior-feature injection into the refiner tokens (Route A):
 #   model.refiner.prior_features.enabled: true
 #   model.refiner.prior_features.use_base_visibility: true
 #   model.refiner.prior_features.use_delta: true
 #
 # (Optional) Two-view spatial consistency regularization (extra compute):
 #   loss.two_view_consistency.enabled: true
 #   loss.two_view_consistency.weight: 0.02
 #   loss.two_view_consistency.teacher: ema
 #   # Optional: skip two-view loss on CUDA OOM instead of crashing:
 #   loss.two_view_consistency.oom_safe: true
 #   # Optional: run EMA teacher forward without AMP (fp32 targets; slightly slower but stabler):
 #   loss.two_view_consistency.teacher_disable_autocast: true
 #   # Optional: reduce compute by subsampling points for the two-view loss:
 #   loss.two_view_consistency.max_points: 512
 #   # Optional: also match visibility predictions (occlusion head):
 #   loss.two_view_consistency.visibility_weight: 0.01
 #   loss.two_view_consistency.visibility_type: mse
 #   # Optional: soft weighting (confidence) instead of hard masking:
 #   loss.two_view_consistency.mask_weighting: soft
 #   # Optional: visibility weight schedule:
 #   loss.two_view_consistency.visibility_schedule.type: linear
 #   loss.two_view_consistency.visibility_schedule.start_epoch: 0
 #   loss.two_view_consistency.visibility_schedule.end_epoch: 10
 #   loss.two_view_consistency.visibility_schedule.start_weight: 0.0
 #   loss.two_view_consistency.visibility_schedule.end_weight: 0.01
 #   # Optional: schedule (ramp up):
 #   loss.two_view_consistency.schedule.type: linear
 #   # Use finer-grained step-based schedule (global batch steps) if desired:
 #   loss.two_view_consistency.schedule.unit: step  # epoch | step | optimizer_step
 #   loss.two_view_consistency.schedule.start_step: 0
 #   loss.two_view_consistency.schedule.end_step: 5000
 #   loss.two_view_consistency.schedule.start_epoch: 0
 #   loss.two_view_consistency.schedule.end_epoch: 10
 #   loss.two_view_consistency.schedule.start_weight: 0.0
 #   loss.two_view_consistency.schedule.end_weight: 0.02
 #   # Optional: invertible temporal views (equivariance; extra compute).
 #   # Uses only reverse/roll (permutation) so predictions can be mapped back exactly.
 #   # Requires cached base_tracks/base_visibility to keep query_points consistent after time transforms.
 #   loss.two_view_consistency.temporal.enabled: true
 #   loss.two_view_consistency.temporal.reverse_prob: 0.5
 #   loss.two_view_consistency.temporal.roll_prob: 0.5
 #   loss.two_view_consistency.temporal.roll_max: null  # null -> sample full range [0, T-1]
 #   # When temporal two-view is enabled, the training script can auto-disable dataset-level temporal augmentation:
 #   loss.two_view_consistency.auto_disable_dataset_temporal: true
 ```

### 6. 评估模型

```bash
# 评估单个数据集
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis

# 评估所有数据集并生成可视化
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset all --visualize

# 使用配置中的验证分辨率/增强
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis --config configs/fspt_base.yaml

# 如需包含查询帧参与指标计算
# 默认会排除查询帧（符合 TAP-Vid 官方评测）
python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis --include-query-frame

# Route A: 若模型返回 base_tracks/base_visibility，可同时汇报 base 与 refined 指标
python evaluate.py --checkpoint checkpoints/fspt_cotracker_refine/best.pth --dataset davis --compare-base

# (Optional) Use precomputed base tracks to skip running CoTracker during evaluation
python evaluate.py --checkpoint checkpoints/fspt_cotracker_refine/best.pth --dataset davis --compare-base \
  --base-tracks-dir outputs/base_tracks --base-tracks-strict
# If your cache was generated with slightly different query_points, relax the tolerance:
#   --base-tracks-query-tol 1e-3

# Save paper-friendly tables
python evaluate.py --checkpoint checkpoints/fspt_cotracker_refine/best.pth --dataset all --compare-base \
  --save-csv summary.csv --save-md summary.md --save-tex summary.tex
```

## 模型架构

```
Input Video + Query Points
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    GeometricBackbone                     │
│                  (ResNet-50 / ResNet-18)                 │
└─────────────────────────────────────────────────────────┘
         │
         ├────────────────────────┐
         ▼                        ▼
┌─────────────────┐     ┌─────────────────┐
│ Feature Sampling│     │ SemanticEncoder │
│ at Query Points │     │   (Frozen CLIP) │
└─────────────────┘     └─────────────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
┌─────────────────────────────────────────────────────────┐
│              SemanticEnhancedLFD                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Learnable Frequency Decomposition (from FM-Track) │  │
│  │   → Band 0 (低频): 全局运动/长程关联               │  │
│  │   → Band 1-2 (中频): 局部运动                     │  │
│  │   → Band 3 (高频): 精确定位                       │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Semantic Modulation                        │  │
│  │   CLIP语义特征调制频率响应                         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              TemporalTransformer                         │
│              (6层 Transformer Encoder)                   │
└─────────────────────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌─────────────────┐   ┌─────────────────────────────┐
│ PositionDecoder │   │ FrequencyAwareOcclusion     │
│  → Δ(y, x)      │   │ Predictor                   │
│  → visibility   │   │  → 高频异常检测              │
└─────────────────┘   │  → 低频轨迹外推              │
         │            │  → 语义一致性传播            │
         │            └─────────────────────────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
           Output: Tracks + Visibility
```

## 评估指标

本项目使用TAP-Vid官方评估指标：

| 指标 | 描述 | 计算方式 |
|------|------|----------|
| **AJ (Average Jaccard)** | 综合评估位置精度和可见性预测 | Jaccard = TP / (TP + FP + FN) |
| **< δ^x_avg** | 各阈值下的位置精度 | 位置误差 < x像素的比例 |
| **OA (Occlusion Accuracy)** | 遮挡预测准确率 | 正确预测遮挡/可见的比例 |

### 预期性能目标

| 方法 | TAP-Vid-DAVIS AJ | TAP-Vid-Kinetics AJ |
|------|-----------------|---------------------|
| TAPIR (baseline) | 61.3 | 49.6 |
| CoTracker3 (SOTA) | 64.8 | 52.1 |
| **FSPT (ours)** | **67.0+** | **54.0+** |

## 数据集

### 训练数据

| 数据集 | 类型 | 规模 | 用途 |
|--------|------|------|------|
| TAP-Vid-Kubric | 合成 | ~10K视频 | 预训练 |
| PointOdyssey | 合成 | 长视频 | 长程追踪训练 |

### 测试数据

| 数据集 | 类型 | 规模 | 特点 |
|--------|------|------|------|
| TAP-Vid-DAVIS | 真实 | 30视频 | 高质量标注 |
| TAP-Vid-Kinetics | 真实 | 1189视频 | 多样场景 |

详细数据集使用指南请参考: [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md)

## 项目文档

| 文档 | 描述 |
|------|------|
| [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | 详细研究计划和时间线 |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验设计和消融实验 |
| [LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) | 文献综述和论文列表 |
| [DATASET_GUIDE.md](docs/DATASET_GUIDE.md) | 数据集下载和使用指南 |
| [CODE_REUSE_PLAN.md](docs/CODE_REUSE_PLAN.md) | 代码复用策略 |
| [QUICK_START.md](docs/QUICK_START.md) | 快速开始指南 |

## 目标会议

- **ECCV 2026** (Deadline: 2026年3月初) - 首选
- **CVPR 2026** (Deadline: 2025年11月中旬) - 备选

## 相关项目

本项目基于以下现有工作构建：

| 项目 | 复用内容 | 来源 |
|------|----------|------|
| [FM-Track](../FMtrack-main/FM-Track) | 可学习频率分解、频率感知时序建模 | 频率感知多目标追踪 |
| [TACO](../taco) | CLIP集成、对比学习框架 | 开放词汇多目标追踪 |

## 参考论文

### 点追踪核心论文
1. **TAPIR** (ICCV 2023) - 基础两阶段追踪框架
2. **CoTracker** (ECCV 2024) - 联合多点追踪
3. **LocoTrack** (ECCV 2024) - 高效4D局部相关性
4. **CoTracker3** (ICCV 2025) - 伪标签自监督
5. **TAPNext** (arXiv 2025) - 下一代架构

### 频率域方法
1. **FD4MM** (CVPR 2024) - 频率解耦运动放大
2. **Phase-based Video Motion Processing** - 相位域运动处理

### 视觉语言模型
1. **SAM-PT** (WACV 2025) - SAM+点追踪
2. **OVTrack** (CVPR 2023) - 开放词汇追踪
3. **Track-On** (arXiv 2024) - VLM驱动点追踪

## 代码结构

```
FSPT-FrequencySemanticPointTracking/
├── models/                      # 模型实现
│   ├── __init__.py              # 模块导出
│   ├── semantic_encoder.py      # CLIP语义编码器
│   ├── freq_semantic_fusion.py  # 语义增强频率分解
│   ├── occlusion_predictor.py   # 频率感知遮挡预测
│   └── point_tracker.py         # 完整追踪模型
├── datasets/                    # 数据集
│   ├── __init__.py
│   ├── tapvid_kubric.py         # Kubric训练数据
│   ├── tapvid_davis.py          # DAVIS测试数据
│   ├── tapvid_kinetics.py       # Kinetics测试数据
│   ├── metrics.py               # 评估指标
│   └── augmentation.py          # 数据增强
├── configs/                     # 配置文件
│   └── fspt_base.yaml           # 基础配置
├── scripts/                     # 工具脚本
│   ├── download_baselines.sh
│   ├── download_datasets.sh
│   ├── setup_environment.sh
│   └── visualize.py             # 可视化工具
├── docs/                        # 详细文档
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
├── verify_project.py            # 项目验证
└── requirements.txt             # 依赖列表
```

## 作者

- **李源** - 研一，计算机视觉方向

## License

MIT License

## 致谢

感谢以下开源项目：
- [TAPIR](https://github.com/deepmind/tapnet)
- [CoTracker](https://github.com/facebookresearch/co-tracker)
- [CLIP](https://github.com/openai/CLIP)
- [timm](https://github.com/huggingface/pytorch-image-models)
