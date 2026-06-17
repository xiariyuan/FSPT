# FSPT 快速启动指南

## 🚀 第一步：在远程服务器上初始化项目

### 1.1 SSH连接到服务器

```bash
# 连接到你的服务器（示例）
ssh your-server
```

### 1.2 创建项目目录

```bash
# 创建FSPT项目目录
PROJECT_DIR="${PROJECT_DIR:-$HOME/FSPT}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 创建目录结构
mkdir -p docs papers baselines/{cotracker,locotrack,tapir,pips,alltracker} \
         models datasets configs scripts experiments logs outputs checkpoints assets
```

### 1.3 上传项目文件

从本地上传规划文件：
```bash
# 在本地Windows执行
scp -r "<本地项目路径>/FSPT-FrequencySemanticPointTracking/*" your-server:/path/to/FSPT/
```

---

## 🔧 第二步：环境配置

### 2.1 创建Conda环境

```bash
# 在服务器上执行
cd "$PROJECT_DIR"

# 创建环境
conda create -n fspt python=3.10 -y
conda activate fspt

# 安装PyTorch (CUDA 12.1)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
```

### 2.2 验证环境

```bash
python verify_project.py
```

---

## 📦 第三步：下载基线代码

### 3.1 下载CoTracker3 (当前SOTA)

```bash
cd "$PROJECT_DIR/baselines"
git clone https://github.com/facebookresearch/co-tracker.git cotracker
cd cotracker
pip install -e .

# 下载预训练权重
mkdir -p checkpoints && cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

### 3.2 下载LocoTrack (效率SOTA)

```bash
cd "$PROJECT_DIR/baselines"
git clone https://github.com/cvlab-kaist/locotrack.git locotrack
cd locotrack
pip install -e .
```

### 3.3 下载TAPIR

```bash
cd "$PROJECT_DIR/baselines"
git clone https://github.com/deepmind/tapnet.git tapir
cd tapir
pip install -r requirements.txt
```

---

## 📊 第四步：下载数据集

### 4.0 一键下载（推荐，跨平台）

```bash
cd "$PROJECT_DIR"
python scripts/download_all_datasets.py --root datasets
# 可选：下载Kubric TFDS调试数据（不含点标注）
# python scripts/download_all_datasets.py --root datasets --kubric-tfds
```

### 4.1 TAP-Vid DAVIS (必需)

```bash
cd "$PROJECT_DIR/datasets"
mkdir -p tapvid_davis && cd tapvid_davis

# 下载DAVIS视频
wget https://data.vision.ee.ethz.ch/cserber/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip && rm DAVIS-2017-trainval-480p.zip

# 下载TAP-Vid标注
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl
```

### 4.2 TAP-Vid Kinetics (推荐)

```bash
cd "$PROJECT_DIR/datasets"
mkdir -p tapvid_kinetics && cd tapvid_kinetics

# 下载标注
wget https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip
unzip tapvid_kinetics.zip && rm tapvid_kinetics.zip

# 注意：视频需要从YouTube单独下载
# pip install yt-dlp
```

---

## 🔄 第五步：复用现有代码

### 5.1 可选：从FM-Track复制模块

```bash
cd "$PROJECT_DIR"
FMTRACK_DIR="${FMTRACK_DIR:-/path/to/FMtrack-main/FM-Track}"

# 可选：如果没有FM-Track源码可跳过（代码会自动回退到SimplifiedLFD）
cp "$FMTRACK_DIR/models/motip/learnable_freq_decomposition.py" models/
cp "$FMTRACK_DIR/models/motip/freq_temporal_transformer.py" models/

# 复制工具函数
mkdir -p util
cp "$FMTRACK_DIR/util/box_ops.py" util/
cp "$FMTRACK_DIR/util/misc.py" util/
```

### 5.2 安装 CLIP 依赖（推荐）

```bash
# 任选其一
pip install git+https://github.com/openai/CLIP.git
# 或
pip install open-clip-torch
```

---

## ✅ 第六步：验证基线

### 6.1 运行CoTracker3演示

```bash
cd "$PROJECT_DIR/baselines/cotracker"

# 下载示例视频
python demo.py --video_path path/to/video.mp4

# 在DAVIS上评估
python scripts/evaluate_tap.py --checkpoint checkpoints/scaled_offline.pth --dataset davis
```

### 6.2 记录基线结果

在 `experiments/baseline_results.yaml` 中记录：
```yaml
cotracker3:
  tapvid_davis:
    AJ: 68.2
    "<4px": 85.1
    OA: 92.3
  date: "2026-01-26"
  notes: "复现成功"
```

---

## 📝 第七步：开始开发

### 7.1 创建初始模型

```python
# models/__init__.py（示例：本仓库内置模块）
from .freq_semantic_fusion import SimplifiedLFD

# 创建适配器
class FSPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lfd = SimplifiedLFD(
            dim=config.dim,
            num_bands=config.num_bands,
        )
        # ... 其他模块
```

### 7.2 运行第一个实验

```bash
cd "$PROJECT_DIR"

# 测试频率分解模块（内置SimplifiedLFD）
python -c "
import torch
from models.freq_semantic_fusion import SimplifiedLFD

lfd = SimplifiedLFD(dim=256, num_bands=4)
x = torch.randn(2, 1, 24, 100, 256)  # (B, G, T, N, C)
output, info = lfd(x)
print(f'Output shape: {output.shape}')
print(f'Bands: {list(info[\"band_features\"].keys())}')
print('Test passed!')
"
```

---

## 🧪 实验追踪与训练增强

建议：从 `configs/fspt_base.yaml` 起步，训练稳定后再逐个启用策略并做消融，避免“堆模块”。

### WandB 集成（可选）

在 `configs/fspt_base.yaml` 中启用：
```yaml
logging:
  wandb:
    enabled: true
    project: "FSPT"
    entity: null
    watch_model: false
    log_freq: 100
```

### EMA 与早停（可选）
```yaml
training:
  ema:
    enabled: true
    decay: 0.9999
    use_for_eval: true
  early_stopping:
    enabled: true
    monitor: "AJ"
    patience: 15
    mode: "max"
```

### 学习率查找器（自动调参）
```yaml
training:
  lr_finder:
    enabled: true
    start_lr: 1.0e-7
    end_lr: 1.0
    num_iter: 100
    apply: true
    plot_path: "outputs/lr_finder.png"
```
或运行脚本：
```bash
python scripts/find_lr.py --config configs/fspt_base.yaml
```

### 伪标签自训练（在线）
```yaml
data:
  pseudo:
    dataset: "tapvid_kubric"
    root: "/path/to/unlabeled_kubric"
    split: "train"
    num_frames: 24
    num_points: 256
    resolution: [256, 256]

training:
  pseudo_labeling:
    enabled: true
    teacher_momentum: 0.999
    confidence_threshold: 0.5
    warmup_epochs: 10
    use_soft_labels: true
```
注意：
- 需要提供 `data.pseudo` 才会启用伪标签训练
- 使用 `OneCycleLR` 或 `gradient.accumulation_steps > 1` 时会自动跳过伪标签步

### 渐进式训练（分辨率/帧数/点数）
```yaml
training:
  progressive:
    enabled: true
    smooth_transition: true
    transition_epochs: 2
    stages: null  # 留空则使用默认三阶段策略
```

### 置信度加权损失（可选）
```yaml
loss:
  confidence_weighted:
    enabled: true
    weight: 0.1
    min_confidence: 0.1
```

---

## 📅 下一步计划

1. **本周（Week 1）**
   - [ ] 完成环境配置
   - [ ] 下载并运行所有基线
   - [ ] 阅读核心论文 (TAPIR, CoTracker3, LocoTrack)

2. **下周（Week 2）**
   - [ ] 复用FM-Track模块并验证
   - [ ] 设计语义增强模块
   - [ ] 开始简单实验

3. **两周后（Week 3-4）**
   - [ ] 完成主模型开发
   - [ ] 开始训练实验

---

## ❓ 常见问题

### Q1: CUDA内存不足
```bash
# 减小batch size
python train.py --config configs/fspt_base.yaml --training.batch_size 4

# 使用梯度检查点
python train.py --config configs/fspt_base.yaml --model.temporal.use_gradient_checkpointing true
```

### Q2: 多卡训练如何启动
```bash
torchrun --nproc_per_node=2 scripts/train_distributed.py --config configs/fspt_base.yaml
```

### Q3: 下载速度慢
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q4: 找不到模块
```bash
# 确保在项目根目录
cd "$PROJECT_DIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## 📞 问题反馈

如遇到问题，请在 `logs/issues.md` 中记录，包括：
- 错误信息
- 复现步骤
- 尝试的解决方案
