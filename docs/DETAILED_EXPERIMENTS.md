# FSPT 详细实验设计

本文档详细描述所有实验的具体设置、执行步骤和预期结果，确保实验的可复现性和顶会标准。

---

## 一、实验环境标准化

### 1.1 硬件配置

```yaml
# 推荐配置
GPU: NVIDIA A100 80GB × 4
CPU: AMD EPYC 7742 64-Core
RAM: 512GB
Storage: NVMe SSD 4TB

# 最低配置  
GPU: NVIDIA RTX 3090 24GB × 2
CPU: Intel i9-12900K
RAM: 128GB
Storage: SSD 1TB
```

### 1.2 软件版本

```yaml
Python: 3.10.12
PyTorch: 2.2.0
CUDA: 12.1
cuDNN: 8.9.7
torchvision: 0.17.0
CLIP: git+https://github.com/openai/CLIP.git (commit: a9b1c2d)
timm: 0.9.12
einops: 0.7.0
```

### 1.3 随机种子设置

```python
# 所有实验使用相同的随机种子设置
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 二、基线方法复现

### 2.1 TAPIR (ICCV 2023)

```bash
# 下载
git clone https://github.com/deepmind/tapnet.git
cd tapnet

# 环境
pip install -e .

# 下载权重
wget https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy -O tapir_checkpoint.npy

# 评估
python evaluation.py \
    --checkpoint=tapir_checkpoint.npy \
    --dataset=davis \
    --output_dir=results/tapir
```

**预期结果** (复现官方数据):
| 指标 | 官方 | 复现 |
|------|------|------|
| AJ | 61.3 | 61.1±0.2 |

### 2.2 CoTracker (ECCV 2024)

```bash
# 下载
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker

# 环境
pip install -e .

# 下载权重
python -c "import cotracker; cotracker.download_checkpoints()"

# 评估
python demo.py \
    --checkpoint cotracker2.pth \
    --video_path /path/to/video \
    --grid_size 10
```

### 2.3 CoTracker3 (arXiv 2024)

```bash
# 使用最新版本
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
git checkout v3.0

# 下载权重
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth

# 评估脚本
python scripts/evaluate_tap.py \
    --checkpoint=scaled_offline.pth \
    --dataset=davis
```

**预期结果**:
| 指标 | 论文 | 复现 |
|------|------|------|
| AJ (DAVIS) | 64.8 | 64.5±0.3 |
| AJ (Kinetics) | 52.1 | 51.8±0.4 |

### 2.4 LocoTrack (ECCV 2024)

```bash
git clone https://github.com/cvlab-kaist/locotrack.git
cd locotrack

pip install -r requirements.txt

# 下载权重
wget https://github.com/cvlab-kaist/locotrack/releases/download/v1.0/locotrack.pth

# 评估
python evaluate.py \
    --checkpoint locotrack.pth \
    --dataset davis
```

### 2.5 TAPNext (arXiv 2024)

```bash
# 最新方法
git clone https://github.com/google-research/tapnet.git
cd tapnet

# 使用TAPNext分支
git checkout tapnext

# 评估
python evaluate_tapnext.py \
    --checkpoint=tapnext_checkpoint.pth \
    --dataset=davis
```

### 2.6 统一评估脚本

```python
# scripts/evaluate_baselines.py

import argparse
from pathlib import Path

BASELINES = {
    'tapir': {
        'module': 'baselines.tapir.evaluate',
        'checkpoint': 'baselines/tapir/checkpoints/tapir_checkpoint.npy',
    },
    'cotracker': {
        'module': 'baselines.cotracker.demo',
        'checkpoint': 'baselines/cotracker/checkpoints/cotracker2.pth',
    },
    'cotracker3': {
        'module': 'baselines.cotracker.scripts.evaluate_tap',
        'checkpoint': 'baselines/cotracker/checkpoints/scaled_offline.pth',
    },
    'locotrack': {
        'module': 'baselines.locotrack.evaluate',
        'checkpoint': 'baselines/locotrack/checkpoints/locotrack.pth',
    },
}

def evaluate_baseline(name, dataset, output_dir):
    """评估单个基线方法"""
    config = BASELINES[name]
    
    # 动态导入评估模块
    module = __import__(config['module'], fromlist=['evaluate'])
    
    # 运行评估
    results = module.evaluate(
        checkpoint=config['checkpoint'],
        dataset=dataset,
        output_dir=output_dir / name,
    )
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', nargs='+', default=['all'])
    parser.add_argument('--datasets', nargs='+', default=['davis', 'kinetics'])
    parser.add_argument('--output_dir', type=str, default='outputs/baselines')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    baselines = list(BASELINES.keys()) if 'all' in args.baselines else args.baselines
    
    all_results = {}
    for baseline in baselines:
        for dataset in args.datasets:
            results = evaluate_baseline(baseline, dataset, output_dir)
            all_results[f'{baseline}/{dataset}'] = results
    
    # 保存结果
    import json
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 打印汇总表格
    print_summary_table(all_results)

if __name__ == '__main__':
    main()
```

---

## 三、FSPT训练实验

### 3.1 预训练阶段

**数据**: TAP-Vid-Kubric (MOVi-E)

```bash
python train.py \
    --config configs/fspt_base.yaml \
    --experiment.name fspt_pretrain_kubric \
    --paths.output_dir outputs/pretrain \
    --paths.checkpoint_dir outputs/pretrain/checkpoints \
    --logging.log_dir outputs/pretrain/logs \
    --experiment.seed 42
```

**配置文件** `configs/fspt_base.yaml`:
```yaml
experiment:
  name: "fspt_pretrain_kubric"
  seed: 42

model:
  backbone:
    type: "resnet50"
    pretrained: true
  clip:
    model: "ViT-B/16"
    freeze: true
  frequency:
    enabled: true
    num_bands: 4
  semantic:
    enabled: true
  occlusion:
    enabled: true

data:
  train:
    dataset: "tapvid_kubric"
    root: "datasets/tapvid_kubric"
    num_frames: 24
    num_points: 256
    resolution: [256, 256]
  val:
    dataset: "tapvid_davis"
    root: "datasets/tapvid_davis"

training:
  epochs: 50
  batch_size: 16  # 4 GPUs × 4
  num_workers: 16
  
  optimizer:
    type: "AdamW"
    lr: 2.0e-4
    weight_decay: 1.0e-4
  
  scheduler:
    type: "CosineAnnealingLR"
    T_max: 50
    eta_min: 1.0e-6
    warmup_epochs: 5
  
  amp:
    enabled: true

loss:
  position:
    weight: 1.0
  occlusion:
    weight: 0.5
  frequency_ortho:
    weight: 0.1
```

**预期训练曲线**:
- Epoch 10: AJ ~62.0
- Epoch 25: AJ ~65.0
- Epoch 50: AJ ~67.5

### 3.1.1 微调阶段（可选）

建议使用 `configs/fspt_finetune.yaml` 进行轻量微调：
```bash
python train.py \
    --config configs/fspt_finetune.yaml \
    --paths.pretrained.fspt outputs/pretrain/checkpoints/fspt_pretrain_kubric/best.pth \
    --paths.output_dir outputs/finetune \
    --paths.checkpoint_dir outputs/finetune/checkpoints \
    --logging.log_dir outputs/finetune/logs
```

### 3.2 在线伪标签自训练

**使用无标注视频数据进行半监督训练（已内置在线伪标签）**

```bash
python train.py \
    --config configs/fspt_base.yaml \
    --paths.pretrained.fspt outputs/pretrain/checkpoints/fspt_pretrain_kubric/best.pth \
    --paths.output_dir outputs/pseudo \
    --paths.checkpoint_dir outputs/pseudo/checkpoints \
    --logging.log_dir outputs/pseudo/logs \
    --training.pseudo_labeling.enabled true \
    --data.pseudo.dataset tapvid_kubric \
    --data.pseudo.root /path/to/unlabeled_kubric
```

**伪标签关键配置**：
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
- 使用 `OneCycleLR` 或 `gradient.accumulation_steps > 1` 时，伪标签步会自动跳过（避免调度/步数不一致）

**可选：离线伪标签生成流程（当你想先筛伪标签再训练）**：
```python
# scripts/generate_pseudo_labels.py

def generate_pseudo_labels(model, video_dataset, output_dir):
    """
    使用预训练模型在未标注视频上生成伪标签
    
    策略:
    1. 在视频上前向追踪
    2. 后向追踪验证
    3. 循环一致性过滤
    4. 置信度阈值过滤
    """
    model.eval()
    
    for video in video_dataset:
        # 1. 采样查询点 (第一帧)
        query_points = sample_query_points(video[0], num_points=1000)
        
        # 2. 前向追踪
        forward_tracks, forward_vis = model(video, query_points)
        
        # 3. 后向追踪
        backward_tracks, backward_vis = model(
            video.flip(0),  # 时间反转
            forward_tracks[:, -1]  # 从最后位置开始
        )
        
        # 4. 循环一致性检验
        cycle_error = (query_points - backward_tracks[:, -1]).norm(dim=-1)
        consistent_mask = cycle_error < 0.02  # 阈值
        
        # 5. 置信度过滤
        confidence = (forward_vis * backward_vis.flip(1)).mean(dim=1)
        high_conf_mask = confidence > 0.8
        
        # 6. 合并过滤
        valid_mask = consistent_mask & high_conf_mask
        
        # 保存伪标签
        save_pseudo_labels(output_dir, video_name, {
            'tracks': forward_tracks[valid_mask],
            'visibility': forward_vis[valid_mask],
            'confidence': confidence[valid_mask],
        })
```

### 3.3 多阶段训练

```
阶段1: Kubric预训练 (50 epochs)
    ↓
阶段2: 伪标签生成 (在YouTube-VOS/Kinetics上)
    ↓
阶段3: 混合训练 (30 epochs, Kubric + 伪标签)
    ↓
阶段4: 微调 (10 epochs, 学习率降低)
```

---

## 四、消融实验详细设计

### 4.1 核心模块消融

**实验矩阵**:

| ID | 频率分解 | 语义编码 | 遮挡预测 | 训练配置 |
|----|----------|----------|----------|----------|
| A1 | ✗ | ✗ | ✗ | baseline |
| A2 | ✅ | ✗ | ✗ | +LFD |
| A3 | ✗ | ✅ | ✗ | +Semantic |
| A4 | ✗ | ✗ | ✅ | +Occlusion |
| A5 | ✅ | ✅ | ✗ | +LFD+Sem |
| A6 | ✅ | ✗ | ✅ | +LFD+Occ |
| A7 | ✗ | ✅ | ✅ | +Sem+Occ |
| A8 | ✅ | ✅ | ✅ | Full |

**执行脚本**:
```bash
for config in configs/ablations/*.yaml; do
    python train.py --config $config --experiment.seed 42
    python train.py --config $config --experiment.seed 123
    python train.py --config $config --experiment.seed 456
done
```

### 4.2 频带数量实验

```bash
for num_bands in 2 3 4 5 6 8; do
    python train.py \
        --config configs/fspt_base.yaml \
        --model.frequency.num_bands $num_bands \
        --experiment.name "ablation_bands_${num_bands}" \
        --paths.output_dir outputs/ablation/bands/${num_bands}
done
```

### 4.3 CLIP模型选择

```bash
for clip_model in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-L/14@336px"; do
    model_name=$(echo $clip_model | tr '/@' '_')
    python train.py \
        --config configs/fspt_base.yaml \
        --model.clip.model "$clip_model" \
        --experiment.name "ablation_clip_${model_name}" \
        --paths.output_dir outputs/ablation/clip/${model_name}
done
```

### 4.4 损失权重敏感性

```python
# scripts/sweep_loss_weights.py

import itertools

position_weights = [0.5, 1.0, 2.0]
occlusion_weights = [0.25, 0.5, 1.0]
ortho_weights = [0.01, 0.1, 0.5]

for pw, ow, ortw in itertools.product(
    position_weights, occlusion_weights, ortho_weights
):
    run_experiment(
        name=f"loss_pw{pw}_ow{ow}_ortw{ortw}",
        loss_config={
            'position.weight': pw,
            'occlusion.weight': ow,
            'frequency_ortho.weight': ortw,
        }
    )
```

### 4.5 骨干网络对比

```bash
for backbone in resnet18 resnet34 resnet50 resnet101 convnext_tiny convnext_small; do
    python train.py \
        --config configs/fspt_base.yaml \
        --model.backbone.type $backbone \
        --experiment.name "ablation_backbone_${backbone}"
done
```

### 4.6 训练策略消融（避免堆模块）

```bash
python train.py --config configs/ablations/temporal_consistency.yaml
python train.py --config configs/ablations/multi_iteration.yaml
python train.py --config configs/ablations/confidence_weighted.yaml
```

---

## 五、场景细分评估

### 5.1 遮挡程度分析

```python
# scripts/analyze_by_occlusion.py

def categorize_by_occlusion(dataset):
    """按遮挡程度分类"""
    categories = {
        'no_occlusion': [],      # 0%
        'light_occlusion': [],   # 0-20%
        'medium_occlusion': [],  # 20-50%
        'heavy_occlusion': [],   # >50%
    }
    
    for sample in dataset:
        occ_ratio = sample['occluded'].float().mean()
        
        if occ_ratio == 0:
            categories['no_occlusion'].append(sample)
        elif occ_ratio < 0.2:
            categories['light_occlusion'].append(sample)
        elif occ_ratio < 0.5:
            categories['medium_occlusion'].append(sample)
        else:
            categories['heavy_occlusion'].append(sample)
    
    return categories

def evaluate_by_category(model, categories):
    results = {}
    for category, samples in categories.items():
        metrics = evaluate_on_samples(model, samples)
        results[category] = metrics
    return results
```

### 5.2 运动类型分析

```python
def categorize_by_motion(dataset):
    """按运动类型分类"""
    categories = {
        'static': [],      # 平均位移 < 2px
        'slow': [],        # 2-10px
        'medium': [],      # 10-30px
        'fast': [],        # 30-100px
        'very_fast': [],   # >100px
    }
    
    for sample in dataset:
        tracks = sample['target_points']  # (N, T, 2)
        displacements = (tracks[:, 1:] - tracks[:, :-1]).norm(dim=-1)
        avg_displacement = displacements.mean() * 256  # 转换为像素
        
        if avg_displacement < 2:
            categories['static'].append(sample)
        elif avg_displacement < 10:
            categories['slow'].append(sample)
        elif avg_displacement < 30:
            categories['medium'].append(sample)
        elif avg_displacement < 100:
            categories['fast'].append(sample)
        else:
            categories['very_fast'].append(sample)
    
    return categories
```

### 5.3 视频长度分析

```python
def categorize_by_length(dataset):
    """按视频长度分类"""
    categories = {
        'short': [],   # ≤50 frames
        'medium': [],  # 51-100 frames
        'long': [],    # 101-200 frames
        'very_long': [], # >200 frames
    }
    
    for sample in dataset:
        num_frames = sample['video'].shape[0]
        
        if num_frames <= 50:
            categories['short'].append(sample)
        elif num_frames <= 100:
            categories['medium'].append(sample)
        elif num_frames <= 200:
            categories['long'].append(sample)
        else:
            categories['very_long'].append(sample)
    
    return categories
```

---

## 六、跨域实验

### 6.1 Zero-shot评估

```bash
# 训练在Kubric，直接测试在真实数据
python evaluate.py \
    --checkpoint outputs/pretrain/checkpoints/fspt_pretrain_kubric/best.pth \
    --dataset davis \
    --output-dir outputs/zeroshot/davis

python evaluate.py \
    --checkpoint outputs/pretrain/checkpoints/fspt_pretrain_kubric/best.pth \
    --dataset kinetics \
    --output-dir outputs/zeroshot/kinetics
```

### 6.2 Few-shot适应

```bash
# 使用少量真实数据微调
for num_samples in 5 10 20 50 100; do
    python train.py \
        --config configs/fspt_base.yaml \
        --data.train.num_samples $num_samples \
        --paths.pretrained.fspt outputs/pretrain/checkpoints/fspt_pretrain_kubric/best.pth \
        --experiment.name "fewshot_${num_samples}" \
        --paths.output_dir outputs/fewshot/${num_samples}
done
```

### 6.3 域适应实验

```python
# scripts/domain_adaptation.py

def semantic_consistency_loss(pred_tracks, semantic_features):
    """
    语义一致性损失用于无标签域适应
    
    思想：语义相似的点应该有相似的运动模式
    """
    B, N, T, _ = pred_tracks.shape
    
    # 计算语义相似度
    semantic_sim = compute_similarity(semantic_features)  # (B, N, N)
    
    # 计算运动相似度
    motion = pred_tracks[:, :, 1:] - pred_tracks[:, :, :-1]  # (B, N, T-1, 2)
    motion_sim = compute_motion_similarity(motion)  # (B, N, N)
    
    # 损失：语义相似的点应该运动相似
    loss = ((semantic_sim - motion_sim) ** 2).mean()
    
    return loss
```

---

## 七、效率实验

### 7.1 推理速度测试

```python
# scripts/benchmark_speed.py

import torch
import time

def benchmark_inference(model, video_size, num_points, num_runs=100):
    """测量推理速度"""
    B, T, C, H, W = video_size
    
    # 预热
    video = torch.randn(1, T, C, H, W).cuda()
    query = torch.rand(1, num_points, 3).cuda()
    for _ in range(10):
        _ = model(video, query)
    
    # 同步
    torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(video, query)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'fps': 1.0 / np.mean(times),
    }

# 不同配置测试
configs = [
    {'T': 24, 'H': 256, 'W': 256, 'N': 256},
    {'T': 50, 'H': 256, 'W': 256, 'N': 256},
    {'T': 100, 'H': 256, 'W': 256, 'N': 256},
    {'T': 24, 'H': 512, 'W': 512, 'N': 256},
    {'T': 24, 'H': 256, 'W': 256, 'N': 512},
    {'T': 24, 'H': 256, 'W': 256, 'N': 1024},
]
```

### 7.2 内存占用测试

```python
def measure_memory(model, video_size, num_points):
    """测量GPU内存占用"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    B, T, C, H, W = video_size
    video = torch.randn(1, T, C, H, W).cuda()
    query = torch.rand(1, num_points, 3).cuda()
    
    # 前向传播
    with torch.no_grad():
        _ = model(video, query)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
    
    return peak_memory
```

### 7.3 模型大小分析

```python
def count_parameters(model):
    """统计模型参数"""
    total = 0
    trainable = 0
    by_module = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        train_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        by_module[name] = {
            'total': params,
            'trainable': train_params,
        }
        
        total += params
        trainable += train_params
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'by_module': by_module,
    }
```

---

## 八、可视化实验

### 8.1 频率分解可视化

```python
# scripts/visualize_frequency.py

def visualize_frequency_bands(model, video, query_point):
    """可视化不同频带的响应"""
    # 提取频带特征
    with torch.no_grad():
        _, info = model.freq_semantic(video_features, semantic_features)
    
    band_features = info['band_features']
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, (band_name, features) in enumerate(band_features.items()):
        # 时域
        axes[0, i].plot(features[0, :, 0, :10].cpu().numpy())
        axes[0, i].set_title(f'{band_name} (Time Domain)')
        
        # 频域
        fft = np.abs(np.fft.fft(features[0, :, 0].cpu().numpy(), axis=0))
        axes[1, i].plot(fft[:len(fft)//2])
        axes[1, i].set_title(f'{band_name} (Frequency Domain)')
    
    plt.savefig('frequency_bands.png')
```

### 8.2 遮挡推理可视化

```python
def visualize_occlusion_reasoning(model, video, tracks, gt_visibility):
    """可视化遮挡推理过程"""
    # 获取遮挡预测
    with torch.no_grad():
        pred_tracks, pred_vis = model(video, query_points)
        
        # 获取中间结果
        occ_prob, pred_positions, confidence = model.occlusion_predictor(...)
    
    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 遮挡概率
    axes[0].imshow(occ_prob[0].cpu().numpy(), aspect='auto', cmap='Reds')
    axes[0].set_title('Occlusion Probability')
    
    # 预测轨迹 vs GT
    axes[1].plot(pred_tracks[0, 0, :, 0].cpu(), label='Predicted Y')
    axes[1].plot(gt_tracks[0, 0, :, 0].cpu(), '--', label='GT Y')
    axes[1].fill_between(range(T), 0, 1, where=~gt_visibility[0, 0].cpu(), alpha=0.3)
    axes[1].legend()
    
    # 置信度
    axes[2].plot(confidence[0, 0].cpu())
    axes[2].set_title('Prediction Confidence')
    
    plt.savefig('occlusion_reasoning.png')
```

### 8.3 语义调制可视化

```python
def visualize_semantic_modulation(model, video):
    """可视化语义特征如何调制频率响应"""
    # 提取语义特征
    semantic_features = model.semantic_encoder(video, mode='spatial')
    
    # 获取调制权重
    modulation_weights = model.freq_semantic.semantic_modulator.get_weights(semantic_features)
    
    # 可视化
    # ...
```

---

## 九、统计分析

### 9.1 多次运行

```bash
# 5次独立训练
for seed in 42 123 456 789 1024; do
    python train.py \
        --config configs/fspt_base.yaml \
        --experiment.seed $seed \
        --experiment.name "fspt_seed_${seed}" \
        --paths.output_dir outputs/multi_run/${seed}
done

# 汇总结果
python scripts/aggregate_results.py --runs outputs/multi_run/
```

### 9.2 显著性检验

```python
# scripts/significance_test.py

from scipy import stats

def paired_t_test(method1_results, method2_results):
    """配对t检验"""
    t_stat, p_value = stats.ttest_rel(method1_results, method2_results)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'significant_0.001': p_value < 0.001,
    }

def bootstrap_confidence_interval(results, n_bootstrap=10000, ci=0.95):
    """Bootstrap置信区间"""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(results, size=len(results), replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    
    return lower, upper
```

---

## 十、实验时间表

### 10.1 阶段划分

| 阶段 | 任务 | GPU时间 | 日历时间 |
|------|------|---------|----------|
| 1 | 基线复现 | 2天 | 3天 |
| 2 | FSPT预训练 | 4天 | 5天 |
| 3 | 伪标签微调 | 2天 | 3天 |
| 4 | 核心消融 | 8天 | 10天 |
| 5 | 场景分析 | 2天 | 2天 |
| 6 | 效率分析 | 1天 | 1天 |
| 7 | 可视化 | 2天 | 3天 |
| 8 | 统计分析 | 1天 | 1天 |
| **总计** | | **22天** | **28天** |

### 10.2 检查点

- [ ] Week 1: 基线复现完成，验证评估pipeline
- [ ] Week 2: FSPT预训练完成，初步结果
- [ ] Week 3: 核心消融完成
- [ ] Week 4: 所有实验完成，开始写作

---

*文档更新日期: 2026-01-26*
