# 先进训练策略

本项目借鉴了近几年点追踪领域顶会论文的训练策略，以提升模型性能和训练稳定性。

## 参考论文

| 论文 | 会议 | 核心贡献 |
|-----|------|---------|
| **CoTracker3** | ICCV 2025 | 伪标签自训练、置信度预测、简化架构 |
| **TAPIR** | ICCV 2023 | 双阶段架构、时序精化、不确定性估计 |
| **BootsTAP** | ACCV 2024 | 自举训练、学生-教师框架、域适应 |
| **TAPNext** | ICCV 2025 | 序列建模、低延迟在线追踪 |
| **LocoTrack** | ECCV 2024 | 4D局部相关、双向一致性 |
| **SpatialTracker** | CVPR 2024 | 3D空间追踪、刚性约束 |

---

## 1. 渐进式训练 (Progressive Training)

**来源**: CoTracker3, TAPIR

渐进式训练从简单设置开始，逐步增加难度：

```yaml
training:
  progressive:
    enabled: true
    smooth_transition: true
    transition_epochs: 2
```

**默认策略**:
| 阶段 | Epochs | 分辨率 | 帧数 | 点数 | 学习率 |
|-----|--------|--------|------|------|--------|
| 1 (热身) | 20% | 128×128 | 8 | 64 | 0.5× |
| 2 (中等) | 30% | 256×256 | 12 | 128 | 1.0× |
| 3 (完整) | 50% | 256×256 | 24 | 256 | 0.5× |

**代码使用**:
```python
from utils.advanced_training import ProgressiveTrainingScheduler, create_progressive_stages

stages = create_progressive_stages(config)
scheduler = ProgressiveTrainingScheduler(stages)

for epoch in range(total_epochs):
    stage = scheduler.get_current_stage(epoch)
    # 使用 stage.resolution, stage.num_frames 等配置数据加载
```

---

## 2. 置信度加权损失 (Confidence-Weighted Loss)

**来源**: CoTracker3

模型预测轨迹置信度，高置信度样本权重更大：

```yaml
loss:
  confidence_weighted:
    enabled: true
    weight: 0.1
    min_confidence: 0.1
```

本项目训练默认在 `PointTrackingLoss` 中通过 `loss.confidence_weighted` 集成该策略；
如需单独使用，可参考下面的独立损失实现。

**损失公式**:
```
L = mean(confidence × position_loss) + λ × confidence_regularization
```

**代码使用**:
```python
from utils.advanced_training import ConfidenceWeightedLoss

criterion = ConfidenceWeightedLoss(
    position_loss_type='smooth_l1',
    confidence_weight=0.1,
)

loss_dict = criterion(pred_tracks, gt_tracks, confidence, visibility)
```

---

## 3. 伪标签自训练 (Pseudo-Label Self-Training)

**来源**: CoTracker3, BootsTAP

使用教师模型在无标签真实视频上生成伪标签：

```yaml
training:
  pseudo_labeling:
    enabled: true
    teacher_momentum: 0.999
    confidence_threshold: 0.5
    warmup_epochs: 10
    # 可选：false 表示用硬可见性标签（pred_visibility > 0.5）
    use_soft_labels: true
```

**训练流程**:
1. 教师模型在真实视频上生成轨迹预测
2. 筛选高置信度预测作为伪标签，并用掩码过滤低置信度点
3. 学生模型在伪标签上训练
4. EMA更新教师模型

**代码使用**:
```python
from utils.advanced_training import PseudoLabelTrainer

trainer = PseudoLabelTrainer(
    teacher_model=teacher,
    student_model=student,
    confidence_threshold=0.5,
    teacher_momentum=0.999,
    use_soft_labels=True,
)

for batch in unlabeled_loader:
    loss = trainer.train_step(
        batch['video'],
        batch['query_points'],
        criterion,
        optimizer,
    )
    # trainer内部使用visibility_mask，仅在高置信度点计算损失
```

---

## 4. 时序一致性正则化 (Temporal Consistency)

**来源**: TAPIR, LocoTrack

鼓励轨迹在时间上平滑，惩罚突变：

```yaml
loss:
  temporal_consistency:
    enabled: true
    order: 2  # 二阶差分（加速度）
    weight: 0.01
```

**损失公式**:
- 一阶: `L = ||x_t - x_{t-1}||`
- 二阶: `L = ||x_t - 2×x_{t-1} + x_{t-2}||`

**代码使用**:
```python
from utils.advanced_training import TemporalConsistencyLoss

temporal_loss = TemporalConsistencyLoss(order=2, weight=0.01)
loss = temporal_loss(pred_tracks, visibility)
```

---

## 5. 多迭代监督 (Multi-Iteration Supervision)

**来源**: CoTracker, RAFT

对每次迭代更新都计算损失，后期迭代权重更大：

```yaml
loss:
  multi_iteration:
    enabled: true
    gamma: 0.8  # 权重衰减因子
```

**权重分布** (4次迭代, γ=0.8):
| 迭代 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| 权重 | 0.13 | 0.16 | 0.20 | 0.25 |

**代码使用**:
```python
from utils.advanced_training import MultiIterationLoss

multi_loss = MultiIterationLoss(num_iters=4, gamma=0.8)
loss_dict = multi_loss(all_predictions, targets, visibility)
```

---

## 6. 域适应 (Domain Adaptation)

**来源**: BootsTAP

减少合成数据和真实数据之间的分布差距：

```python
from utils.advanced_training import DomainAdaptationLoss

domain_loss = DomainAdaptationLoss(
    feature_dim=256,
    domain_weight=0.1,
    use_gradient_reversal=True,
)

loss_dict = domain_loss(synthetic_features, real_features)
```

**包含**:
- 对抗性域判别器
- Maximum Mean Discrepancy (MMD)
- 梯度反转层

---

## 完整训练配置示例

```yaml
training:
  epochs: 100
  batch_size: 8
  
  # 渐进式训练
  progressive:
    enabled: true
    smooth_transition: true
  
  # 伪标签
  pseudo_labeling:
    enabled: true
    confidence_threshold: 0.5
  
  # EMA
  ema:
    enabled: true
    decay: 0.9999

loss:
  position:
    weight: 1.0
    type: "smooth_l1"
  
  # 置信度加权
  confidence_weighted:
    enabled: true
    weight: 0.1
  
  # 时序一致性
  temporal_consistency:
    enabled: true
    order: 2
    weight: 0.01
  
  # 多迭代监督
  multi_iteration:
    enabled: true
    gamma: 0.8
```

---

## 训练技巧总结

| 技巧 | 效果 | 来源 |
|-----|------|------|
| 渐进式分辨率 | 加速收敛，稳定训练 | CoTracker3 |
| 伪标签 | 利用无标签真实数据 | CoTracker3, BootsTAP |
| 置信度加权 | 聚焦可靠样本 | CoTracker3 |
| 时序一致性 | 平滑轨迹，减少抖动 | TAPIR, LocoTrack |
| 多迭代监督 | 加速迭代收敛 | CoTracker, RAFT |
| 模型EMA | 稳定训练，提升泛化 | 通用 |
| 混合精度 | 加速训练，节省显存 | 通用 |

---

## 推荐训练流程

1. **阶段1**: 合成数据预训练（Kubric）
   - 使用渐进式训练
   - 启用所有损失项
   - 50k iterations

2. **阶段2**: 真实数据微调
   - 启用伪标签自训练
   - 降低学习率 (10x)
   - 10k iterations

3. **阶段3**: 测试时适应（可选）
   - 对特定视频微调
   - 自监督损失
