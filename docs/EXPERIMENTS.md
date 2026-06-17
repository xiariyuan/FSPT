# FSPT 实验设计

## 一、评估指标

### 1.1 TAP-Vid 官方指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **AJ (Average Jaccard)** | 综合位置和可见性 | Jaccard指数在多个阈值上的平均 |
| **< δ^x_avg** | 位置精度 | 可见点中位置误差小于阈值的比例 |
| **OA (Occlusion Accuracy)** | 遮挡预测准确率 | 正确预测遮挡/可见状态的比例 |

> 默认**排除查询帧**参与指标计算（与官方评估一致），可在评估脚本中选择包含。

### 1.2 指标计算

```python
from datasets.metrics import compute_tapvid_metrics

metrics = compute_tapvid_metrics(
    pred_tracks,          # (N, T, 2)
    gt_tracks,            # (N, T, 2)
    pred_vis,             # (N, T)
    gt_vis,               # (N, T)
    query_points,         # (N, 3) [t, y, x]
    thresholds=[1, 2, 4, 8, 16],
    resolution=(H, W),    # 或单一整数
    exclude_query_frame=True,
)
```

---

## 二、基线实验

### 2.1 复现基线结果

| 方法 | TAP-Vid Kinetics AJ | TAP-Vid DAVIS AJ | 备注 |
|------|---------------------|------------------|------|
| TAPIR | 60.2 | 62.9 | 官方结果 |
| CoTracker | 63.5 | 66.8 | 官方结果 |
| LocoTrack | 64.1 | 65.2 | 官方结果 |
| CoTracker3 | 65.8 | 68.2 | 官方结果 |
| **Ours (目标)** | **68+** | **70+** | 期望提升 |

### 2.2 复现验证检查点

- [ ] TAPIR在DAVIS上达到62.9 AJ
- [ ] CoTracker3在DAVIS上达到68.2 AJ
- [ ] LocoTrack在DAVIS上达到65.2 AJ

---

## 三、主实验

### 3.1 与SOTA对比

**实验配置**:
- 训练数据: TAP-Vid Kubric
- 测试数据: TAP-Vid DAVIS, Kinetics
- 查询点数: 256 per video
- 分辨率: 256×256 (标准), 512×512 (高分辨率)

**对比方法**:
1. TAPIR (ICCV 2023)
2. CoTracker (ECCV 2024)
3. LocoTrack (ECCV 2024)
4. CoTracker3 (ICCV 2025)
5. AllTracker (ICCV 2025)
6. **FSPT (Ours)**

### 3.2 按场景分析

**遮挡场景**:
```
实验设计: 筛选DAVIS中遮挡比例>30%的视频子集
目标: 验证频率感知遮挡推理的有效性
期望: 在高遮挡场景下比基线提升5%+ AJ
```

**快速运动场景**:
```
实验设计: 筛选平均位移>10像素/帧的视频
目标: 验证高频分量对精确定位的帮助
期望: 位置误差<4px的比例提升3%+
```

**慢速运动场景**:
```
实验设计: 筛选平均位移<2像素/帧的视频
目标: 验证低频分量对长程关联的帮助
期望: 长序列(>100帧)AJ提升
```

---

## 四、消融实验

### 4.1 频率分解模块消融

| ID | 配置 | 说明 |
|----|------|------|
| A1 | Baseline (无频率分解) | 纯CoTracker架构 |
| A2 | + 固定频率分解 | 使用固定拉普拉斯金字塔 |
| A3 | + 可学习频率分解 (LFD) | 使用我们的LFD模块 |
| A4 | + 频率感知时序注意力 | 不同频带不同注意力窗口 |

**期望结果**:
- A3 > A2 > A1 (验证可学习的必要性)
- A4 > A3 (验证频率感知注意力的有效性)

### 4.2 语义模块消融

| ID | 配置 | 说明 |
|----|------|------|
| B1 | 无语义特征 | 纯几何追踪 |
| B2 | + CLIP特征 (冻结) | 添加语义特征，不微调 |
| B3 | + 语义调制 | 语义特征调制频率响应 |
| B4 | + 语义一致性约束 | 添加语义一致性损失 |

**期望结果**:
- B3 > B2 > B1 (验证语义引导的有效性)
- B4进一步提升遮挡处理能力

### 4.3 频带数量消融

| 频带数 | Kinetics AJ | DAVIS AJ | FLOPs | 备注 |
|--------|-------------|----------|-------|------|
| 2 | - | - | 低 | 最简配置 |
| 3 | - | - | 中 | |
| **4** | - | - | 中 | 默认配置 |
| 6 | - | - | 高 | |
| 8 | - | - | 很高 | |

### 4.4 遮挡推理消融

| ID | 配置 | 遮挡场景AJ | 说明 |
|----|------|------------|------|
| C1 | 无遮挡推理 | - | Baseline |
| C2 | 简单遮挡分类 | - | 二分类预测 |
| C3 | 低频轨迹外推 | - | 我们的方法 |
| C4 | + 语义一致性传播 | - | 完整方法 |

---

## 五、效率分析

### 5.1 推理速度对比

| 方法 | FPS (256×256) | FPS (512×512) | GPU内存 |
|------|---------------|---------------|---------|
| TAPIR | 30+ | 15+ | ~4GB |
| CoTracker3 | 25+ | 10+ | ~6GB |
| LocoTrack | 50+ | 25+ | ~3GB |
| **FSPT (Ours)** | 目标: 30+ | 目标: 15+ | <8GB |

### 5.2 模型参数量

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Backbone | ~25M | ResNet-50 |
| CLIP (冻结) | ~150M | 不计入可训练参数 |
| LFD模块 | ~5M | 频率分解 |
| 时序Transformer | ~20M | 时序建模 |
| 遮挡预测器 | ~3M | 遮挡推理 |
| **总计** | ~53M | 不含CLIP |

---

## 六、跨域实验

### 6.1 合成→真实迁移

| 训练数据 | 测试DAVIS AJ | 测试Kinetics AJ |
|----------|--------------|-----------------|
| Kubric Only | - | - |
| + DAVIS (少量) | - | - |
| + 伪标签真实视频 | - | - |

### 6.2 伪标签质量分析

```python
def analyze_pseudo_labels(pseudo_tracks, teacher_tracks, clip_features):
    """
    分析伪标签质量
    """
    # 1. 与教师模型的一致性
    teacher_consistency = compute_trajectory_similarity(pseudo_tracks, teacher_tracks)
    
    # 2. 语义一致性 (轨迹上CLIP特征变化)
    semantic_consistency = compute_feature_variance_along_track(clip_features)
    
    # 3. 频率分析 (轨迹平滑度)
    frequency_smoothness = compute_frequency_spectrum(pseudo_tracks)
    
    # 综合可靠性分数
    reliability = 0.4 * teacher_consistency + 0.3 * semantic_consistency + 0.3 * frequency_smoothness
    
    return reliability
```

---

## 七、可视化分析

### 7.1 必要可视化

1. **频带分解可视化**
   - 显示同一轨迹在不同频带的分解
   - 展示低频=趋势，高频=细节

2. **语义引导可视化**
   - 展示语义相似点的联合追踪
   - 对比有/无语义引导的匹配结果

3. **遮挡恢复可视化**
   - 展示遮挡期间的轨迹预测
   - 对比基线方法的遮挡处理

4. **注意力可视化**
   - 展示不同频带的注意力范围
   - 验证低频用长程、高频用短程

### 7.2 失败案例分析

需要分析的失败场景:
- 极端遮挡 (>80%帧被遮挡)
- 多个相似物体
- 剧烈形变
- 快速运动模糊

---

## 八、补充实验 (如有时间)

### 8.1 3D点追踪
- 在TAPVid-3D上评估
- 验证频率分解是否有助于深度一致性

### 8.2 开放词汇点追踪
- 给定文本描述追踪特定类型的点
- 例如: "追踪所有人脸上的点"

### 8.3 视频编辑应用
- 点追踪辅助的视频稳定
- 点追踪辅助的物体跟踪

---

## 九、实验时间表

| 周次 | 实验内容 | 预计GPU时间 |
|------|----------|-------------|
| W9 | 基线复现 | 10 GPU-hours |
| W10 | 主实验 (完整模型) | 100 GPU-hours |
| W11 | 消融: 频率分解 | 50 GPU-hours |
| W12 | 消融: 语义模块 | 50 GPU-hours |
| W13 | 消融: 遮挡推理 | 30 GPU-hours |
| W14 | 跨域实验 | 80 GPU-hours |
| W15 | 效率优化 | 20 GPU-hours |
| W16 | 补充实验 | 50 GPU-hours |
| **总计** | | **~400 GPU-hours** |

---

## 十、实验记录模板

```yaml
experiment:
  name: "fspt_v1_baseline"
  date: "2026-02-01"
  config: "configs/fspt_base.yaml"
  
settings:
  batch_size: 8
  learning_rate: 1e-4
  epochs: 100
  gpus: 2
  
results:
  tapvid_davis:
    AJ: 0.0
    "<4px": 0.0
    OA: 0.0
  tapvid_kinetics:
    AJ: 0.0
  training_time: "XX hours"
  
notes: |
  - 首次训练
  - 观察到的问题: ...
  - 下一步计划: ...
```
