# PRT Benchmark Definition

## 1. Scope

`PRT` 表示 `Persistent Re-entry Tracking`。

目标不是评估一般意义上的 point tracking，而是评估一个更具体的问题：

> 在严格因果设置下，一个点在经历长遮挡、相机运动、甚至暂时离开视野后，系统能否在 first visible re-entry frame 正确恢复它的身份和位置？

这个 benchmark 服务于当前 world-state 主线，重点覆盖：

- long occlusion
- camera motion
- off-screen return
- first-visible re-entry recovery

---

## 2. Benchmark Variants

### 2.1 PRT-Synth

来源：

- `PointOdyssey`

用途：

- 主分析 benchmark
- oracle / noisy-depth / 方法对比
- 分层统计

### 2.2 PRT-Real

来源：

- `tapvid_rgb_stacking`

用途：

- 真实场景应用验证
- 检查 synthetic-to-real transfer

当前优先级：

- 先完成 `PRT-Synth`
- `PRT-Real` 作为第二阶段扩展

---

## 3. Query Definition

每个 query 对应同一个 point 的一次 `re-entry` 事件。

定义如下：

1. 在 `t_q` 该点可见。
2. 在 `t_q + 1 ... t_r - 1` 该点连续不可见。
3. 在 `t_r` 该点再次可见。
4. `t_r` 必须是 first visible re-entry frame。

其中：

- `query frame = t_q`
- `re-entry frame = t_r`
- `occlusion length = t_r - t_q - 1`

注：

- 这里把“遮挡后重新出现”和“暂时离开视野后重新进入视野”统一成 `re-entry`。
- 评估重点是 `t_r` 这一帧，以及必要时 `t_r ... t_r + K` 的短窗口。

---

## 4. Re-entry Types

对每个 query，必须标记它属于哪种 re-entry。

### 4.1 In-frame Occlusion

在不可见区间内，点的大多数轨迹位置仍落在图像边界内，但由于被遮挡而不可见。

Operational definition:

- 在 occlusion interval 内，至少一半帧满足：
  - `valid = True`
  - `x, y` 在图像范围内
  - `visibs = False`

### 4.2 Off-screen Return

在不可见区间内，点有显著比例的帧落在图像边界外，说明它离开了当前视野。

Operational definition:

- 在 occlusion interval 内，至少一半帧满足：
  - `x, y` 超出图像范围
  - `visibs = False`

### 4.3 Mixed

若两种模式都不占主导，则标记为 `mixed`。

---

## 5. Required Stratification

PRT-Synth 必须至少报告以下分层：

1. `occ_len >= 10`
2. `occ_len >= 20`
3. `occ_len >= 30`
4. `occ_len >= 50`

以及：

1. `camera_motion >= 0.1`
2. `camera_motion >= 0.3`
3. `camera_motion >= 0.5`

再加两类语义分层：

1. `in_frame_occlusion`
2. `off_screen_return`

如果样本量允许，再加：

1. `same_view`
2. `large_view_change`

其中 `camera_motion` 采用 query frame 与 re-entry frame 间相对位姿变化的旋转范数定义。

---

## 6. Core Metrics

### 6.1 Primary Metrics

所有方法必须在 re-entry frame 上报告：

- `reentry_median_px`
- `reentry_lt4px`
- `reentry_lt8px`
- `reentry_lt16px`

### 6.2 Comparative Metrics

对方法与 baseline 的比较，必须报告：

- `better_frac_vs_baseline`
- `delta_median_px`
- `delta_lt4px`

### 6.3 Reliability Metrics

如果方法带有 accept/reject 或 confidence：

- `accept_rate`
- `abstain_rate`
- `coverage_error_curve`

### 6.4 World-State Diagnostics

对 world-state 相关方法，还建议报告：

- `world_position_error`
- `reprojection_error_px`
- `offscreen_retention`

---

## 7. Baseline Ladder

PRT benchmark 上建议固定以下 baseline 梯度：

1. `2D hold`
   - query frame 的 2D 点直接保持不动

2. `3D hold`
   - query frame 的 3D world point 保持不动，再投影到 re-entry frame

3. `3D extrap`
   - query 前可见历史估计速度，在世界坐标外推，再投影

4. `Pred-depth 3D hold`
   - 使用预测深度进行 lift，再做 3D hold

5. `Base tracker re-entry prediction`
   - 当前实际对比方法的 baseline

6. `Candidate-based causal recovery`
   - 后续方法线

---

## 8. Immediate Deliverables

第一阶段必须先产出：

1. `scripts/build_prt_splits.py`
2. `outputs/prt_split_stats.json`
3. `outputs/prt_split_preview.json`

内容要求：

- 每个 split 的 query 总数
- 每种 `re-entry type` 的数量
- 每个 `occ_len x camera_motion` 分桶的数量
- 若可能，输出每个分桶的代表性样例 id

---

## 9. Non-Goals

PRT 不关注：

- 全视频平均 AJ 的微小提升
- 非重入阶段的普通短时 tracking
- 非因果设置下使用未来帧做后处理

这些不是当前 benchmark 的主要目标。

---

## 10. Success Criterion For This Benchmark

如果一个方法要在 PRT 上宣称有效，至少需要：

1. 在 hardest split 上稳定降低 `reentry_median_px`
2. 同时提升 `reentry_lt4px`
3. 不只是偶然命中少量样本
4. 优于单帧 selector / acceptor 这类弱基线

如果只达到：

- 有 oracle gap
- 有 noisy-depth gain
- 但 causal learned recovery 仍失败

那么 PRT 依然成立为一个有价值的问题定义和分析 benchmark。
