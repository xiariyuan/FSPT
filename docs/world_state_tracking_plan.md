# World-State Point Tracking Pivot Plan

当前结论很明确：

- 继续做 `relocalization / verifier / selective gate` 的 2D 后处理线，论文上限很低。
- 新主线要换成 **persistent allocentric world-state tracking**。
- 核心问题不是“像素坐标在下一帧在哪”，而是“一个点作为世界状态，在遮挡、出视野、相机运动之后还能不能被重新恢复”。

这份计划把新方向拆成可执行实验，按 `1 天 -> 1 周 -> 2 周` 的节奏推进。

---

## 1. 论文核心假设

我们要证明的不是“又一个更强的 2D tracker”，而是下面三点：

1. 2D tracker 在长遮挡、出视野、视角变化下会把同一个点错配到别的实例或别的位置。
2. 如果把点提升到 `3D / allocentric world state`，就能获得 2D 做不到的恢复能力。
3. 这种恢复能力不仅在合成数据上成立，也能迁移到真实视频的动作/机器人场景。

一句话版本：

**Tracking a point as a persistent world state is strictly more informative than tracking it as a frame-local 2D correspondence.**

---

## 2. 可直接复用的资源

仓库里已经有足够的基础设施，不需要从零开始。

- 2D base tracker: `CoTracker3`
- 2D baseline inference: `scripts/inference.py`
- 长遮挡评估: `scripts/eval_long_occlusion_subset.py`
- oracle 分析: `scripts/eval_long_occlusion_oracle_gap.py`
- 3D/pose 数据集: `datasets/pointodyssey/`
- 机器人/动作场景: `datasets/tapvid_rgb_stacking/`
- 几何数据: `datasets/megadepth/`
- 预训练特征: `weights/dinov2/dinov2_vits14_pretrain.pth`, `weights/clip/ViT-B-16.pt`
- 现有 TAP-Vid/DAVIS/Kinetics 评估链路

PointOdyssey 的标注结构已经确认，包含：

- `trajs_2d`
- `trajs_3d`
- `valids`
- `visibs`
- `intrinsics`
- `extrinsics`

这意味着它可以直接用于 `2D vs 3D world-state` 的对照实验。

---

## 3. 新主线的定义

### 3.1 不做什么

这条线不是：

- 2D verifier
- 2D selective gate
- 语义重识别后处理
- 继续围绕 TAP-Vid DAVIS 的微小 AJ 提升

### 3.2 要做什么

要做的是：

1. 先把点从像素坐标 lift 到世界坐标或相机坐标。
2. 再用相机位姿把 world state 投影回每一帧。
3. 用这个 world state 去解决长遮挡、出视野、重入场景。

### 3.3 预期论文故事

我们不是说“track 更准了”，而是说：

- 点可以被建模成一个持久的世界状态
- 遮挡期间世界状态仍然连续
- 重入时可以利用世界连续性恢复同一目标

---

## 4. 评估指标

### 4.1 2D 指标

继续保留 TAP-Vid 官方协议下的指标，用于保证和现有方法可比：

- `AJ`
- `OA`
- `<1px`, `<2px`, `<4px`, `<8px`, `<16px`

### 4.2 3D / world-state 指标

这条线的主指标不应该只看 2D AJ，还要看下面这些：

- `world_position_error`：3D 世界坐标误差
- `reprojection_error_px`：3D state 投影回 2D 后的像素误差
- `reentry_error_px`：长遮挡结束后的第一帧重入误差
- `reentry_success@k`：重入帧在阈值 `k` 像素内的成功率
- `offscreen_retention`：出视野后仍能保持正确 world state 的比例
- `visibility_accuracy`：可见性预测准确率
- `camera_motion_stratified_error`：按相机运动强度分桶后的误差

### 4.3 任务相关指标

对 `RGB-Stacking` 这类动作场景，补一个更有应用意义的指标：

- `object_persistence_score`

含义是：点在遮挡、抓取、移动、重叠之后，是否仍保持在同一个对象/同一个结构上。

---

## 5. 实验矩阵

### Stage 0: 1 天 smoke test

目标：先回答一个问题。

**3D world-state 是否真的比 2D tracker 更有恢复能力？**

#### 数据

- `PointOdyssey` 的验证/测试子集
- 只选满足以下条件的 query：
  - 发生长遮挡
  - 遮挡后确实重新出现
  - 相机有明显运动

#### 对照

1. `GT 3D oracle`
   - 用 `trajs_3d + intrinsics + extrinsics` 直接投影回 2D
   - 这是上限，不是模型

2. `CoTracker3 2D baseline`
   - 作为当前强 2D 参照

3. `Naive 3D lift`
   - 先把 2D tracker 的点结合深度或简单几何 lift 到 3D
   - 这是“world-state 但是很弱”的下限

#### 要看什么

- 长遮挡长度越大，2D 与 3D 的差距是否越明显
- 重入帧上，3D 是否明显优于 2D
- 在出视野 / 视角变化强的样本上，3D 是否仍然稳定

#### 通过标准

如果满足下面任意一个条件，就值得继续：

- `reentry_error_px` 明显低于 2D baseline
- `offscreen_retention` 明显高于 2D baseline
- 在 camera-motion heavy 子集上，3D 的优势比普通子集更大

如果 3D 和 2D 的差距还是接近噪声，就直接停止，不再继续投资源。

---

### Stage 1: 3 到 5 天 predicted-depth prototype

目标：验证 **不用 GT 3D，也能不能保留世界状态优势**。

#### 输入

- `CoTracker3` 的 2D tracks
- 单目深度估计
  - 优先 `Depth Anything V2`
  - 备选 `MiDaS`
- `PointOdyssey` 的 camera intrinsics / extrinsics

#### 方法

1. 先把 2D track + depth lift 成 3D point state。
2. 再用相机位姿做 world-to-camera projection。
3. 在连续帧上做 temporal smoothing，避免深度噪声抖动。
4. 加一个简单的 visibility head 或阈值判断，区分“在视野内但遮挡”和“已经出视野”。

#### 对照实验

- `2D CoTracker3`
- `2D + naive depth lift`
- `2D + depth + temporal smoothing`
- `GT 3D oracle`

#### 要看什么

- 预测深度是否能保留至少一部分 GT 3D 的优势
- 3D lift 是否能在重入帧上比纯 2D 更稳定
- 2D AJ 是否被明显破坏

#### 通过标准

如果 predicted-depth 版本在长遮挡 / 重入帧上的收益达到 GT 3D oracle 的可见比例，并且 2D AJ 没有明显退化，这条线才值得进入训练阶段。

---

### Stage 2: 1 到 2 周 minimal learnable model

目标：把 `world-state` 变成一个可学习模块，而不是手工后处理。

#### 模型定义

输入：

- 视频帧
- 2D base tracks
- 深度图
- 相机位姿

输出：

- `world_point`：持久 3D 点状态
- `visibility`
- `confidence`

#### 训练目标

1. `3D position regression`
   - 对齐 `trajs_3d`

2. `reprojection consistency`
   - world state 投影回 2D 后与 GT 2D 对齐

3. `visibility supervision`
   - 对齐 `visibs`

4. `temporal smoothness`
   - 避免 world state 抖动

5. `reentry consistency`
   - 遮挡前后同一个点的 world state 应一致

#### 训练数据

- 主训练：`PointOdyssey train`
- 预训练或辅助几何监督：`MegaDepth`
- 应用验证：`RGB-Stacking`

#### 必须做的 ablation

1. 只用 2D
2. 2D + depth
3. 2D + depth + camera compensation
4. 2D + depth + camera compensation + temporal smoothing
5. 去掉 visibility head
6. 去掉 reentry loss

#### 通过标准

下面三条至少满足两条，才算有论文潜力：

- 3D 指标明显优于 2D baseline
- 长遮挡重入帧上提升显著
- 2D 官方指标没有明显掉点

如果只是“略好一点”，不要继续扩大投入。

---

### Stage 3: 2 到 3 天 application validation

目标：证明这不是一个纯 benchmark 技巧，而是有下游价值。

#### 数据

- `RGB-Stacking`

#### 为什么选它

- 有动作、遮挡、物体交互
- 比纯 DAVIS 更接近“world action”叙事
- 适合证明点状态对动作相关场景更稳

#### 要做什么

1. 在 stacking / pick-place / occlusion 场景上跑 track continuity。
2. 看同一个点是否还能稳定留在同一个物体或结构上。
3. 对比纯 2D track 与 world-state track 的 object persistence。

#### 目标

- 不是一定要刷新所有 2D 指标
- 而是要证明 `world-state` 在 action-heavy 场景更有意义

---

## 6. 推荐执行顺序

### 第 1 步

先做 `Stage 0`。

只要 `PointOdyssey` 上的 3D oracle 没有明显比 2D 更好，这条线就立刻停止。

### 第 2 步

如果 `Stage 0` 通过，马上做 `Stage 1`。

先验证 predicted-depth 版本，不要直接上复杂模型。

### 第 3 步

如果 `Stage 1` 也通过，再做 `Stage 2`。

这时才值得写训练代码。

### 第 4 步

最后补 `Stage 3` 作为应用场景。

---

## 7. 停止条件

这条线必须有明确的停止条件，避免继续浪费时间。

### 立即停止

满足以下任一条就停止：

- `GT 3D oracle` 和 `2D baseline` 的差距不明显
- predicted-depth 版本无法保留 GT 3D 优势
- learnable model 比 predicted-depth 还差

### 继续推进

满足以下任一条可以继续：

- 重入帧明显优于 2D
- 出视野恢复明显优于 2D
- 相机运动越强，3D 优势越大

---

## 8. 你接下来实际要跑的东西

### 先看数据

- 抽取 10 到 20 个 `PointOdyssey` 场景
- 只保留长遮挡 + 重入 + 相机运动大的样本

### 先做 oracle

- 用 GT `3D + extrinsics + intrinsics` 投影回每帧
- 画出 2D baseline vs 3D oracle 的对照曲线

### 再做 predicted depth

- 先不用训练模型
- 先验证 depth lift 是否能提供可用信号

### 最后才训练

- 如果前两步都成立，再训练 minimal 3D world-state tracker

---

## 9. 这条线的论文定位

如果实验成立，这篇论文的定位应该是：

**From frame-local tracking to persistent allocentric world-state tracking.**

它和现有工作最本质的差别不是“多一个模块”，而是“换了问题定义”。

