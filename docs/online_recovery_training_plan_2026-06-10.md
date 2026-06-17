# Online Recovery Training Plan (2026-06-10)

## 1. Decision

应该开始训练，但**不是**训练当前这条 `verifier / acceptor / policy gate / retracking threshold` 线。

应该训练的是一个新的、真正产生几何修正的 recovery head，并且先以 **offline anchor prediction** 为最小目标启动，而不是一开始就追求整体 AJ 提升。

一句话版本：

> 现在可以开训，但只能训 “DINO-conditioned geometric recovery head”；不要再训“只负责接受/拒绝候选”的头。

## 2. Why

当前仓库里的关键证据已经足够说明问题性质：

1. 在线 recovery 链路已经打通
   - `trigger -> search -> confidence -> retracking splice` 在真实数据上已工作。

2. DINO 的确比 CoTracker 原生特征强
   - 它显著提高了 `relocal_conf`，并且能在更合理的 `min_conf` 下独占性地触发 retracking。

3. 但最终几何修正没有发生
   - 当前 DINO search/readout 产生的 anchor 与 base tracker 原始位置几乎一样。
   - 这说明瓶颈不在 trigger，也不在 gate，而在 **geometry-producing readout** 本身。

因此，继续训练当前 gate/verifier 路线，大概率只会得到：

- 更会“决定是否采用”
- 但仍然没有新的几何位移可以采用

这不是用户真正要的恢复能力。

## 3. What Not To Train

以下内容当前不值得作为主训练目标：

1. `relocal_acceptor_head`
   - 它只是一个小 MLP gate，不产生新坐标。

2. `scripts/train_world_state_acceptor.py`
   - 这个脚本的任务定义是 candidate accept/reject，不做坐标回归。

3. 仅调 `conf_threshold / min_conf / temperature / argmax-softargmax`
   - 这些低成本消融已经基本说明不是配置问题。

4. 直接追整体 AJ
   - recovery 帧占比太低，整体 AJ 会把信号淹没。

## 4. Recommended Training Target

推荐的新训练目标是：

**冻结 base tracker + 冻结 DINO backbone + 训练一个 recovery head，直接预测 re-entry anchor。**

### Inputs

1. 当前 re-entry 帧附近的 DINO search feature map
2. 遮挡前支持记忆（1~K 个 support descriptors / support patches）
3. base tracker 在 re-entry 时刻的当前位置
4. 可选的 tracker side info
   - visibility
   - confidence
   - occlusion length
   - support age / support count

### Output

二选一，优先第 1 个：

1. `64x64` 或 `48x48` heatmap
   - 表示 search window 内每个位置是 GT re-entry 的概率
2. 二维 offset
   - 直接回归相对 base position 的位移

优先 heatmap 的原因：

- 监督更稳定
- 更容易看失败模式
- 更适合重尾误差分布

### Loss

第一版建议只做几何，不加复杂 policy：

1. heatmap CE / focal loss
2. soft-argmax 后的 L1 或 L2 position loss
3. 可选 confidence head
   - 但这不是第一优先级

## 5. Minimum Viable Training Protocol

### Stage M0: Offline Anchor Predictor

目标不是在线提升 AJ，而是回答一个更基本的问题：

> 在真实 tracker-conditioned re-entry 场景里，learned recovery head 能否把 `anchor_t0` 做得比 base 更准？

这一步只要做不出来，就不要进入大规模在线训练。

### 数据构建

训练集建议来自 PointOdyssey train，评估集建议至少包含：

1. PointOdyssey val
2. TAP-Vid DAVIS long-occlusion subset

每个样本至少保存：

1. `video_id / sequence_id`
2. `query index`
3. `query frame`
4. `re-entry frame`
5. `occlusion length`
6. `base tracker position at t0`
7. `GT position at t0`
8. `base error at t0`
9. DINO search crop feature map
10. support descriptors / support patch features
11. tracker visibility/confidence at t0

### 触发口径

训练时建议先用 **GT-defined re-entry events** 产样本，但输入必须仍然来自 tracker 自己的状态和图像特征。

原因：

- 这样只用 GT 定位“该学哪些时刻”
- 不把 GT 混进推理特征
- 可以先隔离 trigger 噪声，专心验证几何 head 是否有学习价值

### 模型最小结构

第一版不要复杂化。建议：

1. support encoder
   - 对 1~K 个 support descriptors 做平均或 max pooling
2. similarity volume builder
   - support 与 search feature map 做 cosine 相似度
3. small conv head
   - `sim_map + search feature map (+ optional base-centered coord channels)` -> refined heatmap
4. readout
   - soft-argmax 或 argmax，仅作输出，不要让它主导结论

### 为什么不先做更复杂的 transformer

因为当前最需要确认的是：

- DINO + support memory 是否能学出有效位移

而不是：

- 更复杂模型是否能在大量自由度下偶然过拟合

## 6. Success Criteria

只有满足下面这些条件，才说明训练值得继续：

### 必过指标

1. `anchor_t0_error_mean` 相比 base 下降至少 `15%`
2. `anchor_t0_error_p95` 相比 base 下降至少 `10%`
3. `better_frac` 超过 `0.55`

### 统计要求

1. sequence-level grouped bootstrap
2. train / val / test 必须按序列切分
3. 不允许按 query 随机打散泄漏相邻样本

### 次级指标

1. 按 occlusion length 分桶报告
   - `20~100`
   - `100~200`
   - `200~500`
   - `500+`
2. 报重尾统计
   - median
   - mean
   - p90 / p95

原因很直接：

当前问题不是纯中位数问题，而是明显重尾问题。

## 7. Stop Criteria

以下任一成立，就应该停止当前 recovery-head 方向，不再继续堆训练：

1. M0 上 `anchor_t0_error_mean` 改善 < `5%`
2. `p95` 不降反升
3. 改善只出现在极短遮挡段，`100+` frame 区间无稳定收益
4. 多随机种子结果不稳定
5. 只能在 PointOdyssey 内部成立，迁移到 DAVIS 后消失

这说明：

- 不是没有调好
- 而是当前 signal / task design 不足以支撑可泛化的恢复模块

## 8. Integration Order

正确顺序应该是：

1. 先训练并验证 offline anchor predictor
2. 再把它接回 online recovery pipeline，替换当前 raw cosine readout
3. 再重新校准 confidence / retracking threshold
4. 最后才看 recovery-specific online metrics
5. 最后最后才看整体 AJ

不要倒着做。

如果连第 1 步都还没赢，就不该进入第 4、5 步。

## 9. Baseline Choice

当前仓库下，最合理的工程 baseline 仍然是 **CoTracker3 integration path**，理由不是它一定最好，而是：

1. 现有集成、触发、retracking、诊断代码都已在这条线上打通
2. 当前所有负结果和瓶颈定位也都在这条线上完成
3. 现在切到另一个 tracker，会把“方法问题”和“工程迁移问题”混在一起

所以建议：

- **继续用 CoTracker3 作为 integration baseline**
- **用 DINO 作为 recovery-only feature source**
- **新增一个可训练 geometric head**

## 10. Claude Execution Checklist

下面这份清单可以直接交给 Claude 执行。

### Phase 1: Dataset Builder

1. 新建一个 recovery-anchor 数据构建脚本
2. 输入：
   - PointOdyssey train/val
   - base tracker checkpoint
   - DINO extractor
3. 输出每个 re-entry 样本的：
   - base position
   - GT position
   - support memory features
   - DINO search feature map
   - tracker visibility/confidence
   - occ length
4. 保证按 sequence 保存，便于 grouped bootstrap
5. 先做小样本 smoke，确认 tensor shape、坐标系、归一化一致

### Phase 2: Model

1. 新建 `models/online_recovery_head.py`
2. 第一版只实现：
   - support pooling
   - cosine similarity map
   - 轻量 conv refiner
   - heatmap output
3. 不要先做 transformer
4. 不要先做多任务联合损失

### Phase 3: Training

1. 新建训练脚本
2. 冻结：
   - CoTracker
   - DINO backbone
3. 只训练 recovery head
4. 主损失：
   - heatmap loss
   - position loss
5. 每个 epoch 保存：
   - mean / median / p95
   - better_frac
   - 按 occlusion length 分桶指标

### Phase 4: Evaluation

1. 离线评估 `anchor_t0`
2. 对比：
   - base
   - raw DINO cosine
   - learned recovery head
3. 只有 learned head 明显优于 raw cosine，才进入 online integration

### Phase 5: Online Integration

1. 把 learned head 接回 `models/cotracker_refiner.py`
2. 替换当前 DINO raw cosine readout
3. 保留现有 trigger / retracking skeleton
4. 重新跑：
   - real eval audit
   - anchor diagnosis
   - recovery position error

## 11. Final Recommendation

结论很明确：

可以开始训练，但必须换训练目标。

不该训练“当前 gate 能不能更会开门”；
该训练“门后面到底有没有一个能把点拉回正确位置的几何 head”。

如果你要的是“真正的恢复能力”，那就应该把训练预算投到这个新 head 上，而不是继续堆在已有 acceptor/verifier 上。
