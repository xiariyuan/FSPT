# World-State Paper Strategy (2026-05-30)

## 1. Executive Summary

当前项目最可靠的结论，不是“我们已经做出了一个更强的 point tracker”，而是下面三点：

1. `persistent world-state` 对长遮挡重现 (`re-entry`) 确实有巨大信息优势。
2. 这个优势在使用预测深度时仍然保留，虽然被噪声明显削弱。
3. 当前 `Stage 2` 的 learnable causal selector / ranker / acceptor 还没有形成稳定、可发表的方法结果。

这意味着：

- 如果现在收敛，可以写成一篇强分析/benchmark/diagnostic 论文。
- 如果想冲更高，不能继续在现有单帧 selector/acceptor 上调参，必须换成更强的问题定义和方法范式。

---

## 2. What Is Safely True

### 2.1 Stage 0: GT world-state oracle is very strong

文件：

- `outputs/stage0_world_state_conclusion.md`
- `outputs/stage0_world_state_oracle.json`

最关键子集：`occ20_cam0.30`

- `2D hold median = 186.29 px`
- `3D hold median = 7.00 px`
- `3D hold <4px = 47.4%`
- `2D hold <4px = 0.2%`

结论：

- 在长遮挡 + 明显相机运动下，世界状态表示对重入恢复有压倒性优势。
- 这个结果不是小幅提升，而是数量级差距。
- 这说明“点应被建模为持久状态，而非纯帧局部 2D correspondence”这个大方向是成立的。

### 2.2 Stage 1: predicted depth still preserves substantial gain

文件：

- `outputs/stage1_predicted_depth.json`
- `outputs/stage1_predicted_depth.log`

最关键子集：`occ20_cam0.30`, depth noise `sigma=0.15`

- `2D hold median = 186.29 px`
- `Pred-depth 3D hold median = 37.16 px`
- `Pred-depth 3D hold <4px = 13.0%`
- `2D hold <4px = 0.2%`

结论：

- 即使深度有明显噪声，world-state 仍然比 2D hold 强约 `5x`。
- `Stage 0` 不是“只有 GT 才成立”的空想结果。
- 真实可学习系统仍然有空间，只是这个空间没有自动转化成当前 `Stage 2` 的方法成功。

### 2.3 Stage 2 causal search has signal, but not a reliable method

文件：

- `outputs/world_state_stage2_causal_dino_smoke64_v2.json`

64-sample causal DINO smoke：

- `baseline median = 38.52 px`
- `pred median = 36.91 px`
- `better_frac = 39.1%`
- `baseline <4px = 3.1%`
- `pred <4px = 4.7%`

结论：

- 不是完全没有信号。
- 但信号还远没有强到能支撑一篇方法论文。
- 当前更像是“candidate generation 有时能帮忙”，而不是“已经找到稳定可学习的恢复机制”。

---

## 3. What Is Not Safe To Claim

### 3.1 Early Stage 2 learnable results cannot be used as main evidence

原因已经明确：

- 早期 `Stage 2` 特征里混入了未来信息和 GT 派生信息。
- 包括：
  - `reentry_xy`
  - GT re-entry point crop
  - `history_world` / `history_world_delta` 直接来自 GT `trajs_3d`

相关文件：

- `experiments/world_state_dataset.py`
- `scripts/train_world_state_stage2.py`

因此：

- 早期 scratch / DINO Stage 2 的正结果，不能作为论文主结果。
- 它们只能作为内部探索记录，不能出现在最终 claim 中。

### 3.2 Current acceptor / selector / ranker line is not a publishable method

已验证失败的路线：

- `outputs/world_state_acceptor_v2/metrics.json`
- `outputs/world_state_selector_v1/metrics.json`
- `outputs/world_state_dino_selector_v2/metrics.json`
- `outputs/world_state_dino_top1_acceptor_v2/metrics.json`

代表性现象：

1. `selector_v1`
   - `baseline median = 21.79`
   - `oracle median = 14.43`
   - learned `pred median = 30.32`
   - `accept_rate = 1.0`
   - 说明模型会选候选，但经常选错。

2. `dino_selector_v2`
   - `baseline median = 41.98`
   - `oracle median = 33.14`
   - learned `pred median = 41.98 ~ 42.22`
   - `choose_baseline_frac = 0.80 ~ 0.88`
   - 说明模型学成“几乎总回退 baseline”。

3. `dino_top1_acceptor_v2`
   - `baseline median = 41.98`
   - `cand0 median = 50.22`
   - `cand0 better_frac = 41.7%`
   - learned `accept_rate = 0`
   - 说明 top1 candidate 偶尔更好，但当前因果特征不足以判断何时该接受。

结论：

- oracle gap 存在，但当前单帧特征无法稳定利用它。
- 再继续在这个范式上小修小补，投入产出比很差。

---

## 4. Scientific Meaning Of The Current Results

当前结果其实已经给出了一个很清楚的科学结论：

1. 问题不是“3D / world-state 没用”。
2. 问题是“如何在 causal monocular setting 下，把这个优势转化成可学习、可鲁棒利用的恢复机制”。
3. 当前失败说明，单帧 patch-level acceptance 不是正确答案。

换句话说：

`Stage 0/1` 证明了空间存在。
`Stage 2` 证明了 naive exploitation strategy 不够。

这本身就是一个很像论文的问题结构：

- 有大 oracle gap / representation gap
- 有 realistic noisy regime
- 有失败的简单方法
- 因此需要更正确的方法设计

---

## 5. Why The Current Story Is Not Enough For A Higher Venue

如果只拿现在的 `Stage 0 + Stage 1` 去写，论文会成立，但上限有限。

原因不是结果弱，而是竞争位置已经变化：

1. `TAPIP3D` 已经明确提出 persistent 3D geometry for point tracking，并利用深度与相机运动把特征提升到世界空间。
2. `SpatialTrackerV2` 已经把 monocular depth + pose + motion 做成了 feed-forward 3D point tracker。
3. `PointSt3R` 已经表明 3D-grounded correspondence 能作为 point tracking 的强路线。
4. `TAPVid-360` 又进一步把问题推进到 points outside FoV 的 allocentric direction tracking。

因此，如果我们只说：

- “world-state 比 2D 强”
- “预测深度下仍然有收益”

这不足以支撑更高层级 venue。

我们必须把贡献升级成下面两种之一：

1. 定义一个现有工作还没有打透的问题。
2. 给出一个现有工作没有解决好的因果恢复机制。

---

## 6. Recommended High-Venue Reframe

### 6.1 One Main Line Only

建议只保留一条升级主线：

**Causal Persistent Point Tracking under Occlusion, Camera Motion, and Off-Screen Re-entry**

不要再把主线表述为：

- 更好的 post-hoc refinement
- 更好的 selector
- 更好的 patch acceptor

而要表述为：

- 现有 2D/3D trackers 虽然能做 tracking，但对 `causal re-entry recovery` 的能力并没有被单独定义和系统评估。
- 尤其是在 `long occlusion + camera motion + off-screen return` 的组合场景中，单帧视觉匹配不足以稳定恢复 point identity。
- 需要的是 `state propagation + uncertainty + temporal verification`，而不是单帧候选打分。

### 6.2 Why This Reframe Is Defensible

这个改写是合理的，因为当前结果正好支持它：

1. `Stage 0/1` 证明 persistent state 是有价值的。
2. 当前单帧 selector/acceptor 失败，说明关键瓶颈不是 state representation 本身，而是 `causal recovery mechanism`。
3. `TAPVid-360` 表明社区已经开始重视 beyond-FoV / allocentric 问题，但它关注的是方向 tracking；我们这里的证据更集中在 `re-entry recovery`。

这个位置比“再做一个 3D tracker”更干净。

---

## 7. The Only Two High-Leverage Additions Worth Doing

如果要继续冲更高，我只建议追加下面两件事。

### 7.1 Addition A: Build a benchmark/task layer, not just a method layer

要做的不是再跑更多小模型，而是把任务定义清楚：

- 从 `PointOdyssey` 构建 `causal re-entry` benchmark split
- 从 `RGB-Stacking` 构建真实场景 application split
- 统一报告：
  - `reentry_median_px`
  - `reentry_<4px`
  - `offscreen_retention`
  - `camera-motion-stratified`
  - `occlusion-length-stratified`

必要性：

- 没有这个 benchmark，论文会被 reviewer 视为“分析很好，但任务边界不清楚”。
- 有了这个 benchmark，即使方法还在早期，论文也有更强的独立价值。

这一步本质上是把现有强结果从“现象”升级为“问题定义”。

### 7.2 Addition B: Replace single-frame selection with temporal verification

下一条方法线不应该再是：

- top1 acceptor
- multiclass selector
- pairwise patch ranker

而应该是：

**multi-frame temporal verifier over candidate tracklets**

最小可行定义：

1. 用 causal search 产生少量候选 re-entry points。
2. 不在单帧上决定接受谁。
3. 在 `t, t+1, ..., t+K` 的短窗口上评估候选 tracklet：
   - world-state reprojection consistency
   - short-term motion smoothness
   - visibility evolution consistency
   - appearance consistency across the post-reentry window
4. 输出：
   - best candidate
   - confidence / abstain score
5. 如果 confidence 低，则回退 baseline。

为什么只推荐这个方向：

- 当前所有失败都说明“单帧信息不够”。
- oracle gap 说明 candidate 有时是好的。
- 因此最合理的下一步不是换 optimizer，而是引入时间维上的证据聚合。

---

## 8. Concrete Experimental Decision Rule

后面不要再“边试边想”，而要按下面的停机准则推进。

### 8.1 If benchmark layer is completed, but temporal verifier still fails

那么论文应收敛为：

- 新任务/新分层评估
- world-state oracle and noisy-depth evidence
- 对 causal re-entry recovery 的 failure analysis

这条线仍然能形成一篇完整论文，但更像：

- strong benchmark / analysis / diagnostic paper
- 目标更适合 `WACV / ACCV / BMVC / TCSVT / Pattern Recognition`

### 8.2 If temporal verifier gives stable gains on the new benchmark

通过标准建议定为：

- 比 baseline 在 hardest re-entry split 上稳定降低 `reentry_median`
- 同时提升 `<4px`
- 且不显著伤害 easy split
- 至少优于单帧 selector / acceptor ablation

那么才值得继续冲更高。

这种情况下，论文叙事会变成：

- 现有 tracker 缺的不是 3D state，而是 causal re-entry verification
- 我们定义任务并提出 temporal verifier
- 在 hardest persistent subsets 上取得稳定收益

这才有机会往更高 venue 走。

---

## 9. Practical Recommendation For The Next Round

基于现在的证据，建议的执行顺序是：

1. 先整理成正式 paper package
   - 主结论
   - strongest tables
   - 无效结果和原因
   - 任务定义

2. 不再启动新的单帧 selector / ranker / acceptor 调参

3. 只做两件追加工作
   - `benchmark formalization`
   - `temporal verifier`

4. 在 benchmark formalization 完成前，不要继续谈“冲顶会方法效果”
   - 因为当前最大的缺口已经不是再降几个像素
   - 而是把问题边界定义成 reviewer 会认的形式

---

## 10. Bottom Line

最保守但准确的判断是：

- 当前工作已经证明了一个强 insight：`persistent world-state` 对长遮挡重入恢复有决定性价值。
- 但当前还没有证明一个强方法：如何在 causal monocular setting 下稳定利用这个价值。

截至当前最新结果，还应补一句：

- `lightweight temporal verifier` 也已经在无泄漏条件下失败，因此“简单 temporal statistics + acceptor”不是可行答案。
- `sequence-level verifier` 虽然比轻量版更强，但仍未超过 baseline，因此当前 `causal candidate generation + short-window verification` 机制还没有形成可发表的方法结果。

如果只想尽快收敛，应该写成强分析/benchmark论文。

如果要冲更高，唯一值得继续投资源的路线是：

**把项目升级成“causal persistent re-entry tracking”问题定义 + temporal verifier 方法，而不是继续做单帧候选选择。**
