# World-State High-Venue Upgrade Plan

## Goal

目标不是继续做零散实验，而是把当前项目升级成一个更强、更聚焦的论文：

**Causal Persistent Re-entry Tracking**

核心问题：

> 当一个点经历长遮挡、相机运动、甚至暂时离开视野后，系统能否在严格因果设置下恢复它的身份和位置？

这个问题和已有工作的边界如下：

- 不等于普通 2D point tracking
- 不等于离线 3D point tracking
- 不等于 post-hoc relocalization
- 不等于单帧 re-identification

它强调的是：

- `causal`
- `persistent identity`
- `re-entry recovery`
- `off-screen / camera-motion robustness`

---

## Why This Is The Right Upgrade

当前项目已经有三类证据：

1. `Stage 0` 证明 persistent world-state 的上限极高。
2. `Stage 1` 证明 predicted depth 下仍保留显著收益。
3. 当前 `Stage 2` 失败说明单帧 accept/select 不是正确 exploitation mechanism。

这三点组合起来，最合理的升级不是继续调参，而是：

- 重新定义任务
- 重新设计方法范式

也就是：

1. 先让 reviewer 接受“这确实是一个独立且重要的问题”。
2. 再用最有针对性的方法去解这个问题。

---

## Proposed Paper Structure

### Paper Thesis

现有 point trackers 即使拥有 3D cues，也没有系统解决 `causal re-entry recovery`：

- 需要在看不到未来的条件下
- 在长时间遮挡和相机运动后
- 恢复点的持久身份和位置

我们的核心观点：

**Persistent world-state is necessary but not sufficient; successful causal re-entry tracking requires temporal verification rather than single-frame matching.**

### Three Contributions

建议把贡献压缩成三条：

1. 定义 `causal persistent re-entry tracking` 任务和评估协议。
2. 系统证明 `world-state` 在该任务上具有巨大 oracle 和 noisy-depth 优势。
3. 提出并验证 `temporal verifier over candidate tracklets`，证明时间证据比单帧选择更适合利用 world-state headroom。

如果第 3 条最终做不成，就把贡献改为：

3. 系统失败分析，揭示为什么单帧 causal selection 不能稳定利用 world-state oracle gap。

---

## New Benchmark Layer

这是必须做的，不是可选项。

### Benchmark Name

内部先叫：

`Persistent Re-entry Tracking (PRT)`

正式命名可以后定，但必须有一个统一名字。

### Split Design

从现有数据构造两个层级：

1. `PRT-Synth`
   - 来源：`PointOdyssey`
   - 用于主分析、分层统计、oracle/noisy-depth/方法对比

2. `PRT-Real`
   - 来源：`RGB-Stacking`
   - 用于真实场景验证

### Query Definition

每个 query 必须满足：

1. 在 `t_q` 可见
2. 在 `t_q+1 ... t_r-1` 持续不可见或离开视野
3. 在 `t_r` 重新出现
4. `t_r` 是 first visible re-entry frame

### Required Stratification

必须报告以下分桶：

1. `occ_len >= {10,20,30,50}`
2. `camera_motion >= {0.1,0.3,0.5}`
3. `occluded but in-frame` vs `off-screen return`
4. `same-view` vs `large-view-change`

其中第 3 条很关键。

如果当前数据处理还没把 `off-screen` 单独分离出来，这应该优先补。

### Required Metrics

主指标：

- `reentry_median_px`
- `reentry_<4px`
- `reentry_<8px`

辅助指标：

- `better_frac_vs_baseline`
- `offscreen_retention`
- `camera-motion-stratified error`
- `occlusion-length-stratified error`
- `abstain rate`
- `coverage-error curve`

最后两项是为高 venue 准备的，因为它们把系统从“总要给答案”升级成“知道什么时候不能乱修正”。

---

## Method Upgrade: Temporal Verifier

### Why Single-Frame Selection Failed

现有失败已经足够说明问题：

1. 单帧 patch 没有足够信息区分好候选和坏候选。
2. DINO / scratch / hand-crafted feature 都没能稳定解决这个问题。
3. 但 oracle gap 一直存在，说明 candidate 本身不是完全没价值。

因此，下一步不能再问：

> “这一帧这个 candidate 看起来像不像？”

而要问：

> “这个 candidate 在 re-entry 后的短时间轨迹上，是否形成了自洽的 persistent state？”

### Minimal Temporal Verifier

输入：

1. baseline re-entry point
2. top-k causal candidates
3. re-entry 后 `K` 帧短窗口
4. world-state projection features
5. appearance features
6. visibility / confidence cues

每个 candidate 对应一个短 tracklet。

Verifier 输出：

1. candidate score
2. accept / reject
3. confidence

### Candidate Features

不要再只用单帧 patch feature。

每个 candidate 应聚合：

1. `temporal reprojection consistency`
   - world-state 在 `t_r ... t_r+K` 的 reprojection 是否平滑

2. `short-term motion smoothness`
   - candidate 接上 baseline 后是否产生不合理跳变

3. `temporal appearance consistency`
   - re-entry 后多帧 patch 与 query / visible history 是否持续一致

4. `visibility evolution`
   - 可见性是否随时间演化得合理

5. `uncertainty`
   - 如果所有 candidate 都不稳定，应该 abstain 而不是强行选

### Training Target

目标不要再做脆弱的 pairwise ranking 作为唯一监督。

建议：

1. `multi-class candidate classification`
   - baseline + top-k candidates + abstain

2. `margin loss`
   - 好 candidate 必须显著优于坏 candidate

3. `coverage-aware calibration loss`
   - 让 confidence 真正可用于 abstention

如果要进一步简化，第一版可以先做：

- baseline vs best candidate vs abstain 的 3 类分类

---

## Exact Decision Rule For This Direction

Temporal verifier 值不值得继续，按下面标准判定。

### Success Condition

在 hardest split 上同时满足：

1. `reentry_median_px` 明显低于 baseline
2. `<4px` 明显高于 baseline
3. `better_frac_vs_baseline > 50%`
4. 优于所有单帧 selector / acceptor ablation

如果还加入 abstention，则要求：

5. coverage 降低后 accuracy 单调提高

### Failure Condition

如果 temporal verifier 仍然不能做到：

- 稳定利用 oracle gap
- better_frac 超过 50%
- 在 hardest split 上带来清晰收益

那就停止方法开发，不再追加小模型试验。

此时项目直接收敛为：

- benchmark + oracle/noisy-depth evidence + failure analysis

### Current Status Update

最新无泄漏实验已经说明：

- `lightweight temporal/statistical verifier + MLP acceptor` 失败
- 虽然 val cache 上仍存在 oracle gap：
  - `baseline median = 37.97`
  - `oracle median = 31.33`
- 但 clean verifier best epoch 只能到：
  - `pred median = 40.71`
  - `better_frac = 1%`
- `sequence-level verifier` 进一步提升到了：
  - `pred median = 39.66`
  - `better_frac = 11%`
  但仍未超过 baseline

因此：

- 时间维度本身没有被证伪
- 但“candidate generation + short-window verifier”这一整套当前实现还没有落成有效方法
- 继续在这个机制上做小修小补的价值已经很低
- 后续如果还做方法，应该是完全不同的主轴，而不是继续在当前 verifier 家族里迭代

---

## What To Stop Doing Immediately

以下方向应立即停止：

1. 单帧 top1 acceptor
2. 单帧 multiclass selector
3. scratch patch ranker 微调
4. frozen DINO 单帧 accept/reject 微调
5. 继续围绕 50-100 个样本做反复试错
6. 继续围绕 lightweight temporal-statistics acceptor 调 threshold / hidden dim / lr

这些方向已经给出足够一致的失败信号。

继续投入只会消耗时间，不会抬高论文档次。

---

## 10-Day Execution Plan

### Day 1-2: Formalize Benchmark

产出：

- `docs/prt_benchmark_definition.md`
- `scripts/build_prt_splits.py`
- `outputs/prt_split_stats.json`

必须完成：

- 明确定义 re-entry query
- 统计各 strata 样本量
- 分离 off-screen vs in-frame occlusion

### Day 3-4: Build Temporal Candidate Dataset

产出：

- `scripts/build_prt_candidate_dataset.py`
- 训练/验证样本缓存

每条样本包含：

- baseline candidate
- top-k causal candidates
- re-entry 后短窗口 patch / feature
- GT best choice / abstain label

### Day 5-7: Train Minimal Temporal Verifier

第一版只追求最小可行：

- 简单 temporal encoder
- baseline + top-k + abstain 分类
- hardest split 上做验证

### Day 8-10: Decide

只有两种结果：

1. 如果稳定起效：
   - 继续扩到完整实验
   - 开始论文主表

2. 如果仍不起效：
   - 停止方法开发
   - 收敛为 benchmark / analysis paper

当前状态相当于已经走到了第 2 种情况。

如果还想继续冲更高，必须显式升级为：

- 完全不同的方法主轴
- 而不是继续沿用当前 candidate-verifier 家族

---

## Venue Expectation

### If Only Stage 0 + Stage 1 + Benchmark + Failure Analysis

更现实的目标：

- `WACV`
- `ACCV`
- `BMVC`
- `TCSVT`
- `Pattern Recognition`

### If Temporal Verifier Works Clearly

才有资格考虑更高层级：

- `ECCV`
- `ICCV`
- `CVPR`

前提是：

- benchmark/task framing 清晰
- 方法问题定义准确
- hardest split 上有稳定收益

现在距离这个目标，还差的不是更多调参，而是：

- 正确的问题切法
- 正确的方法范式

---

## Final Recommendation

接下来不要再问“还能不能再试一个 selector / acceptor 版本”。

应该只问两个问题：

1. 我们能不能把 `causal persistent re-entry tracking` 这个问题定义成一个 reviewer 认可的新任务层？
2. 我们能不能用 `temporal verification` 而不是单帧判断，第一次稳定利用 world-state headroom？

如果两个问题都做到了，这条线就有冲更高的可能。

如果第二个没做到，第一个也依然值得保住，因为当前 `Stage 0/1` 的证据已经足够强，足以支撑一篇高质量 benchmark / analysis 论文。
