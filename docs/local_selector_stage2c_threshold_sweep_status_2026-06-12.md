# Local Selector Stage-2c Threshold Sweep Status (2026-06-12)

## 结论

`gate threshold` sweep 已完成。

结论不是“selector 还差一点阈值就能过线”，而是：

1. 当前 `gate + ranker` 在这份 val 集上已经达到 `base>16 better_2px` 的 **oracle ceiling**
2. 因此继续只调 `gate_threshold`，**不可能**把 `base>16 better_2px` 从 `0.526` 推到 `0.55+`
3. 下一步瓶颈不再是 gate，而是：
   - 评估门槛定义不合理
   - positive events 上的 fine ranking 仍然偏弱

## 这次修正了什么

更新脚本：

- `scripts/train_local_selector_v2.py`

新增能力：

1. 区分两种 oracle 口径
   - `oracle_candidate_median`: 只看候选池中最优候选的误差
   - `oracle_selective_median`: 如果没有更优候选则保留 base 的 selective oracle
2. `gate_threshold` sweep
3. per-sample 导出
4. 自动给出推荐 operating point

输出工件：

- `outputs/local_selector_stage2b_from_anchor_sweep/results.json`
- `outputs/local_selector_stage2b_from_anchor_sweep/threshold_sweep.json`
- `outputs/local_selector_stage2b_from_anchor_sweep/per_sample_gate_0.50.jsonl`

## 为什么旧口径会误导

之前看到：

- overall `base_median = 11.23`
- overall `oracle_median = 33.04`

这看起来像“oracle 比 base 还差”，容易误判为方法方向不成立。

但这里的 `33.04` 实际上是 **candidate-only oracle**：

- 对 easy samples，候选池本来就不该替代 base
- 如果强行只看候选误差，中位数会被 easy samples 拉坏

更合理的 selective oracle 是：

- overall `oracle_selective_median = 8.14`

这才反映真实 headroom：

- easy samples 保留 base
- hard positive samples 才切到候选

## 主要结果

默认阈值 `gate_threshold = 0.50`：

| Metric | Overall | Positive | base>16 | base>32 |
|---|---:|---:|---:|---:|
| N | 97 | 20 | 38 | 19 |
| base median | 11.23 | 35.97 | 32.47 | 78.94 |
| top1 median | 34.98 | 26.16 | 51.01 | 66.88 |
| oracle candidate median | 33.04 | 10.67 | 48.10 | 60.55 |
| oracle selective median | 8.14 | 10.67 | 21.92 | 48.36 |
| final median | **11.14** | **20.33** | **22.87** | **62.69** |
| final better\_2px | **0.206** | **1.000** | **0.526** | **0.632** |
| oracle better\_2px | 0.206 | 1.000 | 0.526 | 0.632 |

辅助指标：

- gate accept rate: `0.299`
- gate positive recall: `1.000`
- ranker top1 accuracy on positive events: `0.250`

## Threshold Sweep 结论

推荐阈值仍然是：

- `gate_threshold = 0.50`

原因：

1. `0.45` 和 `0.50` 基本等价
2. 更低阈值会显著伤 overall
3. 更高阈值会掉 hard recall

关键 sweep 结果：

| threshold | overall diff | base>16 better\_2px | base>16 median imp | accept rate | 判定 |
|---|---:|---:|---:|---:|---|
| 0.30 | +9.39px | 0.526 | 24.9% | 0.649 | overall 不可用 |
| 0.40 | +2.86px | 0.526 | 28.6% | 0.454 | overall 仍偏伤 |
| 0.45 | -0.09px | 0.526 | 29.6% | 0.320 | 最优区间 |
| 0.50 | -0.09px | 0.526 | 29.6% | 0.299 | 最优区间 |
| 0.55 | -0.09px | 0.368 | 15.0% | 0.206 | 开始漏 positive |
| 0.60 | -0.09px | 0.316 | 13.0% | 0.186 | 继续掉 recall |

最重要的事实：

- `base>16 final better_2px` 的最高值是 `0.526`
- `base>16 oracle better_2px` 也是 `0.526`

这说明：

- 当前 val 集上，`0.55` 这条线高于 oracle ceiling
- 不是 threshold 没调好
- 也不是 gate 没学会
- 是这份 val 集本身只有 `20/38` 个 `base>16` 样本存在可恢复候选

## Per-sample 分析

### 1. gate 不是当前主瓶颈

在 `gate_threshold = 0.50` 下：

- `all positive n = 20`
- `gate_accept = 20`
- `chosen positive = 20`

也就是说：

- 所有真实 positive events 都被 gate 放过
- 所有 positive events 最终都实现了 `>2px` 改善

因此当前 `better_2px` 指标已经被 **候选池正样本比例** 卡住了。

### 2. ranker 仍然偏弱，但它伤的是“改善幅度”，不是“是否改善”

在 positive events 上：

- ranker oracle accuracy 只有 `0.25`
- 但 final `better_2px = 1.0`

这说明 ranker 常常没选到 oracle candidate，但仍选到了一个“足够比 base 好”的 candidate。

典型现象：

- `dance-twirl` 多个样本里
  - oracle error 在 `3.8 ~ 6.8px`
  - ranker 常选到 `13 ~ 19px`
  - 依然明显优于 base `29 ~ 32px`

所以当前 ranker 的问题是：

- **fine ranking 不够好**
- 但不是完全选错方向

### 3. overall no-harm 的剩余风险来自少量 false accept

在 `gate_threshold = 0.50` 下，主要 false accept 只剩少数：

- `pigs`: `15.14 -> 20.62`
- `parkour`: `15.96 -> 16.94`

这类样本说明：

- gate 边界附近仍会放过少量 hard-negative / near-threshold events
- 但数量已经很少，不再主导整体结论

## 对 Stage-2b 判定的修正

原先的三条标准里：

1. `base>16 final better_2px >= 0.55`
2. `base>16 median improvement >= 10%`
3. `overall final median diff <= 1px`

其中第 1 条在这份 val 上不适合作为绝对门槛，因为：

- 当前 `oracle better_2px = 0.526`
- 方法已经达到这条 ceiling

更合理的验收方式应改成下面二选一：

1. 绝对门槛：
   - `base>16 final better_2px >= 0.50`
   - `base>16 median improvement >= 20%`
   - `overall diff <= 1px`
2. 或 oracle-normalized 门槛：
   - `base>16 final better_2px / oracle better_2px >= 0.9`
   - `base>16 final median <= oracle_selective_median + 2px`
   - `overall diff <= 1px`

## 当前最准确的状态判断

当前 online local selector 路线的状态应描述为：

- `candidate generation`：成立
- `gate`：基本成立
- `better_2px` recovery：在当前 val 上已到 oracle ceiling
- `fine ranking`：仍偏弱
- `online integration readiness`：**接近可用，但还不应正式接入主线上线训练**

## 下一步

不要再继续只做 threshold tuning。下一步应改成：

1. 重写验收标准
   - 用 `oracle-normalized` 标准替换硬编码 `0.55`
2. 做 positive-event fine ranking 改进
   - 目标不是提高 `better_2px`
   - 而是把 `positive final median` 从 `20.33` 继续压向 `oracle 10.67`
3. 若要决定是否接 online
   - 先做一次小规模 integration smoke
   - 只在 `gate_prob >= 0.45/0.50` 的 operating point 上测总体 no-harm

