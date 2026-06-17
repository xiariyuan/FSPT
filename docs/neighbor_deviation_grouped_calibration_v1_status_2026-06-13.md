# Neighbor-Deviation Grouped Calibration v1 Status (2026-06-13)

## 三句话

1. **Held-out grouped calibration 支持主线继续**：`neighbor_deviation raw-risk` 在按视频留一评估下依然稳定，overall `ECE=0.0230`，`Brier=0.0600`，`AUC=0.7639`。
2. **对 neighbor deviation 再做 Platt / Isotonic 没有收益**：overall 上 `Platt ECE=0.0201` 仅微弱变化，但 `AUC` 掉到 `0.5347`；`long_occ` 上也不如 raw-risk。
3. **Calibration 应该分两类讲**：`visibility` 需要显式校准；`neighbor deviation` 天然就是强 risk score，不需要再学一个 calibrator。

## 1. 设置

脚本：

- [eval_neighbor_deviation_grouped_calibration.py](/gemini/code/FSPT/scripts/eval_neighbor_deviation_grouped_calibration.py)

输入：

- [per_sample_predictions.jsonl](/gemini/code/FSPT/outputs/trajectory_manifold_verifier_v1/per_sample_predictions.jsonl)

输出：

- [summary.json](/gemini/code/FSPT/outputs/neighbor_deviation_grouped_calibration_v1/summary.json)
- [per_video_metrics.json](/gemini/code/FSPT/outputs/neighbor_deviation_grouped_calibration_v1/per_video_metrics.json)
- [per_sample_predictions.jsonl](/gemini/code/FSPT/outputs/neighbor_deviation_grouped_calibration_v1/per_sample_predictions.jsonl)

协议：

- 按 `video_name` 留一
- train videos 上拟合 calibrator
- held-out video 上评估
- calibration 目标是 `p(high_error)`

数据规模：

- `n_valid = 3673`
- `n_sequences = 22`
- `videos_with_high_error = 17`

## 2. Overall Held-out 结果

来源：

- [summary.json](/gemini/code/FSPT/outputs/neighbor_deviation_grouped_calibration_v1/summary.json)

| Score | ECE | Brier | AUC |
|---|---:|---:|---:|
| visibility_raw_risk | 0.1570 | 0.1242 | 0.7617 |
| visibility_platt_risk | 0.0149 | 0.0603 | 0.6977 |
| visibility_isotonic_risk | 0.0437 | 0.0620 | 0.6712 |
| **neighbor_deviation_raw_risk** | **0.0230** | **0.0600** | **0.7639** |
| neighbor_deviation_platt_risk | 0.0201 | 0.0613 | 0.5347 |
| neighbor_deviation_isotonic_risk | 0.0390 | 0.0609 | 0.6571 |

关键观察：

- `neighbor_deviation raw-risk` 是当前最好的 overall held-out 结果。
- `visibility` 的 raw-risk 很差，但 Platt 后确实被校准好了。
- `neighbor deviation` 不需要再学一个 calibrator；Platt 只把均值拉准了，但牺牲了区分性。

## 3. Long-occ Held-out 结果

来源：

- [summary.json](/gemini/code/FSPT/outputs/neighbor_deviation_grouped_calibration_v1/summary.json)

| Score | ECE | Brier | AUC |
|---|---:|---:|---:|
| visibility_raw_risk | 0.3936 | 0.3042 | 0.5938 |
| visibility_platt_risk | 0.0587 | 0.1286 | 0.4888 |
| visibility_isotonic_risk | 0.1182 | 0.1334 | 0.4003 |
| **neighbor_deviation_raw_risk** | **0.0731** | **0.1296** | **0.6523** |
| neighbor_deviation_platt_risk | 0.0752 | 0.1297 | 0.4641 |
| neighbor_deviation_isotonic_risk | 0.1061 | 0.1324 | 0.5530 |

关键观察：

- 在 `long_occ` 子集上，`neighbor_deviation raw-risk` 依旧是最强方案。
- `visibility` 的 calibration 虽然让 ECE 看起来更好，但 AUC 会进一步下滑。
- 对 `neighbor deviation`，显式校准既没有改善 calibration，也伤害了 discrimination。

## 4. 这说明什么

### 4.1 对 visibility

- `visibility` 是一个 **需要校准** 的 reliability proxy。
- 它 raw 时偏差大，但显式校准后可变成可解释风险概率。

### 4.2 对 neighbor deviation

- `neighbor deviation` 不是“需要再加工才能用”的弱分数。
- 它本身就是一个 **天然接近 calibrated risk 的强分数**。
- 再加 Platt / Isotonic 反而会引入跨视频尺度不稳定性。

### 4.3 对论文叙事

应该讲：

- `neighbor deviation` 自身就提供了可泛化 risk signal
- 显式 calibration 是协议工具，不是它的性能来源
- 对某些 baseline（例如 visibility），calibration 是必要的
- 对我们的主分数（neighbor deviation），raw-risk 已经足够强

不应该讲：

- “Platt 让 neighbor deviation 变得可用”
- “所有 reliability score 都应该先学一个 calibrator”

## 5. 为什么 Platt 会伤害 neighbor deviation

这是当前最合理的解释：

1. `neighbor deviation` 的原始单调关系已经足够好。
2. 按视频留一时，不同视频的尺度和正负比例差异较大。
3. 每个 fold 单独学的 logistic 映射会改变跨视频概率尺度的一致性。
4. pooled ECE 可能接近，但 pooled AUC 会被这种跨视频映射扰乱。

因此：

- **raw-risk 保留排序与可解释性**
- **learned calibrator 只在弱分数上更有价值**

## 6. 当前主线如何更新

最新、最稳的版本应当是：

**neighbor deviation is a trajectory-local risk signal that is already well calibrated enough in raw form and generalizes across held-out videos.**

进一步展开：

1. overall reliability 成立
2. long-occ selective 场景更强
3. raw-risk held-out calibration 成立
4. visibility 需要显式校准，neighbor deviation 不需要

## 7. 对后续任务的影响

### 保留

- 继续做 `long_occ` 深挖
- 继续做 failure cases
- 继续做 pseudo-label filtering 预备实验

### 收紧

- 不再把 “Platt calibration” 作为主贡献
- 不再把 “calibrated score improved ranking” 作为卖点

### 新的更准确说法

- `neighbor deviation raw-risk` is the main result
- calibration is primarily for protocol / interpretability
- selective gain should be reported mainly on `long_occ`

## 8. 最终判断

**Grouped calibration v1: PASS, but only for the raw neighbor-deviation risk story.**

更具体地说：

- `neighbor deviation raw-risk`：继续保留为主线结果
- `visibility + Platt`：作为 baseline 校准示例保留
- `neighbor deviation + Platt/Isotonic`：降级为负结果或 appendix
