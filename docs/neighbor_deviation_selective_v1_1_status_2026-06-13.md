# Neighbor-Deviation Selective Tracking v1.1 Status (2026-06-13)

## 三句话

1. `neighbor deviation` 的 **reliability signal 仍然成立**：overall AUC=`0.764`，LOOCV avg=`0.743`，Spearman=`0.341`。
2. `neighbor deviation` 的 **risk calibration 也成立**：raw-risk ECE=`0.0211`，Platt ECE=`0.0259`，Brier≈`0.058-0.060`。
3. 但 `v1` 文档里“Platt 后 selective AJ 明显提升”的说法 **不能保留**；原因是原脚本把 `p(high_error)` 当成了“高可靠性分数”去排序。

## 1. 本次修正了什么

修正文件：

- [eval_neighbor_deviation_selective.py](/gemini/code/FSPT/scripts/eval_neighbor_deviation_selective.py)

修正点：

- calibration 现在明确建模的是 `risk probability = p(high_error | score)`
- selective tracking 排序现在使用 `reliability = 1 - risk`
- `visibility` 被视为 reliability-oriented score
- `neighbor deviation` 被视为 risk-oriented score

这次修正 **不改变样本、不改变模型、不改变 LOOCV**，只修正评估方向。

## 2. Reliability Detection 仍然成立

来源：

- [baseline_summary.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/baseline_summary.json)

| Method | ROC-AUC | PR-AUC | Spearman |
|---|---:|---:|---:|
| visibility | 0.238 | 0.041 | -0.118 |
| statistics | 0.138 | 0.037 | -0.386 |
| reconstruction | 0.576 | 0.093 | -0.026 |
| **neighbor_deviation** | **0.764** | **0.219** | **0.341** |

解释：

- `neighbor deviation` 仍然是当前最强的 trajectory-only reliability signal。
- `visibility` 不适合直接拿来做“高误差检测”。
- `reconstruction` 依旧不是主信号，不能包装成主结论。

## 3. LOOCV 泛化结论不变

来源：

- [baseline_summary.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/baseline_summary.json)
- [trajectory_manifold_verifier_v1_loocv_status_2026-06-12.md](/gemini/code/FSPT/docs/trajectory_manifold_verifier_v1_loocv_status_2026-06-12.md)

关键数字：

- `neighbor_deviation pooled AUC = 0.764`
- `neighbor_deviation LOOCV avg = 0.743`
- `drop = 0.021`
- `learned head LOOCV avg = 0.793`

解释：

- 主结论仍应围绕 hand-crafted `neighbor deviation`。
- `learned head` 只能作为增强或 appendix，不能替代主故事。

## 4. Calibration 修正后口径

来源：

- [calibration_raw.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/calibration_raw.json)
- [calibration_platt.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/calibration_platt.json)
- [calibration_isotonic.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/calibration_isotonic.json)

注意：下面所有 calibration 都是在估计 `p(high_error)`，不是 `p(reliable)`。

| Score | Method | ECE | Brier |
|---|---|---:|---:|
| visibility | raw-risk | 0.1564 | 0.1242 |
| visibility | platt-risk | 0.0097 | 0.0580 |
| visibility | isotonic-risk | 0.0000 | 0.0564 |
| **neighbor deviation** | **raw-risk** | **0.0211** | **0.0579** |
| neighbor deviation | platt-risk | 0.0259 | 0.0597 |
| neighbor deviation | isotonic-risk | 0.0000 | 0.0543 |

解释：

- `neighbor deviation` 本身就是很强的 risk score，raw-risk 已经很好。
- 对 `neighbor deviation`，Platt 并没有继续显著改善 calibration。
- 对 `visibility`，Platt/Isotonic 有明显校准收益。
- 论文里不能再写“neighbor deviation 必须靠 Platt 才变可用”；更准确的说法是：
  `neighbor deviation` 天生就接近可校准风险分数，而 `visibility` 需要显式校准。

## 5. Selective Tracking 修正后结论

来源：

- [selective_metrics_overall.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/selective_metrics_overall.json)
- [selective_metrics_subsets.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/selective_metrics_subsets.json)

### Overall

| Score | AURC | Max Selective AJ |
|---|---:|---:|
| visibility_raw | 0.0254 | 0.0500 |
| visibility_platt | 0.0254 | 0.0500 |
| **neighbor_dev_raw** | **0.0260** | **0.0520** |
| **neighbor_dev_platt** | **0.0260** | **0.0520** |

### Long-occ

| Score | AURC | Max Selective AJ |
|---|---:|---:|
| visibility_raw | 0.1053 | 0.0970 |
| visibility_platt | 0.1053 | 0.0970 |
| **neighbor_dev_raw** | **0.0859** | **0.1440** |
| **neighbor_dev_platt** | **0.0859** | **0.1440** |

解释：

- `Platt` 不改变排序，因此不会凭空提升 selective ranking。
- 当前 selective 收益来自 `neighbor deviation` 的原始排序能力，不是来自 calibration 本身。
- calibration 的作用现在应定义为：
  1. 把分数变成可解释风险概率；
  2. 支持统一阈值与 coverage 控制；
  3. 为后续 conformal / abstention 协议做准备。

## 6. 需要撤回或收紧的表述

以下说法不应继续使用：

- “Platt 校准后 selective AJ 从 0.141 提升到 0.177”
- “calibration turns it into a usable selective tracking score” 这句话需要改写

更准确的版本应为：

- `neighbor deviation` 已经提供了有效的 selective ranking signal
- calibration 让它变成了可解释、可阈值化的 risk probability
- ranking gain 与 calibration gain 是两件事，不能混写

## 7. 当前可以保留的主线

当前最稳的主线是：

**trajectory-local neighbor deviation for reliability estimation, plus calibrated risk prediction for selective point tracking**

而不是：

- 新 recovery 模块
- DiReCT
- candidate reranking
- “Platt 本身提升了 selective ranking”

## 8. 当前最合理的论文口径

1. `neighbor deviation` 是可泛化的 reliability / risk signal。
2. 该信号在 LOOCV 下保持稳定。
3. 该信号支持 selective ranking，尤其在 `long_occ` 子集更强。
4. calibration 的价值在于把它变成风险概率，而不是改变排序。

## 工件

- [baseline_summary.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/baseline_summary.json)
- [calibration_raw.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/calibration_raw.json)
- [calibration_platt.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/calibration_platt.json)
- [calibration_isotonic.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/calibration_isotonic.json)
- [selective_metrics_overall.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/selective_metrics_overall.json)
- [selective_metrics_subsets.json](/gemini/code/FSPT/outputs/neighbor_deviation_selective_v1_1/selective_metrics_subsets.json)
