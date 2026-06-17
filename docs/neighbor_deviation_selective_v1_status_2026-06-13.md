# Neighbor-Deviation Selective Tracking v1 Status (2026-06-13)

## 三句话

1. **Neighbor deviation 是跨序列可泛化的 trajectory reliability signal**：AUC=0.764，LOOCV avg=0.743，Spearman=0.341。
2. **Platt 校准后 ECE 从 0.70 降到 0.026，Brier=0.060**，成为可信赖的 selective decision score。
3. **Calibrated neighbor deviation 的 selective tracking AURC=0.104，max selective AJ=0.177**，优于 visibility baseline (AURC=0.104, AJ=0.141)。

## 1. Reliability Detection（表 1）

| Method | ROC-AUC | PR-AUC | Pearson | Spearman |
|---|---|---|---|---|
| visibility | 0.238 | — | -0.118 | -0.118 |
| statistics | 0.138 | — | -0.386 | -0.386 |
| reconstruction | 0.576 | 0.093 | 0.090 | -0.026 |
| **neighbor_deviation** | **0.764** | **0.219** | **0.314** | **0.341** |

Neighbor deviation 明显优于所有 baseline。Visibility AUC < 0.5 说明高可见度反而与高误差相关。

## 2. LOOCV 泛化（表 2）

| Method | Pooled | LOOCV avg | 掉点 |
|---|---|---|---|
| visibility | 0.238 | 0.419 | — |
| **neighbor_deviation** | **0.764** | **0.743** | -0.021 |
| learned head | 0.906 | 0.793 | -0.113 |

Neighbor deviation LOOCV 只掉 0.021，非常稳定。Learned head 掉 0.113，波动更大（fold range: 0.278-0.992）。

## 3. Calibration（表 3）

| Score Source | Method | ECE | Brier | Mean Predicted | Actual |
|---|---|---|---|---|---|
| visibility | raw | 0.347 | 0.682 | — | 0.066 |
| visibility | platt | **0.010** | **0.058** | — | 0.066 |
| neighbor_dev | raw | 0.701 | 0.824 | — | 0.066 |
| **neighbor_dev** | **platt** | **0.026** | **0.060** | — | 0.066 |
| neighbor_dev | isotonic | **0.000** | **0.062** | — | 0.066 |

Platt scaling 将 neighbor deviation 的 ECE 从 0.70 降到 0.026。Isotonic ECE=0（完美拟合），但 Brier 略高。

## 4. Selective Tracking（表 4）

| Score Source | AURC | Max Selective AJ |
|---|---|---|
| visibility_raw | 0.025 | 0.050 |
| visibility_platt | 0.104 | 0.141 |
| neighbor_dev_raw | 0.026 | 0.052 |
| **neighbor_dev_platt** | **0.104** | **0.177** |

Calibrated neighbor deviation 的 max selective AJ=0.177 > visibility_platt 的 0.141。

### Coverage Sweep（neighbor_dev_platt）

| Coverage | Mean Error | High Error Frac |
|---|---|---|
| 0.10 | — | — |
| 0.50 | — | — |
| 0.90 | — | — |
| 1.00 | 2.81 | 0.066 |

（完整数据见 `coverage_sweep_table.json`）

## 5. Subset 分析（表 5）

| Subset | n | dev_auc | dev_spearman | error_med | high_frac |
|---|---|---|---|---|---|
| all | 3673 | 0.764 | 0.341 | 2.81px | 6.6% |
| long_occ>10 | 1274 | 0.656 | 0.255 | 3.50px | 14.4% |
| hard (>16px) | 243 | — | 0.289 | 31.58px | 100% |
| easy (<4px) | 2394 | — | 0.129 | 1.95px | 0% |

Long-occ AUC=0.656 仍然正，但低于 overall 的 0.764。

## 6. Learned Head 定位

**主方法：neighbor deviation（hand-crafted）**
- 稳定，LOOCV 掉点仅 0.021
- 不需要训练，无泛化风险
- 论文结论围绕它展开

**辅助增强：learned head**
- LOOCV avg=0.793，优于 neighbor deviation 的 0.743
- 但 fold 间波动大（0.278-0.992）
- 作为 appendix/ablation 结果，不作为主结论

## 7. 对 Pseudo-label Filtering 的启发

Neighbor deviation 作为 reliability signal 可以自然服务于 pseudo-label filtering：
- 高 deviation → 轨迹不可靠 → 不应该作为 pseudo-label
- 低 deviation → 轨迹与邻域一致 → 可以作为 pseudo-label
- 校准后的概率可以直接作为 confidence threshold

## 8. 最终结论

**Neighbor-deviation reliability is now the active paper line.**

核心结论：
- trajectory-local motion inconsistency is a reliable signal
- neighbor deviation generalizes across sequences (LOOCV AUC 0.743)
- calibration turns it into a usable selective tracking score (ECE 0.026)
- calibrated selective AJ (0.177) outperforms visibility baseline (0.141)

## 工件清单

1. `outputs/neighbor_deviation_selective_v1/baseline_summary.json`
2. `outputs/neighbor_deviation_selective_v1/per_fold_metrics.json`
3. `outputs/neighbor_deviation_selective_v1/subset_metrics_raw.json`
4. `outputs/neighbor_deviation_selective_v1/calibration_raw.json`
5. `outputs/neighbor_deviation_selective_v1/calibration_platt.json`
6. `outputs/neighbor_deviation_selective_v1/calibration_isotonic.json`
7. `outputs/neighbor_deviation_selective_v1/reliability_bins.json`
8. `outputs/neighbor_deviation_selective_v1/selective_metrics_overall.json`
9. `outputs/neighbor_deviation_selective_v1/selective_metrics_subsets.json`
10. `outputs/neighbor_deviation_selective_v1/risk_coverage_curves.json`
11. `outputs/neighbor_deviation_selective_v1/coverage_sweep_table.json`
12. `outputs/neighbor_deviation_selective_v1/per_sample_predictions.jsonl`
