# Neighbor-Deviation Deep Analysis v1.1 Status (2026-06-13)

## 三句话

1. 原始 `deep_analysis_v1` 不能直接用于论文主表，因为它把 `visibility` 在 coverage 表中反向排序了，也把原始 `neighbor deviation` 数值直接当成了 risk probability。
2. 修正后，`neighbor deviation` 仍然优于 `visibility`，但优势比 Claude 那版小得多：50% coverage 下 mean error `3.50 vs 4.05`，90% coverage 下 `5.32 vs 5.58`。
3. 最稳的新增结论是：`smooth-but-wrong` 仍然极罕见，仅 `10/3673 = 0.27%`；`learned head` 仍然只能放 appendix；pseudo-label filtering 目前还不够强。

## 1. 这次修正了什么

脚本：

- [analyze_neighbor_deviation_deep.py](/gemini/code/FSPT/scripts/analyze_neighbor_deviation_deep.py)

修正点：

- coverage-driven selective 表现在按正确方向使用 `visibility_score`
  - `higher visibility = more reliable`
- risk-threshold sweep 现在基于 held-out `neighbor_deviation_raw_risk`
  - 不再拿原始 `neighbor_deviation` 数值直接当概率阈值
- `vis_AUC` 统一改成 `vis_risk_auc`
  - 即对 `-visibility_score` 做高误差检测

修正后输出目录：

- `outputs/neighbor_deviation_deep_analysis_v1_1/`

## 2. 修正后的 Coverage-Driven 表

来源：

- [threshold_sweep_by_coverage.json](/gemini/code/FSPT/outputs/neighbor_deviation_deep_analysis_v1_1/threshold_sweep_by_coverage.json)

| Coverage | dev_mean_err | dev_high% | vis_mean_err | vis_high% |
|---|---:|---:|---:|---:|
| 0.10 | 3.93 | 1.9% | 4.02 | 1.9% |
| 0.20 | 3.18 | 1.4% | 3.87 | 1.6% |
| 0.30 | 3.20 | 2.2% | 4.10 | 1.9% |
| 0.50 | 3.50 | 2.2% | 4.05 | 1.7% |
| 0.70 | 4.32 | 3.0% | 4.46 | 2.6% |
| 0.90 | 5.32 | 4.7% | 5.58 | 5.1% |
| 1.00 | 6.58 | 6.6% | 6.58 | 6.6% |

正确解读：

- `neighbor deviation` 在 **mean error** 上整体更好，尤其 `0.2-0.9` coverage。
- 但它并不是像旧版文档那样“全面碾压” `visibility`。
- 在 `50% coverage` 下：
  - mean error: `3.50 vs 4.05`
  - high-error rate: `2.2% vs 1.7%`
- 所以 50% coverage 不能写成 “5x better high-error rate”。

更稳的说法应是：

- `neighbor deviation` gives a better error-coverage tradeoff in mean pixel error
- high-error fraction advantage appears mainly at higher coverage such as `0.9`

## 3. Occ-Length Bins 修正后仍然支持主线

来源：

- [long_occ_bins.json](/gemini/code/FSPT/outputs/neighbor_deviation_deep_analysis_v1_1/long_occ_bins.json)

| Occ Range | n | high_frac | dev_AUC | vis_risk_AUC |
|---|---:|---:|---:|---:|
| 0-5 | 1895 | 2.0% | 0.684 | 0.533 |
| 5-10 | 411 | 3.6% | 0.943 | 0.837 |
| 10-20 | 485 | 12.4% | 0.675 | 0.614 |
| 20-50 | 756 | 13.8% | 0.647 | 0.593 |
| 50-300 | 126 | 21.4% | 0.744 | 0.610 |

这部分仍然是可用结果，但口径也要收紧：

- `neighbor deviation` 在所有遮挡段都优于 `visibility risk`
- 但 `5-10` 段的差距是 `0.943 vs 0.837`
  - 不是旧版写的 `0.943 vs 0.163`
- 因此可以讲“短到中等遮挡段最强”，但不要夸成断层式领先

## 4. Failure Taxonomy 仍然成立

来源：

- [failure_cases.json](/gemini/code/FSPT/outputs/neighbor_deviation_deep_analysis_v1_1/failure_cases.json)

| 类型 | 数量 |
|---|---:|
| smooth_but_wrong | 10 |
| false_safe | 167 |
| high_dev_accurate | 441 |
| neighbor_drift | 203 |

可保留的结论：

- `smooth-but-wrong` 仅 `10/3673 = 0.27%`
- 这是最危险的失败模式，但它确实很少
- `false_safe` 才是主要风险区

这一部分没有方向性问题，可以直接保留到 discussion。

## 5. Risk Threshold Sweep 修正后更可信，但不够惊艳

来源：

- [threshold_sweep_by_risk.json](/gemini/code/FSPT/outputs/neighbor_deviation_deep_analysis_v1_1/threshold_sweep_by_risk.json)

| Risk Thr | Coverage | Mean Err | High% |
|---|---:|---:|---:|
| 0.01 | 0.240 | 2.98 | 1.1% |
| 0.02 | 0.348 | 3.22 | 2.0% |
| 0.05 | 0.556 | 3.82 | 2.4% |
| 0.10 | 0.758 | 4.85 | 3.7% |
| 0.20 | 0.906 | 5.30 | 4.6% |
| 0.30 | 0.942 | 5.44 | 4.9% |
| 0.50 | 0.983 | 6.09 | 6.1% |

正确解读：

- `raw-risk` 阈值是有意义的
- 但它更像 protocol / abstention 工具，而不是一个特别强的 pseudo-label 过滤器

## 6. Pseudo-Label Filtering 目前不够强

来源：

- [pseudo_label_filtering_stats.json](/gemini/code/FSPT/outputs/neighbor_deviation_deep_analysis_v1_1/pseudo_label_filtering_stats.json)

| Risk Thr | Accept | Reject | Coverage | CorrectRej | FalseRej | Missed |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 2044 | 1629 | 0.556 | 194 | 1435 | 49 |
| 0.10 | 2784 | 889 | 0.758 | 141 | 748 | 102 |
| 0.20 | 3327 | 346 | 0.906 | 90 | 256 | 153 |
| 0.30 | 3459 | 214 | 0.942 | 74 | 140 | 169 |

这意味着：

- `risk < 0.05` 时虽然留下的样本很干净
- 但拒绝集里低误差样本仍然太多
- 目前还不适合作为强主结论

因此 pseudo-label filtering 仍然只能算第二阶段探索，不该现在升级成主贡献。

## 7. Learned Head 结论不变

来源：

- [learned_head_ablation.json](/gemini/code/FSPT/outputs/neighbor_deviation_deep_analysis_v1_1/learned_head_ablation.json)

关键数字：

- `mean_gain = 0.049`
- `median_gain = 0.001`
- `9/17` 视频有正增益
- `1/17` 视频有严重负增益（`loading: -0.339`）

结论不变：

- appendix / ablation only
- 不能作为主方法

## 8. 当前最稳的论文说法

现在可以安全保留的说法是：

1. `neighbor deviation` 是一个可泛化的 trajectory-local risk signal
2. 它在 held-out video calibration 下仍然成立
3. 它在 `occ-length` 分析中 consistently 优于 `visibility`
4. `smooth-but-wrong` 很少见，主要风险是 `false_safe`
5. 它支持 coverage-aware selective tracking，但 overall 提升是温和的，不应夸大

不应再使用的说法：

- “50% coverage 下 high-error rate 5x better”
- “5-10 frame occlusion 上 visibility AUC 只有 0.163”
- “risk < 0.05 时 pseudo-label filtering 已经很强”

## 9. 最终判断

`deep_analysis_v1_1` 的价值在于：

- 提供 failure taxonomy
- 提供 occ-length breakdown
- 提供更规范的 risk-threshold protocol

它 **不** 提供一个新的大幅度性能表。

所以这部分最适合放在：

- 实验分析节
- discussion 节
- appendix 补充表

而不是把它包装成新的主结果突破。
