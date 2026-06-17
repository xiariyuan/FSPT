# Predicted-Track Neighbor Deviation Diagnosis (Corrected) — Status (2026-06-13)

## 三句话

1. **Stage-2 FAIL — pseudo-label 训练线正式关闭。** Neighbor-deviation 在 predicted tracks 上输给 pred_visibility baseline。
2. 正确口径：pred_visibility overall AUC=0.096（reliability 方向，高 vis → 不是 high error），coverage accept mean_error=3.15px；neighbor_dev_confidence AUC=0.450，accept mean_error=5.80px。
3. 之前文档里的 PASS/FAIL 矛盾和方向性混用已全部修正。

## 修正后结果（方向一致）

### Overall (n=3675)

| Score Source | AUC (reliability direction) | Spearman |
|---|---|---|
| pred_visibility | 0.096 | -0.484 |
| neighbor_dev_confidence | 0.450 | -0.187 |
| verifier_scores | 0.427 | -0.067 |

注：pred_visibility AUC 低（0.096）是因为高 visibility 反而与 high error 正相关（Spearman=-0.484），这是 TAP-Vid 的已知模式——难的 tracking 样本本身 visibility 也不低。但在 coverage-aligned accept 场景下，pred_visibility 按 top-visibility 筛选仍是最优。

### Long-Occ (n=1274)

| Score Source | AUC | Spearman |
|---|---|---|
| pred_visibility | 0.073 | -0.699 |
| neighbor_dev_confidence | 0.554 | -0.031 |
| verifier_scores | 0.446 | 0.055 |

### Coverage-Aligned Accept (50%)

| Source | mean_err | high% | <4px |
|---|---|---|---|
| **pred_visibility** | **3.15** | **0.9%** | **78.3%** |
| neighbor_dev | 5.80 | 6.0% | 71.4% |
| verifier_scores | 5.65 | 5.1% | 68.5% |

## 结论

pred_visibility 在 coverage-aligned accept 场景中全面优于 neighbor_deviation。原因可能是：在 teacher predicted tracks 上，visibility 本身已是一个足够好的伪标签质量信号，不需要额外的 neighbor deviation 修正。

**正式关闭：neighbor-deviation pseudo-label training line。** 不进入训练矩阵。
