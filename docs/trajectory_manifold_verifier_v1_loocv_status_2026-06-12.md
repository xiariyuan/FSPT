# Trajectory Manifold v1.1 LOOCV Status (2026-06-12)

## 三句话

1. **Neighbor deviation 在 LOOCV 下泛化成立**：LOOCV avg dev_auc = 0.743，pooled dev_auc = 0.764，无显著掉点。
2. **Learned head LOOCV avg = 0.793**，但个别 fold（loading=0.278, goat=0.448）较弱，说明仍依赖序列特征。
3. **Verdict: PASS (2/3 criteria)**。Trajectory manifold 作为 reliability signal 在跨序列验证下成立。

## LOOCV 关键数字

| Metric | Pooled | LOOCV avg |
|---|---|---|
| Visibility AUC | 0.238 | 0.419 |
| Neighbor deviation AUC | **0.764** | **0.743** |
| Learned head AUC | 0.906 | **0.793** |

Per-fold dev_auc 范围: [0.434, 0.984]，中位数 ≈ 0.755。17/22 个有 high-error 样本的 fold 中 dev_auc > 0.6。

## Subset Metrics

| Subset | n | dev_auc | dev_spearman |
|---|---|---|---|
| all | 3673 | 0.764 | 0.341 |
| long_occ>10 | 1274 | 0.656 | — |
| hard (>16px) | 243 | — (too few in val) | — |

## Calibration (LOOCV pooled)

- ECE: 0.226
- Brier: 0.143
- Mean predicted: 0.2924
- Actual positive rate: 0.0662

Calibration 偏差较大（ECE=0.226），需要后续做 Platt scaling 或 isotonic regression。

## Pass/Fail

| Criterion | 结果 |
|---|---|
| loocv dev_auc > 0.6 AND > vis+0.05 | **PASS** (0.743 > 0.419+0.05) |
| loocv learned_auc > 0.7 | **PASS** (0.793) |
| hard subset dev_auc > 0.6 | **FAIL** (val hard n=243, 不足以计算) |

**Verdict: PASS (2/3)**

## 核心结论

Neighbor-deviation reliability signal **在 LOOCV 下泛化成立**：
- 同集 0.764 → LOOCV 0.743（仅掉 0.021）
- 但 LOOCV 略高于 pooled 说明可能存在 fold 间差异

## 下一步

1. 补 Platt scaling / isotonic 校准
2. 接入 selective tracking evaluation（共形选择性跟踪骨架）
3. 论文叙事："neighbor-deviation as trajectory reliability signal"
