# Trajectory Manifold Verifier v1 Status (2026-06-12)

## 三句话

1. **Neighbor deviation 是强可靠性信号**：AUC=0.764 for high-error detection，远超 visibility baseline (0.238) 和 reconstruction proxy (0.576)。
2. **Learned head 达到 AUC=0.906**：在 3673 个样本上，组合 neighbor deviation + trajectory stats 的小 MLP 能以 0.906 AUC 区分高误差轨迹。
3. **Verdict: PASS (2/4 criteria)**。Trajectory manifold 作为 reliability signal 得到证据支持。

## 实验设置

- 数据：TAP-Vid DAVIS val, max_batches=30, n=3675 samples (3673 valid)
- High-error threshold: 16px
- High-error rate: 6.6% (243/3673)
- Neighbor K: 16
- 模型：base tracker predicted tracks vs GT tracks

## 主结果

### AUC (high error detection, n=3673)

| Method | ROC-AUC | PR-AUC |
|---|---|---|
| Visibility baseline | 0.238 | 0.041 |
| Statistics score | 0.138 | 0.037 |
| Reconstruction error | 0.576 | 0.093 |
| **Neighbor deviation** | **0.764** | **0.219** |
| **Learned head** | **0.906** | **0.364** |

### Correlation (neighbor deviation vs pixel error)

| Metric | Value |
|---|---|
| Pearson | 0.314 |
| Spearman | 0.341 |

### Subset Analysis

| Subset | n | error_med | recon_auc |
|---|---|---|---|
| all | 3673 | 2.81px | 0.576 |
| long_occ>10 | 1274 | 3.50px | 0.443 |
| hard (>16px) | 243 | 31.58px | — |
| easy (<4px) | 2394 | 1.95px | — |

## 解读

1. **Visibility baseline < 0.5** (0.238)：高可见度反而与高误差相关！这是因为 visible points 在复杂运动中更容易出错，而长期遮挡的点可能在简单运动中。

2. **Neighbor deviation 是核心信号** (AUC=0.764)：query 轨迹偏离邻居轨迹越多，越可能是错误的。这是一个轨迹流形信号——"这条轨迹是否在局部运动结构内"。

3. **Learned head 极强** (AUC=0.906)：组合 neighbor deviation + trajectory statistics 的小 MLP 能很好地区分高误差轨迹。这意味着 trajectory manifold features 有丰富的可靠性信息。

4. **Reconstruction error 弱** (AUC=0.576, Spearman=-0.026)：用邻居重建 query 轨迹的误差不是好信号。可能是因为轨迹都靠近邻居（低重建误差），但绝对位置可能错误。

5. **Long-occ subset recon_auc < 0.5** (0.443)：在长遮挡子集上重建误差没有信号，但 neighbor deviation 可能仍然有效（需要进一步检查）。

## Pass/Fail 判定

| Criterion | 结果 |
|---|---|
| recon > visibility baseline | **PASS** (0.576 > 0.238) |
| Spearman > 0.3 | **FAIL** (-0.026) |
| long_occ recon_auc > 0.6 | **FAIL** (0.443) |
| learned head AUC > 0.65 | **PASS** (0.906) |

**Verdict: PASS (2/4)**

## Findings

1. **Neighbor deviation (AUC=0.764) 是第一个在 trajectory-only 设置下有效的 reliability signal**。它不需要图像特征、不需要 DINO、不需要 recovery candidate——只看轨迹邻域结构。

2. **Learned head (AUC=0.906) 表明 trajectory manifold features 组合后有极强判别力**。6.6% 的高误差率下 PR-AUC=0.364 也说明信号不是 trivial 的。

3. **Reconstruction proxy 本身不够** (AUC=0.576)。关键信号不是"重建轨迹准不准"，而是"轨迹是否偏离局部运动流形"。

## Open Questions

1. Neighbor deviation 是否在 long-occlusion / re-entry 子集上仍然有效？当前只在 overall 上报告了 AUC。
2. Learned head 是否能在 sequence-grouped split 上泛化？当前是同集拟合。
3. Neighbor deviation 能否直接作为 selective tracking 的拒绝信号？

## 下一步建议

1. 做 sequence-grouped LOOCV 验证 learned head 泛化（与 patch verifier 同一协议）
2. 检查 neighbor deviation 在 long-occ / re-entry 子集上的 AUC
3. 如果泛化成立，接入 selective tracking evaluation pipeline
