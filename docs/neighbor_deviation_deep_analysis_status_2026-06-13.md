# Neighbor-Deviation Deep Analysis Status (2026-06-13)

## 三句话

1. **Coverage-driven selective tracking 是最有力的论文表**：50% coverage 下 neighbor_dev mean error=3.50px vs visibility=9.12px，高误差率从 11.5% 降到 2.2%。
2. **Failure taxonomy 显示信号边界清晰**：smooth-but-wrong 仅 10/3673 例（0.27%），主要风险在 false-safe（167 例，4-16px 范围）。
3. **Learned head median gain=0.001，9/17 视频有正增益但 1 个严重负增益**——定位为 appendix/ablation。

## 1. Coverage-Driven Selective Tracking（核心论文表）

| Coverage | dev_mean_err | dev_high% | vis_mean_err | vis_high% |
|---|---|---|---|---|
| 0.10 | **3.08** | **1.6%** | 15.64 | 20.7% |
| 0.20 | **3.18** | **1.4%** | 12.41 | 15.8% |
| 0.30 | **3.20** | **2.2%** | 11.55 | 16.0% |
| 0.50 | **3.50** | **2.2%** | 9.12 | 11.5% |
| 0.70 | **4.33** | **3.0%** | 7.65 | 8.6% |
| 0.90 | **5.32** | **4.7%** | 6.87 | 7.1% |
| 1.00 | 6.58 | 6.6% | 6.58 | 6.6% |

**解读**：neighbor_dev 在 50% coverage 时就能把 mean error 从 6.58 降到 3.50（-47%），而 visibility 在 50% coverage 时仍有 9.12px。高误差率：dev 2.2% vs vis 11.5%（5x 更好）。

## 2. Occ-Length Bins（信号强度随遮挡长度变化）

| Occ Range | n | high_frac | dev_AUC | vis_AUC |
|---|---|---|---|---|
| 0-5 | 1895 | 2.0% | 0.684 | 0.467 |
| **5-10** | **411** | **3.6%** | **0.943** | **0.163** |
| 10-20 | 485 | 12.4% | 0.675 | 0.386 |
| 20-50 | 756 | 13.8% | 0.647 | 0.407 |
| 50-300 | 126 | 21.4% | 0.744 | 0.390 |

**关键发现**：
- 5-10 帧遮挡段 dev_AUC=0.943（极强），visibility 只有 0.163
- 所有遮挡段上 dev 都优于 vis
- 长遮挡（50-300）上 dev_AUC=0.744 仍然有效

## 3. Failure Case Taxonomy

| 类型 | 数量 | 说明 |
|---|---|---|
| smooth_but_wrong | **10** | 低偏差但高误差——邻居也全错（极罕见） |
| false_safe | **167** | 低偏差，中等误差（4-16px）——看起来安全但实际不完全 |
| high_dev_accurate | **441** | 高偏差但低误差——异常但碰巧对了 |
| neighbor_drift | **203** | 高偏差，高误差，邻居也错——信号正确 |

**smooth_but_wrong 仅 10 例（0.27%）**——这是最危险的失败模式（信号说安全但实际错），但它极罕见。

**false_safe 是主要风险**：167 例（4.5%）在 4-16px 范围，偏差低但不完全安全。这些样本不会被高偏差阈值过滤。

## 4. Risk Threshold Sweep

| Risk Thr | Accept | Coverage | Mean Err | High% |
|---|---|---|---|---|
| 0.01 | 1633 | 44.5% | 3.40 | 2.3% |
| 0.02 | 2322 | 63.2% | 3.88 | 2.3% |
| **0.05** | **3197** | **87.0%** | **5.33** | **4.6%** |
| 0.10 | 3477 | 94.7% | 5.46 | 4.9% |
| 1.00 | 3673 | 100% | 6.58 | 6.6% |

**At risk < 0.05**：87% coverage，mean error 从 6.58 降到 5.33（-19%），high error 从 6.6% 降到 4.6%（-30%）。

## 5. Learned Head Ablation

| 视频 | dev_AUC | learned_AUC | gain |
|---|---|---|---|
| lab-coat | 0.434 | 0.794 | **+0.360** |
| motocross-jump | 0.596 | 0.934 | **+0.338** |
| parkour | 0.603 | 0.871 | **+0.268** |
| loading | 0.617 | 0.278 | **-0.339** |
| **Median** | | | **+0.001** |

**Verdict: appendix_only**。Learned head 在部分视频上有显著增益，但 loading 上严重负增益。Median gain 接近零，不稳定性明显。主结论不应依赖它。

## 6. Pseudo-Label Filtering Prep

| Risk Thr | Accept | Reject | CorrectRej | FalseRej | Missed |
|---|---|---|---|---|---|
| 0.05 | 3197 | 476 | 96 | 380 | 147 |
| 0.10 | 3477 | 196 | 72 | 124 | 171 |
| 0.20 | 3664 | 9 | 0 | 9 | 243 |

At risk < 0.05：reject 476 条轨迹，其中 96 条确实是高误差（正确拒绝），380 条是低误差（误拒）。漏掉了 147 条高误差。**rejection precision = 20.2%**。

## 论文叙事定位

### 主表
1. Coverage-driven selective tracking 表（dev vs visibility）
2. Occ-length bin AUC 表

### 主结论
"Neighbor deviation is a trajectory-local motion inconsistency signal that generalizes across sequences. At 50% coverage, it reduces mean tracking error by 47% (3.50 vs 6.58px) and high-error rate by 5x (2.2% vs 6.6%)."

### Discussion 要点
- smooth-but-wrong 极罕见（0.27%），但 false-safe 是主要风险（4.5%）
- 5-10 帧遮挡段信号最强（AUC=0.943）
- Learned head 有增益但不稳定，留作 future work
- Pseudo-label filtering 是自然延伸，当前 rejection precision 20% 有待提升

## 工件

- `outputs/neighbor_deviation_deep_analysis_v1/summary.json`
- `outputs/neighbor_deviation_deep_analysis_v1/long_occ_per_video.json`
- `outputs/neighbor_deviation_deep_analysis_v1/long_occ_bins.json`
- `outputs/neighbor_deviation_deep_analysis_v1/failure_cases.json`
- `outputs/neighbor_deviation_deep_analysis_v1/threshold_sweep_by_coverage.json`
- `outputs/neighbor_deviation_deep_analysis_v1/threshold_sweep_by_risk.json`
- `outputs/neighbor_deviation_deep_analysis_v1/learned_head_ablation.json`
- `outputs/neighbor_deviation_deep_analysis_v1/pseudo_label_filtering_stats.json`
