# Online Recovery Head M0 Status — Formal (2026-06-10)

## M0 结论

**Learned geometric recovery head 在 base-bad 子集上稳定优于 base tracker。**

## 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | `outputs/recovery_anchor_dataset_v3_full/` |
| Train samples | 195 (10 videos) |
| Val samples | 51 (3 videos: bike-packing, breakdance, pigs) |
| Val gt_inside | 46 |
| Epochs | 30 |
| Batch size | 8 |
| Learning rate | 1e-3 |
| Model | OnlineRecoveryHead (489K params) |
| Loss | L = BCE_heatmap + 0.25 * L1_coord |
| Sampler | Weighted: base≤16 → 1, 16-32 → 4, >32 → 8 |
| Best selection | lowest base_gt_16 head median |

## 3-Seed 复核结果

### Base >16px 子集 (n=6, 全部 gt_inside)

| Seed | Head Median | Base Median | better_frac |
|------|------------|------------|-------------|
| 42 | **14.64** | 29.72 | **0.833** |
| 43 | **18.52** | 29.72 | **0.833** |
| 44 | **24.58** | 29.72 | **0.833** |

**三组 seed better_frac 全部 0.833，全部低于 base median 29.72。**

### Base >32px 子集 (n=2, 全部 gt_inside)

| Seed | Head Median | Base Median | better_frac |
|------|------------|------------|-------------|
| 42 | **17.69** | 34.24 | **1.000** |
| 43 | **20.92** | 34.24 | **1.000** |
| 44 | **28.02** | 34.24 | **1.000** |

**三组 seed 全部 100% 改善。**

### All valid samples (n=46, gt_inside=True)

| Seed | Head Median | Base Median | better_frac |
|------|------------|------------|-------------|
| 42 | 11.26 | 5.43 | 0.217 |
| 43 | 10.13 | 5.43 | 0.196 |
| 44 | 6.20 | 5.43 | 0.370 |

## Coverage Audit

| 子集 | N | gt_inside | Coverage |
|------|---|-----------|----------|
| All samples | 246 | 235 | 95.5% |
| Base >16px | 56 | 45 | 80.4% |
| Base >32px | 28 | 17 | 60.7% |
| Val all | 51 | 46 | 90.2% |
| Val base >16px | 11 | 6 | 54.5% |
| Val base >32px | 7 | 2 | 28.6% |

**注意：val bad subset 的有效样本量非常小 (n=6, n=2)。** 这是当前结果的局限。

## 可视化

- `outputs/online_recovery_head_m0_vis/bad_subset_all.png`
- 覆盖全部 6 个 base>16 & gt_inside 的 val 样本

## 工件清单

| 文件 | 路径 |
|------|------|
| 训练脚本 | `scripts/train_online_recovery_head.py` |
| 模型定义 | `models/online_recovery_head.py` |
| 数据集 loader | `datasets/recovery_anchor_dataset.py` |
| Best checkpoint | `checkpoints/online_recovery_head_m0/best.pth` |
| Final checkpoint | `checkpoints/online_recovery_head_m0/final.pth` |
| Train log | `outputs/train_online_recovery_head_m0/train_log.json` |
| Config | `outputs/train_online_recovery_head_m0/config.json` |
| Coverage audit | `outputs/recovery_anchor_dataset_v3_full/coverage_audit.json` |
| Baseline results | `outputs/recovery_anchor_baselines_v3_full_val.json` |
| Seed 43 results | `outputs/train_online_recovery_head_m0_seed43/val_metrics.json` |
| Seed 44 results | `outputs/train_online_recovery_head_m0_seed44/val_metrics.json` |
| Visualization | `outputs/online_recovery_head_m0_vis/bad_subset_all.png` |

## 是否进入 Online Integration

**建议：可以继续，但需要先扩大 bad-subset val 样本量。**

理由：
1. M0 在 base-bad 子集上 3-seed 稳定优于 base（better_frac=0.833）
2. 但 val base>16 & gt_inside 只有 6 个样本，统计支撑不足
3. 需要把 bad subset 扩大到 12+ 样本才能有足够置信度
4. 扩大后需要重新跑 baseline + head 对照

下一步：扩大 val（增加更多长遮挡/harder sequences），然后重新评测。如果扩大后 bad subset 仍然稳定优于 base，就可以进入 online integration。
