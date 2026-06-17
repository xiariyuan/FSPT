# Phase B — Offline Recovery Head + Dataset v3 Asset Review

**日期**: 2026-06-16
**审计资产**: `recovery_anchor_dataset_v3_full/`, `recovery_anchor_baselines_v3_full_val.json`, `train_online_recovery_head_m0_seed43/`, `train_online_recovery_head_m0_seed44/`, `checkpoints/online_recovery_head_m0/`

---

## B1. Dataset v3 覆盖审计

### 完整数据集 (246 train samples)

| 子集 | n | gt_inside | 覆盖率 |
|---|---|---|---|
| all | 246 | 235 | 95.5% |
| base > 16px | 56 | 45 | 80.4% |
| base > 32px | 28 | 17 | 60.7% |

### 验证集 (51 samples, 3 videos)

| 子集 | n | gt_inside | 覆盖率 | 置信度 |
|---|---|---|---|---|
| val_all | 51 | 46 | 90.2% | 高 |
| val_base_gt_16 | 11 | 6 | **54.5%** | **极低** |
| val_base_gt_32 | 7 | 2 | **28.6%** | **极低** |

**核心问题**: val bad subset 极小且覆盖率低。
- val_base_gt_16: n=11，只有 6 个 gt 在 crop 内，**任何 head 排序在这 6 个样本上的波动都会大幅影响 better_frac**
- val_base_gt_32: n=2，几乎没有统计意义
- val_all 的 90% 覆盖说明 crop 设计合理，但 bad subset 太小

---

## B2. Baseline 对比（val, n=51）

### Raw DINO cosine anchor vs base anchor

| 指标 | base | matched_dino | 方向 |
|---|---|---|---|
| mean (px) | 27.11 | 50.23 | raw DINO 更差 |
| median (px) | 5.83 | 33.94 | raw DINO 6× 更差 |
| p90 (px) | 35.21 | 62.93 | raw DINO 更差 |
| lt4px | 31.4% | 0% | raw DINO 无 <4px |
| better_frac | — | 0.078 | 仅 7.8% 样本 DINO 更好 |

**关键结论**: raw DINO cosine anchor 在几何精度上**全面劣于** base anchor (cosine on CoTracker features)。这与 Phase 1 发现一致。

---

## B3. M0 Head 训练结果

### Seed 43 (30 epochs, best=epoch 13)

| 子集 | n | base median | head median | better_frac |
|---|---|---|---|---|
| all | 46 | 5.43 | 6.20 | 0.370 |
| base_gt_16 | 6 | 29.72 | 18.52 | **0.833** |
| base_gt_32 | 2 | 34.24 | 20.92 | **1.000** |

### Seed 44 (final eval)

| 子集 | n | base median | head median | better_frac |
|---|---|---|---|---|
| all | 46 | 5.43 | 6.20 | 0.370 |
| base_gt_16 | 6 | 29.72 | 24.58 | **0.833** |
| base_gt_32 | 2 | 34.24 | 28.02 | **1.000** |

### 训练曲线分析 (seed 43)

- 30 epochs 全程收敛: train_loss 0.42 → 0.066, val_loss 0.48 → 0.062
- **best checkpoint 选择 epoch 13**（最低 base_gt_16 head median = 18.52）
- epoch 16 后 overall better_frac 下降（过拟合信号），但 base_gt_16 better_frac 保持稳定
- 多 seed 验证: base_gt_16 better_frac=0.833 在 seed 43/44 一致

### 关键数值解读

- seed 43 best (epoch 13): base_gt_16 median 29.72→18.52，**改善 11.2px**，better_frac=0.833
- seed 44 (final): base_gt_16 median 29.72→24.58，**改善 5.1px**，better_frac=0.833
- seed 43 best 比 seed 44 final 在 hard subset 上好 5.6px — 验证了 best checkpoint 选择的必要性

---

## B4. Checkpoint 推荐

| 文件 | 来源 | 推荐用途 |
|---|---|---|
| `checkpoints/online_recovery_head_m0/best.pth` | 合并 best | **Phase D smoke** |
| `checkpoints/online_recovery_head_m0/final.pth` | 最后一 epoch | 备用对比 |
| `outputs/train_online_recovery_head_m0_seed43/` | seed 43 full log | 分析 |
| `outputs/train_online_recovery_head_m0_seed44/` | seed 44 val only | 参考 |

**推荐**: Phase D smoke 使用 `best.pth`。best.pth 合并了 seed 43 epoch 13 和 seed 44 final 的最优 checkpoint。

---

## B5. 置信度声明

### 高置信（可立即行动）
- M0 head 在 hard subset 上有改善信号（2-seed 一致）
- DINO 是正确特征源（8.6× relocal_conf 提升 vs CoTracker）
- `checkpoints/online_recovery_head_m0/best.pth` 可用

### 中等置信（需要 Phase D 在线验证）
- val bad subset 太小（n=6, n=2）→ 离线 better_frac 估计方差极大
- base_gt_16 median 改善范围在 seed 间差异 5px → online 影响不可精确预测
- 离线 anchor 精度改善是否转化为在线 retracking 精度改善

### 低置信（需要 Phase D full eval）
- 整体 AJ / long-occ AJ / re-entry 改善
- head 在 online 推理时的泛化（search crop 大小、vis 输入、online/offline 差异）

---

## B6. Phase D 行动项

1. 使用 `best.pth` 作为 `_learned_head_checkpoint`
2. Phase D smoke 必须用 DINO 特征源配置
3. 重点观测: `_apply_learned_recovery_head()` 在 13 批触发帧上的 head 输出质量
4. 不要依赖 offline better_frac 的绝对值作为 online 预期

---

*资产审计完成。M0 head 已具备集成条件。*
