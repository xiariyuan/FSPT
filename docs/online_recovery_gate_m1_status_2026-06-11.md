# M1 Gate Status: Blocked (2026-06-11)

## 结论

**M1 gate 暂停。原因：upstream candidate degeneracy，不是 gate 设计失败。**

## 证据

### CoTracker anchor = base（完全相同）
- `outputs/online_recovery_gate_dataset_v1/build_stats.json`
- accept_pos_frac = 0.0, accept_neg_frac = 0.0, accept_ambiguous_frac = 1.0
- 146/146 samples: improve_px = 0.0

### DINO anchor ≈ base（差异 < 0.01px）
- `outputs/online_recovery_gate_dataset_v1_dino/build_stats.json`
- accept_pos_frac = 0.0, accept_neg_frac = 0.0, accept_ambiguous_frac = 1.0
- improve_px mean = 0.0012px, max = +0.03px, min = -0.02px
- 即使 margin 降到 0.01px，也只出现 20 正 / 9 负样本（数值噪声级别）

### 根本原因

当前 online relocalization 的 raw cosine search 产出的 top-1 anchor 在几何位置上与 base tracker 几乎完全相同。差异只有 1e-5 归一化坐标量级，对应像素误差变化约 0.01px。

这不是"margin 设得太严"的问题，而是**上游根本没有产出有信息量的 alternative candidate**。

## 状态标记

**Stop: candidate quality insufficient. Need to fix upstream first.**

## 下一阶段

M0.5: Upstream candidate generation audit
- 目标：让 online path 产出真正不同于 base 的候选
- 优先级：top-k candidate export → search map 诊断 → feature backbone 评估
- 不再继续 gate 训练
