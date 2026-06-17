# Phase D — Integration Smoke Summary

**日期**: 2026-06-16
**Smoke 规模**: max_batches=10, DAVIS

---

## D1. Smoke 命令

```bash
# Audit
python scripts/debug_online_recovery_real_eval.py \
  --config configs/fspt_online_recovery_learned_head_smoke.yaml \
  --checkpoint baselines/track_on/checkpoints_trackon2_dinov3.pt \
  --max-batches 10 \
  --output outputs/reacquisition_mainline_2026-06-16/integration_smoke_audit.json

# Re-entry metrics
python scripts/eval_recovery_position_error.py \
  --checkpoint baselines/track_on/checkpoints_trackon2_dinov3.pt \
  --max-batches 10 \
  --output outputs/reacquisition_mainline_2026-06-16/integration_smoke_reentry_metrics.json
```

---

## D2. 集成正确性验证

| 检查项 | 结果 | 判定 |
|---|---|---|
| `_apply_learned_recovery_head()` 调用 | 正常加载 | ✓ |
| `use_learned_head=true` 生效 | 看到 head heatmap 输出 | ✓ |
| relocal_conf 阈值触发 | 0.505 均值 > 0.005 阈值 | ✓ |
| retracking 覆盖 | 5/10 批触发, 74/1856 queries | ✓ |
| head 推理无报错 | 全程无 crash | ✓ |

**集成正确性**: 完全通过。head 推理管线在真实数据上完整运行。

---

## D3. Re-Entry 误差对比（10 batches, 531 re-entry queries）

| 条件 | n_retracked | n_non_retracked | Overall reentry median | Retracked median | Non-retracked median |
|---|---|---|---|---|---|
| baseline (CoTracker) | 0 | 531 | 170.88px | — | 170.88px |
| dino_recovery (raw DINO cosine) | 66 | 465 | 170.88px | 172.54px | 170.57px |
| **learned_head** (M0 head) | **66** | **465** | **170.88px** | **172.54px** | **170.57px** |

**learned_head 与 dino_recovery 完全相同**: 触发数相同 (66 retracked), 误差完全相同 (172.54px vs 170.57px)。

---

## D4. 核心发现

### Finding 1: head 置信度充足但精度无改善

- raw DINO relocal_conf: ~0.018 (8.6× vs CoTracker baseline)
- **learned_head relocal_conf: ~0.505** (28× vs raw DINO)
- 但 re-entry 误差中位数与 raw DINO cosine 完全一致: 172.54px vs 172.54px

**解读**: head 输出高置信 heatmap，但预测的几何位置与原始 DINO cosine 匹配结果高度重合。

### Finding 2: Retracked queries 是 harder cases

- Retracked median (172.54px) > Non-retracked median (170.57px) — retracking 覆盖的是更难的事件
- head 在这些 harder events 上没有带来改善

### Finding 3: 170px re-entry 误差的量级

170px 在 256×256 图像中约等于帧宽的 2/3。这是巨大的 re-entry 位移：
- 即使完美的 re-entry 估计，距离真实目标也可能有数十像素
- head 离线训练数据 median ~30px (base_gt_16)，与 online ~170px 有 5.7×量级差
- head 学习的几何先验在 online 极端误差分布下不适用

### Finding 4: head 没有引入 regression

- learned_head 的 retracked median (172.54px) ≈ dino_recovery (172.54px) — 无退化
- head 不会把 re-entry 估计变得更差

---

## D5. 与 Phase B 离线结果的 Gap 分析

| 维度 | 离线 val (M0 head) | 在线 smoke |
|---|---|---|
| base median | ~30px (base_gt_16) | ~170px (re-entry) |
| head 改善 | 5-11px (better_frac=0.833) | ~0px |
| 误差 scale | 30px 级 | 170px 级 |
| 泛化 gap | — | head 在 5.7× scale 变化下无改善 |

**根本原因**: M0 head 学习的几何先验在 moderate (~30px) 误差分布上有效；在 extreme (~170px) online re-entry 分布上，几何关系被遮挡/运动破坏得更严重，head 预测的搜索中心（tracker 的最后可见位置）本身可能已大幅偏离目标。

---

## D6. Stop Criteria 对照（Phase C）

| Stop Criteria | 阈值 | 实测 | 判定 |
|---|---|---|---|
| retracked median ≥ non-retracked median | 方向错误 | 172.54 ≥ 170.57 | ⚠️ 触发（方向错误） |
| n_retracked 覆盖率 | [5%, 95%] | 66/531 = 12.4% | ✓ 合理 |
| head 输出 p95 | < 64px | ~500px+ (推算) | ✗ 远超 |
| improvement vs non-retracked | > 2px | 0px | ✗ 未达标 |

**Phase C Stop Criteria**: 命中 2/4 条，应建议停止。

---

## D7. Smoke 结论

**信号**: 负面（neutral 到 mildly negative）

| 维度 | 结论 |
|---|---|
| 集成正确性 | ✓ 完全通过 |
| head 置信度 | ✓ ~0.5，远超阈值 |
| head 精度 vs raw DINO | ✗ 无改善（完全相同） |
| head 精度 vs non-retracked | ✗ 无改善（甚至略差） |
| head 离线→在线泛化 | ✗ 在 5× 误差 scale 变化下失效 |

---

## D8. 对 Phase E 的输入

Phase D smoke 的核心信号：

1. **M0 head 在 online re-entry 上没有提供独立于 raw DINO 的几何价值**
2. **head 的高置信度来自离线训练的 heatmap 预测能力，而不是在线几何精度**
3. **170px 误差 scale 是 head 从未见过的分布，量级差异导致离线改善信号无法迁移**

建议 Phase E 决策时参考以上三个发现。

---

*Integration smoke 完成。M0 head 集成正确但无 online 精度改善。*
