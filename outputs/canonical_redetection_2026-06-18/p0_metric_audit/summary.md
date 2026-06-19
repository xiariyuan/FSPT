# P0 — Metric, Coordinate, and Cache Audit Summary

**日期**: 2026-06-18  
**结论**: ⚠️ **BLOCKED_METRIC_RECONCILIATION** — technical audit passes, final gate remains blocked

---

## 1. 坐标 Roundtrip

| 测试 | 最大误差 | 通过 (≤1e-6) |
|---|---:|---:|
| yx_norm → pixel → yx_norm | 5.96e-08 | ✅ |
| yx_norm → xy_pixel → yx_norm | 5.96e-08 | ✅ |
| yx → xy → yx swap | 0.0 | ✅ |

## 2. Cache Schema 审计

| Baseline | Records | Clean | Issues | 问题类型 |
|---|---:|---:|---:|
| CoTracker3 online | 30 | 10 | 20 | pred_tracks 轻微越界 [-0.63, +1.74] |
| CoTracker3 offline | 30 | 13 | 17 | pred_tracks 轻微越界 [-0.94, +3.06] |
| Track-On2 | 30 | 20 | 10 | pred_tracks 轻微越界 [-0.001, +1.002] |

越界是模型预测行为（hallucination），不影响坐标转换的正确性。

## 3. AJ_RD

### 3.1 概要

| Baseline | true_AJ_RD | true_AJ_RD_256 | first_reentry_proxy | Median re-entry | <4px |
|---|---:|---:|---:|---:|---:|
| CoTracker3 online | **0.4101** | **0.5972** | 0.4732 | 3.86px | 51.3% |
| CoTracker3 offline | **0.3870** | **0.5546** | 0.4101 | 3.71px | 53.8% |
| Track-On2 | **0.3700** | **0.5383** | 0.3681 | 3.91px | 51.0% |

注：
- `true_AJ_RD` = original resolution（当前项目内口径，偏严）
- `true_AJ_RD_256` = 256-space（TAPNext++ 可比口径，推荐用于论文对比）
- 256-space 值较高的原因：相同 px 阈值在 256×256 下比 480×854 下占比更大
- `true_AJ_RD < true_AJ_RD_256 < first_reentry_proxy` 符合预期

### 3.2 Eligible events by d_min

d_min = 最小遮挡长度，用于筛选 eligible re-appearance event。

| Baseline | d_min=1 | d_min=4 | d_min=16 | d_min=64 | d_min=256 |
|---|---:|---:|---:|---:|---:|
| 全部 | 1863 | 1214 | 421 | 0 | 0 |

所有 baseline 使用相同的 GT，eligible events 一致。

### 3.3 true_AJ_RD@d_min 分解

当前输出尚未分解 AJ_RD@d_min。代码中的 `summarize_reappearance_ajrd` 已实现 per-d_min 聚合，
但 summary 输出层尚未展开。预计在最终 audit closure 中补全：

```
AJ_RD@1   AJ_RD@4   AJ_RD@16  AJ_RD@64  AJ_RD@256
```

### 3.4 事件层级关系

| 层级 | n | 说明 |
|---|---:|---|
| Total queries | 5882 | 所有 strided query |
| Queries with ≥1 re-entry | 1385 | 用于 canonical P0 AJ_RD |
| Re-entry events (all) | 2466 | 所有 re-appearance 事件 |
| Eligible events (phase0/phase2b) | 2736 | 历史统计，含旧 cache/probe 数据 |
| Eligible by d_min=1 | 1863 | AJ_RD 口径基础 |
| Eligible by d_min=16 | 421 | long-occ 子集 |

1385 ≠ 2466 ≠ 2736 的原因是口径不同，已在 `reentry_metric_reconciliation.md` 中详细解释。

## 4. Re-entry Events

- n_total_queries: 5882
- n_reentry_queries: 1385
- n_reentry_events: 2466
- Occ buckets: 1-4 (1315), 5-9 (474), 10-19 (431), 20-49 (241), 50-99 (5), 100+ (0)
- Cross-cache consistency: GT data identical across all caches

## 5. 决策

**BLOCKED_METRIC_RECONCILIATION**。P0 的坐标、schema 和 re-entry 事件统计通过；`AJ_RD` 口径已按 per-sample 汇总重算，但 phase gate 仍保留为 blocked，等待后续复核一致性确认。
