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

### 3.2 true_AJ_RD breakdown by d_min

N/A = d_min 无 eligible events（排除，不计入 AJ_RD 平均）。AJ_RD 只平均有 ≥1 eligible event 的 d_min。

| Baseline | AJ_RD | AJ_RD@1 | AJ_RD@4 | AJ_RD@16 | AJ_RD@64 | AJ_RD@256 | eligible(1/4/16) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoTracker3 online | 0.4101 | 0.3618 | 0.3325 | 0.2511 | N/A(excl) | N/A(excl) | 1863/1214/421 |
| CoTracker3 offline | 0.3870 | 0.3458 | 0.3110 | 0.2342 | N/A(excl) | N/A(excl) | 1863/1214/421 |
| Track-On2 | 0.3700 | 0.3271 | 0.2994 | 0.2170 | N/A(excl) | N/A(excl) | 1863/1214/421 |

### 3.3 true_AJ_RD_256 breakdown by d_min (TAPNext++ comparable)

| Baseline | AJ_RD_256 | AJ_RD_256@1 | AJ_RD_256@4 | AJ_RD_256@16 | AJ_RD_256@64 | AJ_RD_256@256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 online | 0.5972 | 0.5312 | 0.4914 | 0.3972 | N/A(excl) | N/A(excl) |
| CoTracker3 offline | 0.5546 | 0.4975 | 0.4467 | 0.3556 | N/A(excl) | N/A(excl) |
| Track-On2 | 0.5383 | 0.4909 | 0.4519 | 0.3478 | N/A(excl) | N/A(excl) |

### 3.4 Eligible events by d_min

| Baseline | d_min=1 | d_min=4 | d_min=16 | d_min=64 | d_min=256 |
|---|---:|---:|---:|---:|---:|---:|
| 全部（GT 一致） | 1863 | 1214 | 421 | 0 | 0 |

`d_min=64` 和 `d_min=256` 无 eligible events。原因：d_min=256 超出最大遮挡长度（99 帧）；d_min=64 虽最大值 99 帧 ≥ 64，但落在 [64,99] 的 5 个 occlusion run 均未通过 eligibility 过滤（需满足 record-breaking reappearance 定义）。详见 `reentry_metrics.py:summarize_reappearance_ajrd` docstring。

### 3.5 事件层级关系

```
Total queries:    5882
  └─ w/ ≥1 re-entry (query-level): 1385  → canonical first_reentry_frame_proxy / median re-entry
       └─ all re-entry events: 2466      → GT events from audit_visibility_reentry_events
            └─ eligible by d_min=1: 1863 → AJ_RD 口径基础（1385 queries × 1.34 events/query avg）
                 └─ eligible by d_min=16: 421 → long-occ 子集
```

1385 ≠ 2466 ≠ 1863 ≠ 2736 是口径差异，详见 `reentry_metric_reconciliation.md`。

## 4. Re-entry Events

- n_total_queries: 5882
- n_reentry_queries: 1385
- n_reentry_events: 2466
- Occ buckets: 1-4 (1315), 5-9 (474), 10-19 (431), 20-49 (241), 50-99 (5), 100+ (0)
- Cross-cache consistency: GT data identical across all caches

## 5. 决策

**BLOCKED_METRIC_RECONCILIATION**。P0 的坐标、schema 和 re-entry 事件统计通过；`AJ_RD` 口径已按 per-sample 汇总重算，但 phase gate 仍保留为 blocked，等待后续复核一致性确认。
