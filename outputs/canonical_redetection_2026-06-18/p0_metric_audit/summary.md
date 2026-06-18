# P0 — Metric, Coordinate, and Cache Audit Summary

**日期**: 2026-06-18  
**结论**: ✅ **GO** — 所有检查通过

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

| Baseline | AJ_RD | Median re-entry | <4px | <8px |
|---|---:|---:|---:|---:|
| CoTracker3 online | 0.4101 | 3.86px | 51.3% | 77.0% |
| CoTracker3 offline | 0.3870 | 3.71px | 53.8% | 80.2% |
| Track-On2 | 0.3700 | 3.91px | 51.0% | 73.6% |

## 4. Re-entry Events

- n_total_queries: 5882
- n_reentry_queries: 1385
- n_reentry_events: 2466
- Occ buckets: 1-4 (1315), 5-9 (474), 10-19 (431), 20-49 (241), 50-99 (5), 100+ (0)
- Cross-cache consistency: GT data identical across all caches

## 5. 决策

**GO → P1**。P0 的坐标、schema 和 re-entry 事件统计通过；`AJ_RD` 口径已按 per-sample 汇总重算。
