# Track-On2 strided+original Status

**日期**: 2026-06-18  
**结论**: ✅ **Complete — full 30-video run finished**

---

## 1. 事件时间线

| 阶段 | 状态 | 详情 |
|---|---|---|
| 初次执行 (Jun 17) | ❌ 运行时报错 | dtype mismatch: `Index put requires the source and destination dtypes match` |
| 修复后 probe (Jun 17) | ✅ 部分产出 | 14+ npz 文件 |
| 完整运行 (Jun 18) | ✅ 完成 | 30/30 videos, AJ=28.38, OA=58.33, delta_avg=33.23 |
| Unified cache export (Jun 18) | ✅ 完成 | 30 records in caches/trackon2_strided_original.pt |
| Schema validation (Jun 18) | ✅ 通过 | All 30 records pass protocol checks |
| Phase 1 oracle (Jun 18) | ✅ 完成 | n=2736 re-entry events, median=405.04px |

---

## 2. 根因与修复

- 报错: `Index put requires the source and destination dtypes match`
- 位置: `baselines/track_on/model/trackon.py:89`
- 修复: `trackon.py:80-81` — dtype alignment of q_features to point_memory.dtype
- 验证: 修复后完整 30-video run 无报错

---

## 3. 最终结果

| 指标 | 值 |
|---|---|
| AJ | 28.38 |
| OA | 58.33 |
| delta_avg | 33.23 |
| delta_4px | 35.90 |
| Re-entry oracle (median) | 405.04px |
| Oracle better 4px | 20.4% |

---

## 4. 对 Phase 0/1/2 的影响

- Phase 0: ✅ 完成
- Phase 1: ✅ 完成（oracle 已计算）
- Phase 2: ✅ 可放行

---
