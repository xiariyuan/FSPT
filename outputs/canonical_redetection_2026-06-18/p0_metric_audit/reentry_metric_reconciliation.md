# Re-entry Metric Reconciliation

**日期**: 2026-06-19 (v2, expanded)  
**目标**: 解释仓库中多个 Track-On2 / CoTracker3 re-entry 数字的冲突

---

## 1. 冲突概览

仓库中目前存在四套 strided+original re-entry 数字：

| Source | n | Track-On2 median | CT-offline median | pred_vis handling | event unit | canonical? |
|---|---|---|---|---|---|---|
| `redetection_ladder/phase0_metric_sanity` | 2736 | 405.04px | 3.15px | 忽略（含 [0,0] occl pred） | per-query 所有 re-entry | ❌ deprecated |
| `attempt1_head2head` | 1385 | 3.91px | 3.71px | 不过滤（含 occl pred） | per-query 首次 re-entry | ⚠️ 旧版本 |
| `phase2b candidate audit` | 2736 | 6.5px | 3.2px | 不过滤（含 occl pred） | per-query 所有 re-entry | ⚠️ 旧版本 |
| `canonical P0 eval_aj_rd` | 1385 | 3.91px | 3.71px | 不过滤（visibility 包含于 Jaccard） | per-query 首次 re-entry | ✅ 当前 canonical |

## 2. 为什么存在三套 n_reentry

### 2.1 第一维度：per-query first re-entry (1385) vs all re-entry (2736)

| | 1385 | 2736 |
|---|---|---|
| 统计对象 | queries with ≥1 re-entry | 所有 re-entry event（一个 query 可能多次） |
| 来源方法 | first re-entry 停算 | 所有 occlusion run 都算 |
| 用于何处 | AJ_RD（TAPNext++ 按 eligible event） | 全量 oracle 上界 |
| canonical P0 使用 | ✅ | ❌（仅作参考） |

### 2.2 第二维度：Track-On2 的 [0,0] 问题

Track-On2 在 strided+original 下约 44% 的预测为 `[0,0]`（模型标为 occluded）：
- **405.04px**（`phase0_metric_sanity` 旧版）：直接算位置误差，[0,0]→GT 产生几百 px
- **6.5px**（`phase2b` 候选审计）：同 2736 events，但使用不同 cache 版本
- **3.91px**（`canonical P0`）：per-query first re-entry，query-level 中值直接算位置误差

虽然都是 position-only error（不排除 occluded pred），但 405px 的来源是旧 cache/旧脚本。

### 2.3 第三维度：cache 版本

| Source | Track-On2 cache | 时间 |
|---|---|---|
| `phase0_metric_sanity` (405px) | 旧 `/tmp/probe` + `outputs/trackon2_dinov3_davis_cache` | 2026-06-17 |
| `canonical P0` (3.91px) | `caches/trackon2_strided_original.pt` | 2026-06-18 |

旧 cache 数据不完整（probe 只有 14 npz，后来重跑 30），造成了 405px 与 3.91px 的差异。

## 3. 六套数字的最终对齐

| 来源 | n | 模型 | median | 数据源 | pred_vis | 建议 |
|---|---|---|---:|---|---|---|
| `phase0_metric_sanity` | 2736 | Track-On2 | 405.04 | 旧 cache (probe) | 含 [0,0] | ❌ 废弃 |
| `phase0_metric_sanity` | 2736 | CT-offline | 3.15 | 旧 cache | 含 [0,0] | ❌ 废弃 |
| `attempt1_head2head` | 1385 | Track-On2 | 3.91 | strided unified pt | 含 [0,0] | ⚠️ 历史参考 |
| `attempt1_head2head` | 1385 | CT-offline | 3.71 | strided unified pt | 含 [0,0] | ⚠️ 历史参考 |
| `phase2b audit` | 2736 | Track-On2 | 6.5 | strided unified pt | 含 [0,0] | ⚠️ 历史参考 |
| `phase2b audit` | 2736 | CT-offline | 3.2 | strided unified pt | 含 [0,0] | ⚠️ 历史参考 |
| **canonical P0** | **1385** | **Track-On2** | **3.91** | **strided unified pt** | **含 [0,0]** | **✅ canonical** |
| **canonical P0** | **1385** | **CT-offline** | **3.71** | **strided unified pt** | **含 [0,0]** | **✅ canonical** |
| **canonical P0 true_AJ_RD** | **eligible** | **CT-offline** | **—** | **strided unified pt** | **visibility-aware** | **✅ canonical** |

## 4. 405px 的解释（最终）

405.04px 来源：旧 `redetection_ladder` 脚本使用了当时新跑的 Track-On2 cache（包含 [0,0]），计算所有 2736 个 re-entry events 的 position-only error。这不是 Track-On2 的"正常"re-entry 精度（3.91px 更接近实际），而是暴露了 Track-On2 在 strided+original 下有 44% 的 tracking failure rate。

两个数字反映的是**不同的问题维度**：
- **405px**：Track-On2 在 44% 的 case 上完全丢失目标
- **3.91px**：在剩余 case 上精度接近 CT-offline

两者都有效，但口径不同。

## 5. 事件层级关系（1385 vs 2466 vs 2736）

三个 n 值反映不同统计粒度，不是同一口径的冲突：

```
Total queries:    5882  (30 videos × strided queries)
  └─ w/ ≥1 re-entry: 1385  → canonical P0 first-reentry-per-query
       └─ all events: 2466  → 同一数据集所有 re-appearance 事件
            └─ historical: 2736  → 含旧 cache/probe 数据，已废弃
```

- **1385** = per-query first re-entry（canonical，与 AJ_RD 对齐）
- **2466** = all re-entry events（同一 GT，同一 cache，full coverage）
- **2736** = 历史版本（旧 cache / probe 数据，`deprecated`）

## 6. canonical 定义

| 用途 | 使用 | 原因 |
|---|---|---|
| AJ_RD 主指标 | `true_AJ_RD`（post-reappearance segment AJ） | 官方 TAPNext++ 口径 |
| 位置误差参考 | `first_reentry_frame_proxy`（单帧 Jaccard） | 与历史记录对齐 |
| oracle 上界 | 2736 events（含所有 re-entry） | 覆盖更全 |
| 训练/审计 | 1385 queries（per-query first re-entry） | 与 AJ_RD 事件定义一致 |
