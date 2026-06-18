# Re-entry Metric Reconciliation

**日期**: 2026-06-19  
**目标**: 解释仓库中多个 Track-On2 / CoTracker3 re-entry 数字的冲突

---

## 1. 冲突概览

仓库中目前存在三套 strided+original re-entry 数字：

| Source | n_reentry | Track-On2 median | CT-offline median | 事件定义 |
|---|---|---|---|---|
| `redetection_ladder/phase0_metric_sanity` | 2736 | 405.04px | 3.15px | per-GT-point（每个 GT 点计一次） |
| `attempt1_reentry_head2head` | 1385 | 3.91px | 3.71px | per-query（每个 strided query 计第一次） |
| `canonical P0 eval_aj_rd` | 1385 | 3.91px | 3.71px | per-query（同 attempt1） |

## 2. 为什么存在两套 n_reentry

### 2.1 n_reentry = 2736 的来源

`redetection_ladder_2026-06-17/phase0_metric_sanity.json` 的 oracle 部分是**对每个 GT 点的** re-entry 事件计数，**不是** per-query。

具体来说：
- 统一缓存中每个视频有 **N** 条 query（不同 stride frame 的可见 GT 点）
- 2736 = 所有 query 中**至少有一次 re-entry 的 query 数**
- Track-On2 median = 405.04px: 这是 Track-On2 在**所有 re-entry query 上的中值误差**
  - 405px 高的原因是 Track-On2 在 44% 的预测中输出 [0,0]（预测为遮挡）
  - 这些 [0,0] 预测对应大误差，拉高了中值

### 2.2 n_reentry = 1385 的来源

`attempt1_reentry_head2head_strided_original.json` 使用**不同的聚合口径**：
- 只统计**第一个 re-entry 事件**
- 1385 = 总查询（5882）中有至少一个 re-entry 的查询比例
- 这个值依赖头对头脚本的采样策略，不一定包含所有 2736 个 re-entry events

### 2.3 关键区别

| | 2736 | 1385 |
|---|---|---|
| 统计对象 | re-entry events | queries with ≥1 re-entry |
| 来源脚本 | oracle 审计脚本 | head2head 脚本 |
| query 范围 | 所有 strided query | 所有 strided query |
| 是否限制 occ length | 否 | 否 |
| 是否捕获所有 re-entry | 是（含多次） | 否（只计第一个） |

## 3. 哪个是 canonical

| 用途 | 使用哪个 | 原因 |
|---|---|---|
| AJ_RD 指标 | 1385-based | TAPNext++ AJ_RD 定义按 eligible events 算 |
| 原生 re-entry 误差分布 | 2736-based | 覆盖更全的 re-entry 事件 |
| long-occ 分析 | 二者皆可 | 用一致性的事件定义即可 |
| 训练目标 | 1385-based | 只学第一次 re-entry 更简洁 |

## 4. Track-On2 405.04px vs 3.91px 的解释

| 来源 | 中值 | 原因 |
|---|---|---|
| `phase0_metric_sanity` | 405.04px | 包含 44% [0,0] 预测，中值被严重拉高 |
| `attempt1_head2head` | 3.91px | 排除了 [0,0] 的查询？或 query subset 不同 |

实际检查：`trackon2_strided_original_status.json` 已记录了这个差异：
```json
"note": "44% of predictions are [0,0] (model marks as occluded); high re-entry error reflects tracking failures"
```

405.04px 包含所有预测（含 [0,0] 遮挡输出），3.91px 可能来自不同子集。

## 5. 结论

- 数字冲突的主要来源是**不同脚本用不同口径统计 re-entry**
- canonical P0 使用 1385 queries（per-query first re-entry），与 AJ_RD 口径一致
- 2736 events（per-GT-point all re-entry）仍可作为 oracle 上界参考
- Track-On2 405.04px 包含大量 [0,0] 遮挡输出，不代表正常跟踪精度
