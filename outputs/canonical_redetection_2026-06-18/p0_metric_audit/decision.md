# P0 Decision

**日期**: 2026-06-19

**Decision**: `BLOCKED_METRIC_RECONCILIATION`

## 当前状态

- ✅ `first_reentry_frame_proxy` 已实现（512/N frame jaccard）
- ✅ `true_AJ_RD` 已实现（报告 `true_AJ_RD` 和 `true_AJ_RD_256` 字段）
- ✅ re-entry metric reconciliation 文档已创建（解释 405px vs 3.91px 等冲突）
- ✅ 坐标 roundtrip 通过
- ✅ Cache schema 一致
- ✅ 256-space AJ_RD 变体已实现
- ✅ query-weighted 聚合已落地
- ✅ AJ_RD@d_min 分解已输出
- ✅ consistency_check 通过（pass=true）
- ✅ determinism 通过（diff=0）
- ✅ manifest 已写
- ✅ GT cache identity 已验证

## 解锁判据（非循环）

以下条件独立可判定的，不依赖 P1：

1. `true_AJ_RD_256` 被采纳为论文 canonical 口径 → 直接放行 P0
2. 或：外部复核确认 `true_AJ_RD_256` 与 `true_AJ_RD` 的双轨并报满足审计要求 → 放行 P0
3. 或：明确决定保留 `BLOCKED_METRIC_RECONCILIATION` 作为最终状态（审计闭合但 gate 不升级）

P0 不依赖 P1 来解锁。P0 由自身的口径完备性判定来解锁。

## 当前阻塞原因

- `true_AJ_RD_256` 是否作为论文 canonical 口径尚未最终确认
- 旧 GO 文档虽已标记 superseded，但仓库内仍有两套口径共存
