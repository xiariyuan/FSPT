# P0 Decision

**日期**: 2026-06-19

**Decision**: `BLOCKED_METRIC_RECONCILIATION`

## 当前状态

- ⚠️ `first_reentry_frame_proxy` 已实现（512/N frame jaccard）
- ⚠️ `true_AJ_RD` 已实现（报告 `true_AJ_RD` 字段）
- ⚠️ re-entry metric reconciliation 文档已创建（解释 405px vs 3.91px 等冲突）
- ✅ 坐标 roundtrip 通过
- ✅ Cache schema 一致

## Unsolved

- P0 通过了技术审计（coord、schema、cache），但 true AJ_RD 的口径需要在 P1 复核通过后才能解锁 P0 的完全通过状态。
- 当前状态：`BLOCKED_METRIC_RECONCILIATION`。P1 暂不能引用 P0 的 `ADVANCE_TO_P1` 权限。

## 阻塞原因

1. 需要 P1 确认 true AJ_RD 口径与官方指标一致后，才可将 P0 改为 PASS
2. 当前 reconciliation 已记录冲突原因，但尚未完全消除口径歧义
