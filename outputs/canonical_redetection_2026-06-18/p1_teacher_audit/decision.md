# P1 Decision

**日期**: 2026-06-19

**Decision**: `INVALID_UNTIL_P0_FIXED`

## 状态说明

P1 的 oracle teacher gain 当前基于 `first_reentry_frame_proxy`，不是 true AJ_RD。在 P0 的 true AJ_RD 口径完全锁定前，P1 的 Go/Stop 判定视为无效。

## Oracle Teacher Selection（基于 first_reentry_frame_proxy）

| Metric | Fixed Best (CT-offline) | Oracle Selection | Gain |
|---|---:|---:|---:|
| first_reentry_frame_proxy | 0.4101 | 不变待复核 | TBD |
| Median re-entry | 3.71px | 1.70px | -2.01px |

## 当前阻止 P1 通过的问题

1. P0 = `BLOCKED_METRIC_RECONCILIATION`，P1 不能单独 GO
2. Oracle gain 使用的 AJ_RD proxy 与 true AJ_RD 存在差异
3. 需要等 P0 的 true AJ_RD 完全确认后，用真实 AJ_RD 重算 oracle gain

## 下一步

P0 通过 → P1 用 true AJ_RD 重算 oracle teacher gain → 再判定 Go/Stop
