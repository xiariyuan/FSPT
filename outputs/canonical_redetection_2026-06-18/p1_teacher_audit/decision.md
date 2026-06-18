# P1 Decision

**日期**: 2026-06-19

**Decision**: `INVALID_UNTIL_P0_FIXED`

## 状态说明

`oracle_teacher_selection.json` 已经计算出 true `AJ_RD` headroom，但 phase gate 仍保持 `INVALID_UNTIL_P0_FIXED`。原因不是算不出结果，而是 P0 仍处于 `BLOCKED_METRIC_RECONCILIATION`，不能把这组 headroom 当成最终放行依据。

## Oracle Teacher Selection（基于 true_AJ_RD）

| Metric | Fixed Best (CT-offline) | Oracle Selection | Gain |
|---|---:|---:|---:|
| true_AJ_RD | 0.3870 | 0.4117 | +0.0247 (+2.5pp) |
| Median re-entry | 3.71px | 2.99px | -0.72px |
| Long-occ median re-entry | 5.58px | 4.06px | -1.52px |

## 当前阻止 P1 通过的问题

1. P0 = `BLOCKED_METRIC_RECONCILIATION`，P1 不能单独 GO
2. `oracle_teacher_selection.json` 已给出 `WEAK_GO`，但 phase gate 仍需要 P0 先被正式放行
3. 后续若要把 P1 作为正式结论使用，需要在 P0 解除 blocked 后再复核一次 phase decision

## 下一步

P0 通过 → P1 再次确认 oracle teacher gain → 再判定 Go/Stop
