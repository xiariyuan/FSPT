# P3A Decision

**日期**: 2026-06-18
**结论**: `SMOKE_STOP`

## Smoke 结果

| 指标 | 值 |
|---|---:|
| Labels | 62 |
| Videos | 1 |
| Re-entry events | 62 |
| Long-occ (>=16) ratio | 0.4839 |
| Pseudo-label median error | 46.15px |
| Pseudo-label <16px | 8.06% |
| CT-offline median error | 5.99px |
| Better than CT-offline | 3.23% |
| Max video share | 100% |

## 判定

- smoke 伪标签精度远低于 CT-offline raw
- 样本完全被单视频支配
- FB / flow consistency 当前未接入，不能作为有效质量提升证据
- 现有 smoke 结果不足以支持进入 full P3A rollout

## 下一步

停止 `P3A` 的 feature-based pseudo-label 主线，不进入 `P4`。
保留本次 smoke 作为负结果记录。
