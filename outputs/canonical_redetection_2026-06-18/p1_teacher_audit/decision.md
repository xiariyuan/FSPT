# P1 Decision

**日期**: 2026-06-19

**Decision**: `WEAK_DIAGNOSTIC_ONLY`

## 状态说明

`oracle_teacher_selection.json` 已经计算出 true `AJ_RD` headroom，但 phase gate 仍保持在 **WEAK_DIAGNOSTIC_ONLY**。原因：

1. **Oracle teacher AJ_RD gain = +2.5pp，低于原定 5pp gate**
2. Oracle teacher selection 使用 GT error，不能证明可训练 selector 可实现
3. 即使 +2.5pp gain，真实可实现收益可能只有 20-50%（+0.5pp ~ +1.2pp）
4. P0 仍处于 `BLOCKED_METRIC_RECONCILIATION`

## Oracle Teacher Selection（基于 true_AJ_RD）

| Metric | Fixed Best (CT-offline) | Oracle Selection | Gain |
|---|---:|---:|---:|
| true_AJ_RD | 0.3870 | 0.4117 | +0.0247 (+2.5pp) |
| Median re-entry | 3.71px | 2.99px | -0.72px |
| Long-occ median re-entry | 5.58px | 4.06px | -1.52px |

## Go/Stop 判定

| Criteria | 要求 | 实测 | 判定 |
|---|---:|---:|---:|
| Oracle AJ_RD gain >= 5pp | Strong GO | +2.5pp | ❌ |
| Oracle AJ_RD gain >= 2pp | Weak GO | +2.5pp | ✅ Weak |

## 当前允许的动作

- **仅诊断用途**: teacher diversity 有弱信号，可用于后续诊断参考
- **不允许**: 进入 P4 训练，不能作为训练启动依据

## 下一步

P0 通过后可用 true AJ_RD 复核，但 +2.5pp 低于训练 gate +5pp，复核后也**不能**自动放行训练。
