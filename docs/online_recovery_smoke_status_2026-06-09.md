# Online Recovery Smoke Status (2026-06-09)

## Smoke 配置

- Config: `configs/fspt_online_recovery_smoke_eval256.yaml`
- Checkpoint: `checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth`
- Smoke 脚本: `scripts/debug_online_recovery_smoke.py`
- 审计结果: `outputs/online_recovery_smoke_audit.json`

## 链路验证结果

| 检查项 | 状态 | 说明 |
|--------|------|------|
| online_recovery 配置入口 | **生效** | `requested=True`, `mode=tracker_conditioned` |
| 模型构建 | **成功** | 正确继承完整 config chain（11层） |
| Forward pass | **成功** | tracks shape [2,4,8,2], visibility [2,4,8] |
| online_recovery_summary | **存在** | 包含 mode, feature_source, template_mode 等 9 个字段 |
| verifier_scores | **非空** | 64/64 nonzero |
| relocal_mask | **全零** | 预期行为：合成输入无真实遮挡恢复事件 |
| retracking | **未触发** | 因为 relocal_mask 全零，retracking 条件未满足 |

## 已稳定输出的 Recovery 字段

```
online_recovery_requested:  True
online_recovery_mode:       tracker_conditioned
online_recovery_summary:    {requested, mode, feature_source, template_mode,
                             template_bank_size, min_occlusion_len, frames_after,
                             vis_threshold, retracking_enabled}
verifier_scores:            (B, N, T) tensor, nonzero
```

## Retracking 触发条件

retracking 未触发是因为当前输入是合成数据（随机 noise），没有真实的遮挡恢复事件。在真实数据上，retracking 的触发条件是：

1. `relocal_mask` 非空（relocalization 找到了有效候选）
2. `relocal_conf` 超过 `min_conf` 阈值
3. 模型的 `_maybe_apply_retracking()` 被调用

这些条件在 `forward()` 主路径中已经接入（line 5104-5122 of cotracker_refiner.py）。

## 当前链路还缺什么

1. **真实数据 smoke**：需要在 TAP-Vid DAVIS 或 PointOdyssey 上跑一次真实 eval，验证 relocal_mask 在有遮挡恢复事件时非空，且 retracking 真正触发
2. **DINO recovery feature branch**：`feature_source` 目前只支持 `cotracker` 和 `geo`，DINO 分支是 placeholder
3. **Learned trigger**：当前用规则触发（min_occlusion_len + vis_threshold），learned trigger 还未接入
4. **强特征 recovery branch 接口**：见下方

## 强特征 Recovery Branch 接口状态

在 `models/cotracker_refiner.py` 中：

- `relocalization_feature_source` 字段已存在（line 1791-1805）
- 当前支持：`cotracker`, `geo`
- DINO 分支：**placeholder**，`feature_source not in ("cotracker", "geo")` 时会 warning 并禁用 relocalization
- 配置路径：`refiner.online_recovery.search.feature_source` 已生效

下一步需要：
- 在 `relocalization_feature_source` 的分支中加 `dino` 选项
- 实现 DINOv2 feature extraction 作为 recovery-only 分支
- 与主 tracker 的 cotracker fnet feature 解耦
