# Online Recovery DINO Branch Status (2026-06-09)

## 当前状态

`DINO recovery-only` 分支已经从“链路可跑但 conf 全零”推进到“真实数据上可触发、可给出非零 conf、可驱动 retracking”。

当前可复现的主审计文件：

- `outputs/online_recovery_dino_real_eval_audit.json`
- `outputs/online_recovery_real_eval_lowconf10_audit.json`

这两个文件使用同一 checkpoint、同一 TAP-Vid DAVIS 子集、同一 `max-batches=10`、同一 `min_conf=0.001`，可以做同口径对照。

## 已完成的代码修复

1. `models/recovery_features.py`
   - 提供 `DINORecoveryExtractor`
   - 支持 `feature_map()` / `encode_patch()` / `encode_patches_batch()`

2. `models/cotracker_refiner.py`
   - `feature_source: dino` 已接入 `online_recovery`
   - 修复了 DINO feature map 的 tensor layout，使其与 recovery 路径统一为 `(B, T, H, W, C)`
   - 在 `_compute_global_relocalization_step()` 中为 `feature_source == "dino"` 增加独立的 raw cosine search 路径
   - DINO recovery 分支不再复用为 CoTracker 特征训练的 `map_proj/query_proj`

## 同口径对照结果（DAVIS, 10 batches, min_conf=0.001）

| 指标 | CoTracker | DINO |
|------|-----------|------|
| audit 文件 | `online_recovery_real_eval_lowconf10_audit.json` | `online_recovery_dino_real_eval_audit.json` |
| total queries | 1856 | 1856 |
| relocal_mask queries | 74 | 74 |
| relocal_mask elements | 75 | 75 |
| relocal_conf mean | 0.00208 | 0.01797 |
| relocal_conf max | 0.00252 | 0.01882 |
| retracking triggered batches | 5 | 5 |
| retracking mask nonzero total | 74 | 74 |

## 正确解读

1. `relocal_mask` 完全一致不是巧合。
   - `relocal_mask` 来自 base visibility 的 re-appearance 规则，不依赖 `feature_source`
   - 因此它相同，说明比较口径一致，不说明 DINO 在“触发率”上更强

2. `DINO` 的提升体现在 `relocal_conf`，不是体现在 `mask`
   - mean: `0.01797 / 0.00208 = 8.6x`
   - max: `0.01882 / 0.00252 = 7.5x`

3. 当前 `retracking` 触发数相同，不能据此宣称两者一样强
   - 这里的 `min_conf=0.001` 低于 CoTracker 和 DINO 的所有 masked conf
   - 所以两者都会“全部通过”该门槛
   - 这只说明 pipeline 已被打通，不说明最终 recovery 质量已经相同

## 关键结论

| 结论 | 状态 |
|------|------|
| 之前 DINO `relocal_conf=0` 是实现 bug，不是方法结论 | 已确认 |
| 修复后 DINO 可给出稳定非零 conf | 已确认 |
| DINO conf 显著高于 CoTracker conf | 已确认 |
| DINO 已经在真实数据上驱动 retracking | 已确认 |
| DINO 一定带来更好的 recovery 精度/AJ | 否，当前证据不支持 |

## 最新更新

后续 anchor 诊断与 `argmax / temperature` 消融表明：

- DINO 确实提供更强 `relocal_conf`
- 但在当前 query/template 构造与 raw cosine search 下，生成的 anchor 与 base tracker 位置几乎相同
- `argmax` 与默认 `softargmax` 几乎等价
- 降低 temperature 会显著恶化 anchor 质量

因此，当前最合理的结论是：

- `DINO` 改善了 policy activation
- 但没有改善几何 correction

完整状态请见：

- `docs/online_recovery_status_report_2026-06-10.md`
