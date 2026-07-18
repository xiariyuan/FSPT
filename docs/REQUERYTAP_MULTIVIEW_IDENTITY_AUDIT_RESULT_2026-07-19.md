# ReQueryTAP Gate D：多视图身份聚合结果（2026-07-19）

## 结论

预注册主方案 `max8` 未通过 Gate D，直接聚合 TAPNext++ `lin_proj + image_pos_emb` 点特征的路线关闭，不允许据此训练第二个定位器。

## 固定协议

- 仅使用 5 个既定 PointOdyssey fit-dev 场景；
- 每场景沿用冻结缓存中的 256 个点身份；
- 身份记忆来自遮挡前帧 0–7 的 GT-visible 精确点特征；
- 目标为帧 40–55；
- 主方案在结果出现前固定为跨视图最大余弦 `max8`；
- 未读取 model-validation、holdout、test、DAVIS 或 Kinetics。

## Gate D 结果

`max8`：

- 2/5 场景中位误差改善；
- 场景中位改善 `-0.6601px`；
- scene-bootstrap 95% CI `[-6.1109, +11.0824]px`；
- aggregate hit@16 仅提升 `+0.02477`；
- 最差场景回退 `-10.4805px`；
- 决策：`CLOSE_MULTI_VIEW_FEATURE_AGGREGATION`。

其他预注册变体同样不稳定：

- `mean8`：3/5 场景改善，中位 `+6.9287px`，CI 下界 `-10.1722px`，最差回退 `-17.1424px`；
- `logsumexp8`：2/5 场景改善，中位 `-0.9667px`，CI 下界 `-7.0877px`。

因此不能在看到结果后把主方案切换为 `mean8`。

## 诊断边界

该失败关闭的是“直接聚合包含绝对位置编码的 patch token”，不是 fresh-query token 生命周期本身。Gate A 已证明正确位置重生具有稳定未来收益。下一步只允许做一次不训练的 position-deconfounding 审计：移除 `image_pos_emb` 后检查多视图视觉身份是否出现跨场景稳定信号；若仍失败，则关闭当前 TAPNext++ patch-feature 身份定位路线。
