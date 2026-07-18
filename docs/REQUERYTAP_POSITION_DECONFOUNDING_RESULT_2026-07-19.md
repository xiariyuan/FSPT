# ReQueryTAP Gate E：位置去混杂身份审计（2026-07-19）

## 结论

移除 TAPNext++ `image_pos_emb` 后，单帧和多帧身份匹配均未跨场景稳定改善。Gate E 决策为：

```text
CLOSE_TAPNEXT_PATCH_IDENTITY_ROUTE
```

## 结果

### conv-only 单帧

- 仅 1/5 场景优于原 position-encoded frame-0 基线；
- 场景中位变化 `-20.8801px`；
- bootstrap CI 下界 `-28.1104px`；
- aggregate hit@16 下降 `10.93` 个百分点；
- 最差回退 `32.6113px`。

### conv-only 8 帧最大证据

- 仅 1/5 场景改善；
- 场景中位变化 `-29.7856px`；
- bootstrap CI 下界 `-32.6336px`；
- aggregate hit@16 下降 `8.47` 个百分点；
- 最差回退 `34.8919px`。

### 多视图增量

相对于 conv-only frame-0，`conv_max8` 仅 2/5 场景改善，中位变化 `-2.5610px`，最差回退 `18.3008px`。

## 解释边界

- fresh-query 生命周期本身没有被否定：Gate A 的正确位置重生仍在 16/17 场景产生正未来收益；
- 被关闭的是“用 TAPNext++ 输入 patch token 直接承担长期精确点身份”的路线；
- 后续不得继续调 TAPNext patch 聚合、温度、rank、位置编码比例或同类 MLP；
- 唯一允许的新方向是独立的持久身份描述器，并且仍须先通过 fit-only 因果/信息 gate。

PointOdyssey model-validation、holdout、test、DAVIS 和 Kinetics 均未读取。
