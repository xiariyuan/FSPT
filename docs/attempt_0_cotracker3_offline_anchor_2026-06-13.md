# Attempt 0 CoTracker3-Offline Anchor (2026-06-13)

## 0. 角色

这份短文档只做一件事：

**明确 `CoTracker3-Offline` 在 Attempt 0 中必须作为强公开 2D 锚点出现。**

它不是新的第一优先级方向，但它是统一复现表中不能缺的一行。

## 1. 为什么必须加这一行

如果 Attempt 0 只比较当前仓库 `CoTracker3 baseline` 与几个外部候选，你只能回答：

> “这些候选里谁比当前仓库 online baseline 更强？”

你回答不了：

> “这些候选是否已经超过当前公开强 2D point tracking baseline？”

`CoTracker3-Offline` 的意义就是提供这个锚点。

## 2. 在 Attempt 0 中的职责

`CoTracker3-Offline` 只承担 3 个职责：

1. 作为 `reference strong 2D baseline`
2. 约束 `Track-On-R / TAPNext++ / AllTracker` 的真实位置判断
3. 提供一个“如果所有候选都没有明显更强时”的 fallback 参考

## 3. 需要记录的内容

Attempt 0 对 `CoTracker3-Offline` 至少要记录：

1. checkpoint 来源
2. repo commit
3. repo-native 指标口径
4. unified rescoring 指标
5. 如果无法直接运行，至少保留论文参考行

## 4. 口径注意事项

`CoTracker3-Offline` 经常会伴随两类数字同时被引用：

1. `AJ`
2. `delta_avg^vis`

它们不能混成同一列。

## 5. 如果 checkpoint 暂时不可直接复现

如果因为环境、依赖、权重获取等原因，Attempt 0 第一轮暂时无法直接跑通 `CoTracker3-Offline`，也不能把它完全删掉。

正确处理方式：

1. 在总表中保留 `reference-only row`
2. 明确标注：
   - `not yet directly reproduced`
   - `paper/reference only`
3. 不让这行参与严格统一排序，但让它继续作为解释性强锚点存在

## 6. 与 Track-On-R 的关系

`Track-On-R` 和 `CoTracker3-Offline` 不是互斥的。

它们代表的是同一大类研究方法中的两种公开强实现：

- 更强 base model
- 更强 teacher / pseudo-labeling / real-video adaptation

## 7. 实际结论

Attempt 0 如果没有 `CoTracker3-Offline`，最终结论可能失真。

## 8. 一手来源

- CoTracker 官方仓库：<https://github.com/facebookresearch/co-tracker>
- CoTracker3 论文：<https://arxiv.org/abs/2410.11831>

