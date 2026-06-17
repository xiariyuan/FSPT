# Track-On2 Memory Gating Fix Note (2026-06-16)

## 1. 先说结论

这个问题 **不属于“应该停止的失败方向”**。

它属于：

- 当前最强 runnable baseline `Track-On2 + DINOv3` 的
- 一个真实的
- 结构性实现缺陷 / 推理逻辑 bug

因此正确动作不是“停掉这条线”，而是：

1. 先修这个 bug
2. 再重新评估 overall DAVIS 和 long-occ 子指标
3. 再决定 Track-On2 当前主线是否进一步推进

---

## 2. 之前说“需要停止”的，到底指什么

之前应该停止、或者已经高置信关闭的是这些方向：

- CoTracker3 后处理 recovery / rerank
- local geometry rerank
- DINO patch rerank
- DiReCT / direct recovery attention
- neighbor-deviation pseudo-label filtering
- reliability / calibration 作为主指标提升主线
- RouteA integrated relocal / verifier

这些方向的共同点是：

- 都是在 **已有 tracker 特征空间上加后处理 patch**
- 不是在修一个明确的实现错误
- 即使能看到 oracle ceiling，也无法在真实指标上稳定泛化

所以“停掉”的是这类 **后处理补丁方向**。

---

## 3. 为什么这次 Track-On2 memory 问题不能归到“停止方向”

这次 Claude 提出的根因是：

- `trackon.py` 中 `point_memory` 每帧无条件 FIFO 写入
- 遮挡帧也照写
- `forward()` / `forward_online()` / streaming predictor 都没有用 `v_logit` 阻止低置信 query 刷新 memory

这和前面那些“失败方向”不是一类问题。

这是一个 **主干模型推理状态更新逻辑** 的问题，性质更接近：

- memory contamination bug
- online state management bug
- long-occlusion regime 下的结构性错误

它不是“又试了一个花哨 rerank 但没泛化”。

它是“当前最强 baseline 本身可能被一个错误实现拖低了 long-occ 指标”。

---

## 4. 根因判断

当前代码中，旧逻辑的核心问题是：

1. query 一旦激活，后续每帧都产生 `q_t`
2. 不论该帧是否可见，都会把 `q_t` 写入 FIFO memory
3. 长遮挡时，错误位置对应的 feature 会持续覆盖历史 memory
4. 当 `M_i = 72` 左右时，整段记忆被错误特征滚满
5. 首帧锚点 feature 也会被逐步 roll 出

因此长遮挡子指标出现明显退化，是一个完全合理的结果。

这个判断与当前代码一致。

相关代码入口：

- `baselines/track_on/model/trackon.py`
- `baselines/track_on/model/trackon_predictor.py`

本轮修复位置：

- `trackon.py`: `self.delta_v` 注入模型侧
- `trackon.py`: 新增 `_update_point_memory(...)`
- `trackon.py`: `forward()` 改为 selective memory write
- `trackon.py`: `forward_online()` 改为 selective memory write
- `trackon_predictor.py`: streaming predictor 改为 selective memory write

---

## 5. 当前修复策略

修复后的 memory 写入条件是：

- 当前 query **预测为可见**，则允许刷新 memory
- 当前 query **是刚初始化的新 query**，则允许首帧写入 memory

也就是说：

- 可见帧继续更新 memory
- 遮挡期低置信帧不再无条件污染 FIFO
- 首帧锚点 feature 不会因为门控而丢失初始化写入

这个策略的目标不是“发明新方法”，而是：

- 恢复 Track-On2 memory 机制本来应该具备的行为

---

## 6. 这件事对项目方向的真实含义

这次修复 **不会推翻** 之前的战略判断：

- CoTracker3 后处理 patch 空间大体已经穷尽
- 下一阶段主要变化层级应该是更强 base tracker / 多 teacher / 新训练范式

但这次修复 **会改变** 我们对 Track-On2 当前强弱的判断边界：

- 如果修复后 long-occ 显著回升，说明 Track-On2 当前可能被实现问题低估了
- 如果修复后 overall 稳定、long-occ 提升，那 Track-On2 主线应继续推进
- 如果修复后整体提升有限，说明当前主线仍然只是“更好的 baseline”，但不是新的大方向

所以：

> 这不是“停掉方向”，这是“先修 bug，再重新判断 Track-On2 当前价值”。

---

## 7. 接下来必须做的验证

修复后不能只看代码，要重新做评估。

最少应完成：

1. `trackon2_dinov3` repo-native DAVIS overall 指标重跑
2. long-occ 子指标重跑
3. 检查 overall 是否稳定，防止 long-occ 提升但普通样本退化
4. 如 repo-native 通过，再重新走 unified bridge / parity / rescore
5. 产出 before/after 对比文档

建议输出：

- `outputs/trackon2_memory_gating_fix_repo_native_before_after.json`
- `outputs/trackon2_memory_gating_fix_longocc_before_after.json`
- `docs/trackon2_memory_gating_validation_2026-06-16.md`

---

## 8. 当前最准确的表述

当前应该这样说：

- “需要停止”的，是 CoTracker3 后处理 patch 路线
- “需要继续”的，是 Track-On2 当前主线里的 memory contamination bug 修复与重新验证

这两件事不是同一类问题，不能混为一谈。

