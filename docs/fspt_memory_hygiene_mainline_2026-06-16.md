# FSPT Memory Hygiene Mainline (2026-06-16)

## 1. 当前主线的一句话版本

我们的核心假设是：

> 长遮挡 point tracking 的关键瓶颈之一，不是缺少一个后处理 reranker，而是遮挡期间 tracker 持续把低置信状态写入 memory，导致 state contamination；因此应优先验证 visibility-aware memory retention，并将其作为 long-occ / re-entry 改进的切入点。

这句话里有两个关键词：

- `state contamination`
- `visibility-aware memory retention`

---

## 2. 这条主线不是什么

这条主线 **不是**：

- 再做一个 CoTracker3 的 post-hoc rerank
- 再做一层 local selector / geometry rerank / direct recovery attention
- 再把 verifier 当成输出空间补丁接在现有轨迹后面

这些路线当前已经高置信关闭，原因很一致：

- 它们作用在输出空间
- 但如果 tracker 内部状态已经被污染，候选本身就会越来越差
- 结果是有 oracle ceiling，但真实方法无法稳定泛化

也就是说，**问题不再是“怎么从坏候选里挑一个更好的”**，而是：

> **怎么避免 tracker 在遮挡期把自己写坏。**

---

## 3. 为什么现在把问题收敛到 memory hygiene

我们目前对 Track-On2 的代码审查发现了一个重要事实：

- `point_memory` 在旧实现里按 FIFO 每帧写入
- 遮挡帧也照写
- `forward()` / `forward_online()` / streaming predictor 都没有用 `v_logit` 阻止低置信 query 刷新 memory

这意味着：

1. query 一旦激活，后续每帧都会生成新的 `q_t`
2. 即使该帧处于遮挡或错误漂移状态，也会把 `q_t` 写回 memory
3. 长遮挡时，错误特征会连续覆盖 FIFO memory
4. 当 `M_i=72` 左右时，首帧锚点和可见期的可靠特征可能被逐步滚出
5. re-entry 时 decoder 依赖的是已经污染过的状态

因此 long-occ 失败可以被解释为：

- 不是单纯“定位有点漂”
- 而是 **状态持续被错误更新**

相关代码：

- `baselines/track_on/model/trackon.py`
- `baselines/track_on/model/trackon_predictor.py`

相关修复说明：

- `docs/trackon2_memory_gating_fix_note_2026-06-16.md`

---

## 4. 项目主线与论文主线要分开

### 4.1 项目主线

项目层面的方向判断可以写成：

1. 停止继续在 CoTracker3 后处理空间打补丁
2. 保留 `Track-On2 + DINOv3` 作为当前最强 runnable baseline
3. 优先验证 long-occ failure 是否主要来自 memory contamination
4. 若成立，再扩展到 learned retention / anchor memory / teacher-guided training

### 4.2 论文主线

论文不能只写成“我们换了更强模型”或“我们做了个 gate”。

更合理的论文主线应当落在：

- state contamination diagnosis
- visibility-aware memory retention
- persistent anchor retention / dual-memory design
- long-occ / re-entry 指标上的针对性改进

也就是说：

- “换 base / 换 teacher”是项目策略
- “memory hygiene mechanism”才可能成为论文方法

---

## 5. 当前最重要的决定性实验

在 all-in 之前，应该先做一个最便宜但判别力最强的实验：

## Oracle Memory Gating

### 定义

推理时使用 **GT visibility** 门控 memory 写入：

- 如果 GT visible：允许写 memory
- 如果 GT invisible：冻结 / 不写 memory
- 如果 query 刚初始化：允许首帧写入

### 为什么先做它

这个实验回答的是：

> memory contamination 到底是不是当前主线真正有 headroom 的瓶颈？

### 判别逻辑

- 如果 **oracle gate 几乎没提升**
  - 说明瓶颈不主要在 memory
  - 重心应该转向 re-acquisition / global re-detection / 更强 base model

- 如果 **oracle gate 明显提升 long-occ**
  - 说明 memory contamination 是强根因之一
  - 值得继续做 predicted gate / learned gate / anchor memory

### 为什么它比直接做 learned gate 更重要

因为它先回答“这条线值不值得继续”，而不是一开始就往一个可能没 headroom 的方向堆复杂度。

---

## 6. 推荐的最小实验顺序

### Step 1: contamination audit

先不求提分，只求证据。

建议记录：

- 每帧 `v_logit`
- 每帧预测误差
- 每次写入 memory 的 feature
- memory slot 与 initial anchor feature 的 cosine similarity
- 遮挡长度
- re-entry 后若干帧误差

要验证的关系：

- occlusion length 增加时，memory-anchor similarity 是否下降
- similarity 下降时，re-entry error 是否上升
- long-occ failure 是否对应更严重的状态污染

这一步如果成立，后面无论论文还是内部判断，都会有很强的机制证据。

### Step 2: oracle gate

这是本周最高优先级实验。

建议重点看：

- long-occ AJ
- long-occ delta_avg
- re-entry first-frame error
- overall DAVIS AJ 是否被明显伤害

### Step 3: predicted visibility gate

如果 oracle gate 有信号，再做 predicted gate。

建议不要第一版就用 hard gate，更稳妥的是 soft write / blend：

```text
memory_new = alpha * q_new + (1 - alpha) * memory_old
alpha = f(v_logit, maybe u_logit)
```

原因：

- hard gate 容易死锁
- 一旦 visibility 被低估，memory 可能长期不更新

### Step 4: persistent anchor slot

如果 gate 有效果，再引入一个不滚动的 anchor slot：

```text
memory_for_attention = [persistent_anchor, rolling_short_term_memory]
```

这一步的目标是验证：

- long-occ 的问题是否部分来自首帧身份特征被 FIFO 滚出

### Step 5: verifier / teacher 入环

这属于中期扩展，不是第一个实验。

只有前面确认 memory hygiene 确实有 headroom，才值得把 verifier-guided state selection 或 multi-teacher pseudo-labeling 接进来。

---

## 7. 成功 / 失败分叉条件

### 如果 oracle gate 成功

可以进入下一阶段：

- learned retention gate
- soft memory write
- persistent anchor slot
- dual-memory architecture

这时可以把方法名收敛成：

- `Visibility-aware Memory Retention`
- 或 `Memory Hygiene for Long-Occlusion Tracking`

### 如果 oracle gate 失败

说明 memory contamination 不是主要瓶颈。

这时应当把注意力转向：

- re-acquisition / re-detection
- 更强 base tracker
- Track-On-R / TAPNext++ / AllTracker 这类更换问题表示或训练范式的路线

也就是说，**oracle gate 是这条主线的分水岭实验**。

---

## 8. 当前最准确的项目陈述

正式版：

> 我们的核心假设是：长遮挡 point tracking 的失败，关键瓶颈之一来自遮挡期间的时序 memory contamination，而不是缺少一个后处理 reranker。基于这一判断，短期应优先验证 visibility-aware memory retention 与 persistent anchor retention 是否能够改善 long-occ / re-entry；中期再将该机制与更强 base tracker 及多 teacher / verifier-guided training 结合。

短版：

> 不是再发明一个后处理补丁，而是先让 tracker 在遮挡期间保持干净记忆。

---

## 9. 当前结论

到 2026-06-16 为止，我们最稳的说法不是：

- “我们已经找到最终方法”

而是：

- “我们已经把问题从现象层收敛到了一个可验证的机制假设”

换句话说：

> **当前最值得继续验证的主线，不是输出空间补丁，而是 memory hygiene。**

