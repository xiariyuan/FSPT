# Attempt 0 Unified Reproduction Checklist (2026-06-13)

## 0. 目的

在正式投入任何新训练之前，先完成一次**统一外部 baseline 复现**。

这一步是强制的，原因很简单：

1. 不同论文和仓库报告的协议不完全一样。
2. 只看论文表格，容易把 `query-first / strided / d_avg / AJ` 混在一起。
3. 当前仓库后续所有尝试，都应该建立在一个**同协议横向表**上，而不是建立在多个不兼容的 README 数字上。

这份清单的目标只有一个：

**把当前仓库关心的候选 baseline 放到同一个 evaluator 下，重算同一组指标，再决定先押哪条线。**

---

## 1. 这一步结束后必须产出的东西

Attempt 0 结束时，必须有下面 5 个工件：

1. 一张 `repo-native reproduction table`
2. 一张 `unified rescoring table`
3. 每个 baseline 的 `environment + commit + checkpoint manifest`
4. 每个 baseline 导出的统一预测缓存
5. 一份明确的 go / no-go 决策

没有这 5 个工件，就不能进入 Attempt 1。

---

## 2. 冻结的候选集合

Attempt 0 只复现下面 6 个对象，不额外加新模型：

1. 当前仓库的 `CoTracker3 baseline`
2. `CoTracker3-Offline`
3. `Track-On2`
4. `Track-On-R`
5. `TAPNext++`
6. `AllTracker`

第一轮不引入：

- 3D / TAPVid-3D
- PointSt3R / SpatialTracker
- 任何新的本仓库后处理 recovery 变体
- 任何新的 reliability / calibration 训练线

---

## 3. 冻结的主指标

Attempt 0 的主指标只看：

- `TAP-Vid DAVIS`: `AJ / OA / <avg / <4px`
- `TAP-Vid Kinetics`: `AJ / OA / <avg / <4px`

辅助记录但不作为第一裁决标准：

- `repo-native d_avg / delta_avg`
- 推理速度
- 显存
- `long-occ>20` 子集的 `AJ / <avg / <4px`
- `re-entry first-frame error`
- `AJ_RD`，如果支持

这些辅助指标不能代替主表，但它们可以改变 `Attempt 1` 与 `Attempt 2` 的优先级判断。

---

## 4. 冻结的统一评测协议

### 4.1 当前仓库的统一口径

统一重算时，固定使用本仓库现有 TAP-Vid evaluator 包装层：

- [datasets/metrics.py](/gemini/code/FSPT/datasets/metrics.py)
- [datasets/tapvid_official_eval.py](/gemini/code/FSPT/datasets/tapvid_official_eval.py)
- [configs/fspt_base.yaml](/gemini/code/FSPT/configs/fspt_base.yaml)

说明：

- 当前 Attempt 0 脚手架直接调用的是 [datasets/metrics.py](/gemini/code/FSPT/datasets/metrics.py)
- 它内部再调用 vendored official core [datasets/tapvid_official_eval.py](/gemini/code/FSPT/datasets/tapvid_official_eval.py)
- [datasets_code/metrics.py](/gemini/code/FSPT/datasets_code/metrics.py) 仍可作为历史实现参考，但不再作为 Attempt 0 parity 脚本的直接入口

统一口径固定为：

- `query_mode = strided`
- `metric_resolution_mode = original`

这是当前仓库已经明确写死的“official / SOTA-comparable”口径。

### 4.2 为什么要这样锁

外部一手来源说明了当前候选之间的协议差异：

- `Track-On-R` 论文主表在 TAP-Vid 上采用 **query-first** 评测。
- `tapnet` 官方仓库同时列出 `DAVIS First (AJ)` 和 `DAVIS Strided (AJ)`，但 `TAPNext++` 当前 README 只明确列出 `DAVIS First (AJ)=65.6%`，没有给出 strided AJ。
- `AllTracker` 官方仓库 README 直接给出 `da / aj / oa`，并通过 `test_dense_on_sparse.py` 跑稠密到稀疏评测。

所以必须拆成两层：

1. **Repo-native reproduction**
   - 先确认各自仓库原生协议下的数字能跑通
2. **Unified rescoring**
   - 再把预测导入同一个 evaluator
   - 统一重算 `AJ / OA / <avg / <4px`

如果不做第二层，表面上“复现成功”，实际上还是 apples-to-oranges。

### 4.3 `d_avg / delta_avg / delta_avg^vis / AJ` 口径映射

Attempt 0 必须显式区分两类指标：

1. `repo-native reproduction`
   - 记录各仓库原生表里的指标名
   - 例如 `d_avg`、`delta_avg^x`、`delta_avg^vis`、`AJ`
2. `unified rescoring`
   - 统一只用本仓库固定 evaluator 重算：
   - `AJ / OA / <avg / <4px`

此外必须额外记录下列元信息：

- `query-first` 还是 `strided`
- `visible-only` 还是 `all-query-target frames`
- `input resolution` 还是 `original resolution`

这些数字不能直接混成一张表。

### 4.4 Evaluator parity check

Attempt 0 不允许直接假设本仓库 evaluator 与官方 TAP-Vid evaluator 完全等价。

必须新增一个强制 sanity step：

1. 选 2-3 个 DAVIS sequences
2. 取同一份 GT + 同一份 prediction cache
3. 同时运行：
   - 本仓库 evaluator wrapper：[datasets/metrics.py](/gemini/code/FSPT/datasets/metrics.py)
   - vendored official core：[datasets/tapvid_official_eval.py](/gemini/code/FSPT/datasets/tapvid_official_eval.py)
4. 记录：
   - `AJ`
   - `OA`
   - `<avg`
   - `<4px`

通过标准：

- 对齐差异应控制在 `1e-4` 到 `1e-3` 量级
- 如果出现明显偏差，先修 evaluator / adapter，不进入模型比较

---

## 5. 两张表必须同时存在

### 5.1 表 A：Repo-Native Reproduction Table

作用：

- 检查环境、checkpoint、官方脚本是否跑通
- 检查数值是否接近论文或 README 声明

每个模型都至少记录：

- repo commit
- checkpoint source
- dataset split
- repo-native metric names
- repo-native numbers
- 与官方数字的差值

### 5.2 表 B：Unified Rescoring Table

作用：

- 真正决定后续主线

所有 baseline 都必须在同一协议下比较：

- dataset: `DAVIS`, `Kinetics`
- query protocol: `strided`
- metric resolution: `original`
- metrics: `AJ / OA / <avg / <4px`

**只有表 B 决定优先级。**

同时保留一张 `protocol-bridge table`：

- `query-first + repo-native`
- `strided + original + unified rescoring`

---

## 6. 每个 baseline 需要的最小导出格式

为了做 unified rescoring，每个外部模型都必须导出统一预测缓存。

最小字段：

- `model_name`
- `repo_commit`
- `checkpoint_path`
- `checkpoint_sha256`
- `dataset_name`
- `split`
- `protocol`
- `video_id`
- `sequence_index`
- `frame_count`
- `query_points`
- `pred_tracks`
- `pred_visibility`
- `original_size`
- `model_input_size`
- `adapter_version`
- `raw_coordinate_note`

推荐统一成 `npz` 或 `pt`，但关键不是文件后缀，而是字段语义一致。

统一要求：

- `pred_tracks` 必须明确是归一化坐标还是像素坐标
- 坐标顺序必须明确是 `(x, y)` 还是 `(y, x)`
- `pred_visibility` 必须明确是 bool 还是 logits / probs
- 必须保留 query frame 索引
- 必须明确序列长度是否包含 query frame 位置本身

如果这一步不严格，统一重算没有意义。

### 6.1 Adapter sanity check

每个 baseline 在进入 unified rescoring 之前，必须通过导出适配 sanity check。

至少包含 5 项：

1. `query_points` 与 query frame 上的预测位置是否对齐
2. `(x, y)` / `(y, x)` 是否已转换正确
3. 像素 / 归一化是否已转换正确
4. `original_size` 是否为 `(H, W)` 而不是 `(W, H)`
5. 随机抽 3-5 个样本可视化，肉眼确认轨迹叠加不离谱

必须产出：

- `adapter_sanity_report.json`

### 6.2 Export schema adapter layer

不要在每个外部仓库里零散魔改导出逻辑。

推荐在当前 FSPT 仓库内维护统一适配层，每个 baseline 单独写一个 adapter，并明确记录：

- 输入坐标语义
- 输出坐标语义
- visibility 二值化规则
- query frame 对齐策略
- 序列长度对齐策略

---

## 7. 各候选的最小复现目标

### 7.1 CoTracker3 baseline

这不是外部新方向，但它是 Attempt 0 的锚点。

要求：

1. 用当前仓库现有 protocol 重跑一次
2. 生成 unified rescoring 对照行
3. 固定成后续所有 Attempt 的 base reference

如果连这一行都不冻结，后面无法判断新基线是否真有提升。

### 7.1.1 CoTracker3-Offline

`CoTracker3-Offline` 必须加入 Attempt 0，作为“当前公开强 2D baseline 锚点”。

原因：

1. 你需要知道新的候选是在赢当前仓库 online baseline，还是在赢整个公开 2D 赛道中的强基线。
2. 如果统一协议下 `Track-On-R / TAPNext++ / AllTracker` 都不如 `CoTracker3-Offline`，那后续主线决策会完全不同。

Attempt 0 对它的要求是：

1. 优先从官方 `co-tracker` 仓库获取 offline checkpoint
2. 如果 checkpoint 可跑，则纳入 repo-native 与 unified rescoring
3. 如果 checkpoint 暂时不可直接复现，也必须把论文参考行单独写进 `reference-only row`

### 7.2 Track-On2 / Track-On-R

优先级最高，所以 Attempt 0 中最先跑。

官方一手来源可直接支撑两点：

1. `track_on` 官方仓库明确提供统一 evaluation pipeline、6 个 benchmark dataloader、8 个 baseline wrapper。
2. README 直接给出 `Track-On-R` 和 `Track-On2` 的 repo-native 目标值。

按仓库 README，正确配置下应接近：

- `Track-On-R`: DAVIS `80.3`, Kinetics `71.0`
- `Track-On2`: DAVIS `79.9`, Kinetics `69.3`

这里是 `delta_avg^x` 口径，不是当前仓库主表的最终裁决口径。

Attempt 0 对它们的要求是：

1. 先复现上述 repo-native 数量级
2. 再导出预测
3. 再在本仓库 evaluator 下统一重算
4. 区分：
   - `Track-On family` 是否值得替代当前 baseline
   - `Track-On-R fine-tune` 是否真的比 `Track-On2` 有额外价值

### 7.3 TAPNext++

官方一手来源已经确认：

- `tapnet` 官方仓库把 `TAPNext++` 作为改进后的 checkpoint
- 强调 `40x longer stable tracking`
- 明确指向 long-term tracking、occlusion tracking 和 re-detection
- README 表中给出 `TAPNext++` 的 `DAVIS First (AJ)=65.6%`、`Kinetics First (AJ)=53.9%`

Attempt 0 对它的要求是：

1. 先在 repo-native 口径下确认 checkpoint 可跑
2. 再导出预测
3. 再统一重算 `strided + original`

注意：

`TAPNext++` 的价值不仅在主表数字，还在它是否在 long-occ / re-entry 场景上形成结构性优势。但这件事只能在 unified rescoring 之后再说。

### 7.4 AllTracker

官方一手来源已经明确：

- 官方仓库提供 `test_dense_on_sparse.py`
- README 直接给出 DAVIS 默认评估示例：
  - `da: 76.3`
  - `aj: 63.3`
  - `oa: 90.0`
- 更高分辨率时还给出更高的 `da / aj / oa`

Attempt 0 对它的要求是：

1. 先跑 repo-native `test_dense_on_sparse.py`
2. 验证 dense-to-sparse 评测流程通
3. 导出统一预测
4. 再重算本仓库主表指标

### 7.5 数据泄漏与公平性检查

对 `Track-On-R` 及任何真实视频微调方法，Attempt 0 应同时记录：

1. fine-tuning 数据是否与 `TAP-Vid DAVIS / Kinetics` 存在 clip overlap
2. 如果无法完全排除 overlap，必须在结果表里明确标注

---

## 8. 建议的执行顺序

Attempt 0 内部也要排序，不要五个一起乱跑。

### 8.1 Phase A：先冻结锚点

1. `CoTracker3 baseline`
2. `CoTracker3-Offline`
3. evaluator parity check

### 8.2 Phase B：第一优先级候选

4. `Track-On2`
5. `Track-On-R`

### 8.3 Phase C：第二优先级候选

6. `TAPNext++`

### 8.4 Phase D：第三优先级候选

7. `AllTracker`

原因：

- Track-On 这条线的公开代码、评测入口、真实域微调叙事最完整
- 它最像“可直接拿来替换当前主 baseline 的对象”
- TAPNext++ 更偏“长遮挡 / re-detection 专项强者”
- AllTracker 更偏“范式正交备选”和 teacher 候选

---

## 9. 每个 baseline 的 go / no-go 标准

### 9.1 通用 stop 条件

任一 baseline 只要满足以下任一条，就不要继续往 training / adaptation 扩展：

1. 两天内连 repo-native eval 都跑不通
2. 数值离官方 README / 论文差得太多，且无法定位是协议差异还是实现问题
3. 无法稳定导出统一预测缓存
4. adapter sanity check 未通过
5. evaluator parity check 未通过

### 9.2 Track-On family 与 Track-On-R

这里必须拆成两层判断。

#### A. `Track-On family` 是否值得成为新 baseline

继续条件：

1. `max(Track-On2, Track-On-R)` 在 unified rescoring 后优于当前 `CoTracker3 baseline`
2. 至少在 `DAVIS` 上满足：
   - `AJ` 不低
   - `OA` 不明显下降
   - `<avg` 或 `<4px` 至少一个明确更好

#### B. `Track-On-R fine-tune` 是否值得继续

继续条件：

1. `Track-On-R` 相比 `Track-On2` 有稳定增益；或
2. 在 `Kinetics` / 更接近真实视频的 benchmark 上更强，且 `DAVIS` 不显著退化

如果 B 不成立，应停止的是 `Track-On-R fine-tune 主线`，不是整个 `Track-On family baseline replacement`。

### 9.3 TAPNext++

继续的最低条件：

1. 官方 checkpoint 可稳定运行
2. unified rescoring 后至少接近或超过当前 `CoTracker3` 水平
3. long-occ / re-entry 上能看到结构性优势迹象

如果 unified rescoring 后主表不 competitive，且长遮挡场景也没有额外亮点，就先暂停。

### 9.4 AllTracker

继续的最低条件：

1. repo-native DAVIS 结果接近 README 示例量级
2. unified rescoring 后没有明显掉出竞争梯队

如果它在主表不够强，但作为 teacher 可能有价值，可以保留为“teacher-only asset”，不一定进入主 baseline 竞争。

---

## 10. Attempt 0 结束时的决策规则

Attempt 0 应区分：

1. `partial decision`
2. `final decision`

Attempt 0 最终可输出下面几种结论：

### 10.1 结论 A

`Track-On-R` 在 unified rescoring 后最强。

动作：

- 进入 Attempt 1：`Track-On-R`

### 10.2 结论 B

`Track-On-R` 与 `Track-On2` 没拉开，但 `TAPNext++` 在长遮挡 / re-entry 上更有潜力。

动作：

- 直接把 Attempt 2 提前

### 10.3 结论 C

`Track-On-R`、`Track-On2`、`TAPNext++` 都没有形成足够清晰的统一协议优势，而 `AllTracker` 也只是在密集评测下好看。

动作：

- 暂停后续大规模训练
- 先补协议和导出适配问题
- 不进入任何“凭感觉继续练”的分支

### 10.4 结论 D

`Track-On-R` overall 最强，但 `TAPNext++` 在 `long-occ>20` / re-entry 子集上显著更优。

动作：

- 以 `Track-On-R` 作为 `Attempt 1` 主 baseline
- 同时把 `TAPNext++` 作为 long-occ 专项对照线

### 10.5 结论 E

所有候选在 unified rescoring 下都不如 `CoTracker3-Offline`。

动作：

- 以 `CoTracker3-Offline` 作为参考强锚点
- 暂不默认 `Track-On-R` 接管主线

### 10.6 结论 F

多个候选在 unified rescoring 下差距都很小，没有单一压倒性胜者。

动作：

- 不急于单押一个模型
- 优先保留最强 2-3 个模型作为 teacher pool 候选

---

## 11. 这一步明确不做什么

Attempt 0 不做：

- 新训练
- 新蒸馏
- 新 pseudo-label 数据清洗
- 新 verifier 设计
- 新后处理 recovery
- 3D 新方向

Attempt 0 的意义是：

**先把“谁值得成为新 baseline”这件事做实。**

---

## 12. 当前推荐的实际解释

截至 `2026-06-13`，最合理的执行口径是：

1. 当前仓库已经充分证明，继续在 `CoTracker3 + 后处理` 空间里找增益先验很低。
2. 下一步不是再修当前仓库的 recovery trick，而是先完成外部强 baseline 的统一复现。
3. Attempt 0 完成后，再按：
   - `Track-On-R`
   - `TAPNext++`
   - `AllTracker`
   的顺序逐个尝试。

补充说明：

- `Track-On-R Week 1` 只能视为 `Attempt 0` 的 `Track-On family` 子阶段
- 正式主线锁定，原则上应等 `CoTracker3-Offline / TAPNext++ / AllTracker` 的统一表补齐

---

## 13. 一手来源

- Track-On 官方仓库：<https://github.com/gorkaydemir/track_on>
- Track-On-R 论文：<https://arxiv.org/abs/2603.12217>
- TAP / TAPNext / TAPNext++ 官方仓库：<https://github.com/google-deepmind/tapnet>
- TAPNext++ 论文：<https://arxiv.org/abs/2604.10582>
- AllTracker 官方仓库：<https://github.com/aharley/alltracker>
- AllTracker 论文：<https://arxiv.org/abs/2506.07310>
- TAP-Vid 官方页：<https://tapvid.github.io/>
