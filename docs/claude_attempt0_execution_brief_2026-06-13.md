# Claude Attempt 0 Execution Brief (2026-06-13)

## 0. 任务目标

你的任务不是继续讨论研究方向，而是**执行 Attempt 0**。

Attempt 0 的唯一目标是：

**在统一协议下复现并重算外部强 baseline，产出一张可信的横向表，然后再决定谁值得成为新的主 baseline。**

你不应该做：

- 新方法设计
- 新训练
- reliability / calibration 新实验
- 当前 CoTracker3 后处理变体

你应该做：

1. 冻结评测协议
2. 冻结锚点 baseline
3. 复现外部 baseline
4. 导出统一预测
5. 统一重算
6. 输出 partial / final decision

## 1. 必读文档

开工前先完整阅读下面文件：

1. [new_direction_ranking_2026-06-13.md](/gemini/code/FSPT/docs/new_direction_ranking_2026-06-13.md)
2. [attempt_0_unified_reproduction_checklist_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_unified_reproduction_checklist_2026-06-13.md)
3. [attempt_0_cotracker3_offline_anchor_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_cotracker3_offline_anchor_2026-06-13.md)
4. [attempt_0_status_template_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_status_template_2026-06-13.md)
5. [trackonr_execution_checklist_2026-06-13.md](/gemini/code/FSPT/docs/trackonr_execution_checklist_2026-06-13.md)
6. [experiment_review_table_2026-06-09.md](/gemini/code/FSPT/docs/experiment_review_table_2026-06-09.md)

## 2. 冻结规则

### 2.1 候选集合

Attempt 0 只处理这 6 行：

1. 当前仓库 `CoTracker3 baseline`
2. `CoTracker3-Offline`
3. `Track-On2`
4. `Track-On-R`
5. `TAPNext++`
6. `AllTracker`

### 2.2 主表协议

统一主表固定：

- dataset:
  - `TAP-Vid DAVIS`
  - `TAP-Vid Kinetics`
- query mode:
  - `strided`
- metric resolution:
  - `original`
- metrics:
  - `AJ`
  - `OA`
  - `<avg`
  - `<4px`

### 2.3 辅助子表

必须额外保留：

- `long-occ>20` 的 `AJ / <avg / <4px`
- `re-entry first-frame error`
- `AJ_RD`，如果支持

## 3. 第一优先级工作顺序

### Step 0: Day 0 前置核验

必须先完成：

1. 核对 `track_on` README 是否仍公开列出 `Track-On2 / Track-On-R`
2. 核对对应论文链接和 arXiv ID
3. 测试 `DINOv3` 访问是否可用
4. 冻结当前仓库 `CoTracker3 baseline`

如果 `DINOv3` 在 `24-48h` 内不可用：

- 不要空等
- 暂停 `Track-On-R`
- 并行切到 `TAPNext++`

### Step 1: 锚点与 evaluator

必须先完成：

1. 用当前仓库 evaluator 冻结 `CoTracker3 baseline`
2. 尽量补上 `CoTracker3-Offline`
3. 做 `evaluator parity check`

输出：

- `metric_parity_report.json`

如果 parity check 失败：

- 不许进入外部 baseline 比较
- 先修 evaluator / adapter

### Step 2: Track-On family 子阶段

按 [trackonr_execution_checklist_2026-06-13.md](/gemini/code/FSPT/docs/trackonr_execution_checklist_2026-06-13.md) 执行，但记住：

- 它只是 Attempt 0 子阶段
- Week 1 只能输出 `partial decision`

必须产出：

- `trackon_adapter_sanity_report.json`
- `track_on2_repo_native_metrics.json`
- `track_onr_repo_native_metrics.json`
- `track_on_unified_rescoring.json`
- `week1_partial_decision.md`

### Step 3: TAPNext++

如果 Track-On 线顺利：

- 继续补 TAPNext++ repo-native reproduction
- 再进入 unified rescoring

如果 Track-On 线被 DINOv3 卡住：

- 直接把 TAPNext++ 提前

### Step 4: AllTracker

AllTracker 不应无限后置。

至少要完成：

1. repo-native DAVIS reproduction
2. unified rescoring DAVIS

如果它很强，再补 Kinetics。

## 4. Adapter / Export 规范

不要在外部 repo 里到处魔改。

在当前 FSPT 仓库内维护统一 adapter 层，并为每个 baseline 明确：

1. 输入坐标语义
2. 输出坐标语义
3. 像素 / 归一化转换
4. `(x, y)` / `(y, x)` 转换
5. visibility 二值化规则
6. query frame 对齐方式
7. 序列长度对齐方式

最低导出字段：

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

## 5. 必做 sanity checks

### 5.1 Adapter sanity

每个 baseline 必须通过：

1. query frame 位置对齐检查
2. `(x, y)` / `(y, x)` 检查
3. 像素 / 归一化检查
4. `(H, W)` / `(W, H)` 检查
5. 3-5 个样本可视化

### 5.2 Evaluator parity

同一份 GT + prediction cache：

- 用本仓库 evaluator 算一遍
- 用官方 evaluator 算一遍

差异超过合理量级就先修，不比较模型。

### 5.3 Kinetics quick smoke

Kinetics 可能很大。

如全量太慢：

- 先跑 `100 clips quick check`
- 再决定是否补全量

## 6. 记录模板

每跑完一个 baseline，都必须填一份：

- [attempt_0_status_template_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_status_template_2026-06-13.md)

不允许只留终端日志。

## 7. Partial Decision 规则

Week 1 只能输出下面四种之一：

1. `provisional go Track-On family`
2. `hold Track-On, investigate protocol/adapter gap`
3. `stop Track-On-R fine-tune, keep or drop Track-On2 separately`
4. `stop Track-On family, move to TAPNext++`

这不是 final decision。

## 8. Final Decision 规则

Attempt 0 final decision 必须在至少下面几行补齐后再做：

1. `CoTracker3 baseline`
2. `CoTracker3-Offline`，能跑则跑，不能跑则 reference-only
3. `Track-On2`
4. `Track-On-R`
5. `TAPNext++`
6. `AllTracker`，至少 DAVIS

允许的结论：

1. `Track-On-R` 最强
2. `Track-On family` 可替代 baseline，但 `Track-On-R` 不一定比 `Track-On2` 值得继续
3. `TAPNext++` 在 long-occ / re-entry 上更值得单开专项线
4. `CoTracker3-Offline` 仍是强锚点，其他候选未明显超过
5. 多个候选差距都很小，不急于单押，先保留为 teacher pool 候选

## 9. 明确不做的事

Attempt 0 期间不要做：

- 新训练
- 新蒸馏
- 新 pseudo-label 过滤实验
- 新 verifier 设计
- 当前仓库 CoTracker3 后处理变体
- 3D 新方向

## 10. 完成标准

只有同时满足下面几点，Attempt 0 才算真正完成：

1. 有统一协议主表
2. 有 repo-native reproduction 表
3. 有 adapter sanity 报告
4. 有 evaluator parity 报告
5. 有 long-occ 子表
6. 有明确 final decision

没有这些，不允许跳到“开始训练新主线”。
