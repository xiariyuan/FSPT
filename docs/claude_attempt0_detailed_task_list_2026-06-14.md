# Claude Attempt 0 Detailed Task List (2026-06-14)

## 0. 目标

这不是研究 brainstorming。

你的任务是基于当前仓库已经准备好的 Attempt 0 文档和脚手架，**执行一次可审计、可复算、可中断恢复的统一 baseline 复现**，并在完成后给出主 baseline 决策建议。

本轮只允许做：

1. 基线复现
2. 统一导出
3. 统一重算
4. 协议对齐
5. 结果记录
6. Go / hold / stop 决策

本轮不允许做：

1. CoTracker3 后处理新变体
2. verifier / reliability 新训练
3. 新 loss / 新 head
4. 任何“先训一点看看”的偏航实验

---

## 1. 必读文件

开始前必须完整阅读以下文件：

1. [docs/new_direction_ranking_2026-06-13.md](/gemini/code/FSPT/docs/new_direction_ranking_2026-06-13.md)
2. [docs/attempt_0_unified_reproduction_checklist_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_unified_reproduction_checklist_2026-06-13.md)
3. [docs/attempt_0_cotracker3_offline_anchor_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_cotracker3_offline_anchor_2026-06-13.md)
4. [docs/attempt_0_status_template_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_status_template_2026-06-13.md)
5. [docs/trackonr_execution_checklist_2026-06-13.md](/gemini/code/FSPT/docs/trackonr_execution_checklist_2026-06-13.md)
6. [docs/claude_attempt0_execution_brief_2026-06-13.md](/gemini/code/FSPT/docs/claude_attempt0_execution_brief_2026-06-13.md)
7. [utils/attempt0_schema.py](/gemini/code/FSPT/utils/attempt0_schema.py)
8. [scripts/init_attempt0_run.py](/gemini/code/FSPT/scripts/init_attempt0_run.py)
9. [scripts/attempt0_validate_cache.py](/gemini/code/FSPT/scripts/attempt0_validate_cache.py)
10. [scripts/attempt0_rescore_cache.py](/gemini/code/FSPT/scripts/attempt0_rescore_cache.py)
11. [scripts/attempt0_metric_parity.py](/gemini/code/FSPT/scripts/attempt0_metric_parity.py)
12. [datasets/metrics.py](/gemini/code/FSPT/datasets/metrics.py)
13. [datasets/tapvid_official_eval.py](/gemini/code/FSPT/datasets/tapvid_official_eval.py)

---

## 2. 你要交付的最终工件

Attempt 0 结束时，至少必须交付：

1. `repo-native reproduction table`
2. `unified rescoring table`
3. `protocol-bridge table`
4. `evaluator parity report`
5. `adapter sanity report`
6. `long-occ / re-entry / AJ_RD` 辅助表
7. 每个 baseline 一份 status JSON
8. `final_decision.md`

如果只完成 repo-native reproduction，但没有 unified rescoring，不算完成。

---

## 3. 固定候选集合

Attempt 0 只处理这 6 个对象：

1. `cotracker3_baseline`
2. `cotracker3_offline`
3. `trackon2`
4. `trackonr`
5. `tapnextpp`
6. `alltracker`

不要引入第 7 个 baseline。

---

## 4. 固定主评测协议

统一主表固定为：

1. dataset:
   - `tapvid_davis`
   - `tapvid_kinetics`
2. query mode:
   - `strided`
3. metric resolution mode:
   - `original`
4. metrics:
   - `AJ`
   - `OA`
   - `<avg`
   - `<4px`

辅助表强制保留：

1. `long-occ>20`: `AJ / <avg / <4px`
2. `re-entry first-frame error`
3. `AJ_RD`，如果该 baseline 支持

---

## 5. 现成脚手架与用途

当前仓库已经提供 5 个可直接使用的脚手架：

1. [scripts/init_attempt0_run.py](/gemini/code/FSPT/scripts/init_attempt0_run.py)
   - 初始化 Attempt 0 工作目录
   - 自动创建 manifest 和 baseline status skeleton
2. [scripts/attempt0_validate_cache.py](/gemini/code/FSPT/scripts/attempt0_validate_cache.py)
   - 对 unified cache 做 schema 校验
   - 检查 query anchor 对齐
   - 可要求 GT 字段
3. [scripts/attempt0_rescore_cache.py](/gemini/code/FSPT/scripts/attempt0_rescore_cache.py)
   - 对 unified cache 做统一重算
   - 输出 overall 和 long-occ 子表
4. [scripts/attempt0_metric_parity.py](/gemini/code/FSPT/scripts/attempt0_metric_parity.py)
   - 比较 `datasets/metrics.py` wrapper 与 official core
5. [utils/attempt0_schema.py](/gemini/code/FSPT/utils/attempt0_schema.py)
   - 定义 cache schema
   - 定义 status skeleton
   - 提供读写和校验函数

这些脚手架已做过本地 smoke test，至少以下链路已通过：

1. `py_compile`
2. `init_attempt0_run.py`
3. `attempt0_validate_cache.py`
4. `attempt0_rescore_cache.py`
5. `attempt0_metric_parity.py`

---

## 6. 执行顺序

按下面顺序执行，不要打乱：

### Step 1. 初始化 Attempt 0 工作区

运行：

```bash
python scripts/init_attempt0_run.py --name attempt0_<date> --out-root outputs
```

验收标准：

1. 生成 `outputs/attempt0_<date>/`
2. 存在：
   - `manifests/`
   - `prediction_caches/`
   - `repo_native/`
   - `unified_rescoring/`
   - `reports/`
   - `status/`
   - `logs/`
3. `status/` 下已经有 6 个 baseline 的 skeleton JSON

如果这一步失败，先修路径或 import，不要开始 baseline 复现。

### Step 2. 冻结 CoTracker3 baseline 行

先在当前仓库里冻结：

1. `cotracker3_baseline`
2. 若可得，再补 `cotracker3_offline`

要求：

1. 先记录 repo-native numbers
2. 再导出 unified cache
3. 再跑 unified rescoring

必须先把锚点行写进 Attempt 0 表，否则后续所有 delta 都没有参照。

### Step 3. 做 evaluator parity check

使用同一份带 GT 的 unified cache，运行：

```bash
python scripts/attempt0_metric_parity.py \
  --cache <cache.pt> \
  --out <report.json> \
  --query-mode strided
```

通过标准：

1. `max_abs_diff <= 1e-3`
2. 输出 `pass_threshold_1e-3 = true`

如果不通过：

1. 不允许继续比较 baseline
2. 先修 evaluator / adapter / 坐标语义

### Step 4. 做 cache validator 检查

所有 baseline 的 unified cache，在进入统一重算前都必须先过：

```bash
python scripts/attempt0_validate_cache.py \
  --cache <cache.pt> \
  --require-gt
```

强制检查点：

1. top-level schema 完整
2. 每条 record 字段完整
3. `query_points` shape 为 `(N, 3)`
4. `pred_tracks` shape 为 `(N, T, 2)`
5. `pred_visibility` shape 为 `(N, T)`
6. 坐标看起来是 normalized `[y, x]`
7. `query anchor` 对齐误差不离谱

如果 validation 失败，不要进入 unified rescoring。

### Step 5. Track-On family 子阶段

按 [docs/trackonr_execution_checklist_2026-06-13.md](/gemini/code/FSPT/docs/trackonr_execution_checklist_2026-06-13.md) 执行，但补充下面硬要求：

1. `Day 0` 先核对：
   - `track_on` README
   - `Track-On2 / Track-On-R` checkpoint release
   - 论文链接与 arXiv ID
   - `DINOv3` 权限
2. `DINOv3` 最多给 `24-48h` timebox
3. 超时立刻切 `TAPNext++`，不要空等
4. `Track-On-R` 只是 Attempt 0 子阶段，不是 final decision

Track-On 子阶段必须产出：

1. `track_on_env_manifest.md`
2. `track_on_checkpoint_manifest.md`
3. `trackon_adapter_sanity_report.json`
4. `trackon_metric_parity_report.json`
5. `track_on2_repo_native_metrics.json`
6. `track_onr_repo_native_metrics.json`
7. `track_on_unified_rescoring.json`
8. `week1_partial_decision.md`

### Step 6. TAPNext++

无论 Track-On family 成功与否，TAPNext++ 都必须进入 Attempt 0 统一表。

最少完成：

1. repo-native reproduction
2. unified cache 导出
3. unified rescoring
4. `long-occ>20` 子表
5. 若支持，额外报告 `AJ_RD`

若 `DINOv3` 阻塞，则 TAPNext++ 自动升级为当前优先级最高的 baseline。

### Step 7. AllTracker

AllTracker 不允许无限后置。

最低要求：

1. DAVIS repo-native reproduction
2. DAVIS unified cache
3. DAVIS unified rescoring

如果 DAVIS 下表现很强，再补 Kinetics。

### Step 8. Kinetics 规模控制

Kinetics 可采用两段式：

1. `100 clips quick check`
2. 再决定是否上全量

如果 quick check 连流程都跑不稳，不要直接砸全量算力。

---

## 7. Unified Cache 规范

每个 baseline 的导出 cache 必须符合以下字段语义：

### Top-level

1. `schema_version`
2. `model_name`
3. `repo_commit`
4. `checkpoint_path`
5. `checkpoint_sha256`
6. `dataset_name`
7. `split`
8. `protocol`
9. `records`

### Record-level

1. `video_id`
2. `sequence_index`
3. `frame_count`
4. `query_points`
5. `pred_tracks`
6. `pred_visibility`
7. `original_size`
8. `model_input_size`
9. `adapter_version`
10. `raw_coordinate_note`

若有 GT，再加：

1. `gt_tracks`
2. `gt_visibility`

统一语义固定为：

1. `query_points`: `[t, y, x]`, normalized
2. `pred_tracks`: `[y, x]`, normalized
3. `pred_visibility`: bool 或 `[0,1]`
4. `original_size`: `[H, W]`

---

## 8. Adapter 层硬要求

不要把导出逻辑零散写在各个外部 repo 里。

应在当前 FSPT 仓库内维护 adapter 层，并为每个 baseline 明确记录：

1. 原始坐标语义
2. 目标坐标语义
3. `(x, y) -> (y, x)` 是否转换
4. pixel -> normalized 是否转换
5. visibility 二值化阈值
6. query frame 是否保留
7. query frame 对齐策略
8. 序列长度对齐策略

每个 adapter 至少做 5 个 sanity checks：

1. query frame 上的预测位置与 query 坐标对齐
2. 坐标顺序无翻转
3. 像素 / 归一化转换正确
4. `(H, W)` / `(W, H)` 没写反
5. 抽 3-5 个样本可视化

---

## 9. 每个 baseline 跑完后必须更新 status

每跑完一个 baseline，必须更新：

1. [docs/attempt_0_status_template_2026-06-13.md](/gemini/code/FSPT/docs/attempt_0_status_template_2026-06-13.md)
2. `outputs/<run>/status/<model>.json`

至少填写：

1. `repo_commit`
2. `checkpoint_path`
3. `repo_native_metric_names`
4. `repo_native_numbers`
5. `official_reference_numbers`
6. `delta_vs_official`
7. `raw_coordinate_format`
8. `raw_visibility_format`
9. `query_format`
10. `adapter_version`
11. `adapter_sanity_status`
12. `AJ / OA / <avg / <4px`
13. `delta_vs_cotracker3_baseline`
14. `long_occ_AJ`
15. `AJ_RD`
16. `peak_memory_gb`
17. `eval_wall_time`
18. `keep_for_main_ranking`
19. `keep_as_teacher_candidate`
20. `next_action`

不允许只留终端日志。

---

## 10. 结果判定规则

### Week 1 partial decision 只允许 4 种

1. `provisional go Track-On family`
2. `hold Track-On, investigate protocol/adapter gap`
3. `stop Track-On-R fine-tune, keep or drop Track-On2 separately`
4. `stop Track-On family, move to TAPNext++`

### Attempt 0 final decision 至少考虑以下 5 类场景

1. `Track-On-R` unified rescoring 明显最强
   - 进入 Attempt 1 主 baseline
2. `Track-On2` 强于 `Track-On-R`
   - 保留 Track-On family，但不默认 real-world fine-tune 值得继续
3. `TAPNext++` overall 不一定最强，但 `long-occ>20` 显著最强
   - Track-On-R 作为 overall 主 baseline
   - TAPNext++ 作为 long-occ 对照和组合候选
4. 所有候选都不如 `CoTracker3-Offline`
   - 直接以 `CoTracker3-Offline` 为主 baseline
   - 后续参考 Track-On-R verifier 方案做 pseudo-label scaling
5. 所有候选差距都很小
   - 暂缓单一主线决策
   - 先做 protocol gap 排查或 teacher ensemble 价值评估

---

## 11. 执行期间的禁止事项

在 Attempt 0 完成前，不要做下面这些事：

1. 新训练
2. 额外 verifier head
3. 新的 recovery selector
4. calibration-only 分析
5. 论文故事写作发散
6. 3D 新方向实现

Attempt 0 的纪律就是：**先建立统一横向表，再谈下一阶段。**

---

## 12. 最低可执行命令清单

### 初始化

```bash
python scripts/init_attempt0_run.py --name attempt0_<date> --out-root outputs
```

### 校验 unified cache

```bash
python scripts/attempt0_validate_cache.py \
  --cache <cache.pt> \
  --require-gt
```

### 统一重算

```bash
python scripts/attempt0_rescore_cache.py \
  --cache <cache.pt> \
  --out <summary.json> \
  --query-mode strided \
  --metric-resolution-mode original
```

### parity audit

```bash
python scripts/attempt0_metric_parity.py \
  --cache <cache.pt> \
  --out <parity.json> \
  --query-mode strided
```

---

## 13. 完成标准

只有满足下面条件，Attempt 0 才算完成：

1. 6 个候选至少都进入过 repo-native reproduction 阶段，或明确记录阻塞原因
2. 至少 5 个候选进入 unified rescoring，若不足则必须有强阻塞说明
3. `CoTracker3 baseline` 已冻结
4. `CoTracker3-Offline` 已跑通，或至少以 reference-only 明确入表
5. parity report 通过
6. 每个参与重算的 baseline 都有 adapter sanity 记录
7. DAVIS 和 Kinetics 的 unified table 已生成
8. long-occ 辅助表已生成
9. final decision 已写出

如果以上任何一项未完成，就不要声称 Attempt 0 已闭环。
