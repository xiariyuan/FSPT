# Claude 执行计划：Canonical Re-Detection 主线（2026-06-18）

来源：Notion 子页面 `FSPT 新方向调研 · 论文研究点（2026-06-18）`

用途：这是一份本地可执行文档。执行者按此修改代码、脚本、配置和评测工件，我负责复核。

当前裁决：

> 不再继续 `online patch / local refiner / verifier gate / threshold sweep` 这类近场补丁路线。  
> 主线切换为 `audit -> teacher audit -> feature audit -> pseudo-label audit -> student training -> full eval -> online probe`。

---

## 0. 直接结论

### 0.1 已关闭的方向

以下方向不要再作为主线继续投入：

- `DINOv2` whole-frame retrieval 新变体
- `local refiner / offset regression` 主线
- `local grid verifier / ranking` 主线
- `memory hygiene / oracle gate` 主线
- `online verifier gate` 主线
- `threshold sweep` 主线
- `first+input` 作为 2026-06-18 之后的新主线证据

这些结果保留为：

- 失败记录
- 负结果分析
- 论文中的 diagnostic / benchmark 证据

### 0.2 当前主线

当前唯一可执行主线是：

1. 先把 `strided+original` 的坐标、cache、指标全部锁死
2. 再做 teacher ceiling / oracle selection
3. 再做零样本特征分水岭审计
4. 再做 re-entry pseudo-label 数据审计
5. 再训练 student
6. 再做 full DAVIS + 跨数据验证
7. 最后才允许 online probe

### 0.3 评测优先级

结论优先级固定为：

1. `re-entry`
2. `long-occ`
3. `tail after retracking`
4. 最后才看 overall `AJ / OA / delta_avg`

`first+input` 只允许作为历史兼容参考，不允许再支撑主线判断。

---

## 1. 统一交付约定

### 1.1 统一输出目录

所有本轮产物统一落在：

```text
outputs/canonical_redetection_2026-06-18/
```

建议子目录：

```text
outputs/canonical_redetection_2026-06-18/
  p0_metric_audit/
  p1_teacher_audit/
  p2_feature_audit/
  p3_pseudo_label_audit/
  p4_student_training/
  p5_full_eval/
  p6_online_probe/
```

### 1.2 每个阶段必须成对输出

每个阶段都必须产出：

- `*.json`：机器可读指标与 manifest
- `*.md`：人类可读 decision summary

`manifest.json` 至少包含：

- `git_commit`
- `command_line`
- `dataset_path`
- `protocol`
- `cache_schema_version`
- `coord_format`
- `teacher_name`
- `feature_source`
- `checkpoint_sha256`
- `environment`

### 1.3 不允许的写法

不要把 Notion 里的伪命令直接当成可运行入口。

真实执行入口必须优先使用仓库现成脚本：

- `train.py`
- `scripts/train_distributed.py`
- 现有 `scripts/` 里的 audit / eval / export 工具

如果 Notion 中提到的脚本当前不存在，就明确标记为“待新建”。

---

## 2. 仓库现实入口

### 2.1 当前可复用脚本

优先复用这些已存在入口：

- `train.py`
- `scripts/train_distributed.py`
- `scripts/attempt0_export_strided_original_cache.py`
- `scripts/attempt0_validate_cache.py`
- `scripts/attempt0_rescore_cache.py`
- `scripts/eval_recovery_position_error.py`
- `scripts/eval_long_occlusion_oracle_gap.py`
- `scripts/eval_davis_anchor_topk_recall.py`
- `scripts/eval_attempt0_reentry_head2head.py`
- `scripts/trackon2_memory_hygiene_diagnostic.py`
- `scripts/build_ctoffline_local_refiner_dataset.py`
- `scripts/build_ctoffline_grid_verifier_dataset.py`
- `scripts/build_online_recovery_anchor_dataset.py`
- `scripts/build_online_recovery_anchor_dataset_v2.py`
- `scripts/build_online_recovery_anchor_dataset_v3.py`
- `scripts/eval_global_retrieval.py`
- `scripts/eval_multi_support_dino.py`
- `scripts/eval_dense_readout.py`
- `scripts/eval_prt_hybrid_selector.py`

### 2.2 待新建脚本

Notion 里提到但仓库当前没有的脚本，执行时应新建：

- `fspt/coords.py`
- `scripts/eval_aj_rd_from_cache.py`
- `scripts/audit_cache_schema.py`
- `scripts/audit_visibility_reentry_events.py`
- `scripts/eval_teacher_reentry_audit.py`
- `scripts/oracle_teacher_selection.py`
- `scripts/eval_redetection_feature_audit.py`
- `scripts/extract_feature_cache.py`
- `scripts/audit_vdit_resource_smoke.py`
- `scripts/audit_fb_consistency.py`
- `scripts/build_reentry_pseudo_labels.py`
- `scripts/audit_pseudo_label_quality.py`
- `scripts/filter_pseudo_labels_fb_flow.py`
- `scripts/split_pseudo_label_train_val.py`

说明：

- `fspt/coords.py` 当前仓库里不存在；如果执行者不想引入新顶层包，也可以等价放到 `utils/coords.py`，但必须只有一个 canonical coord helper。
- 以上脚本如果已有等价实现，可以重命名或封装复用，但不要在多个地方各写一套口径。

---

## 3. 总体执行 DAG

| Phase | 目标 | 通过后进入 | 失败后动作 |
|---|---|---|---|
| P0 | 指标 / 坐标 / cache 审计 | P1 | 只修 P0 |
| P1 | Teacher ceiling / oracle selection | P2 | 只修 teacher audit，不训练 |
| P2 | 零样本特征分水岭审计 | P3A 或 P3B | 只修 feature audit，不训练 student |
| P3 | Re-entry pseudo-label 数据审计 | P4 | 只修数据与过滤，不训练 student |
| P4 | Student 训练 | P5 | 只修训练配方，不进 online probe |
| P5 | Full DAVIS + 跨数据验证 | P6 或 stop | 只保留诊断结果 |
| P6 | Online probe | 收尾 | 若失败，停止方法线 |

---

## 4. P0 — 指标、坐标与 cache 审计

### 4.1 目标

建立所有后续实验的唯一评测口径，防止再次出现：

- coordinate 混乱
- visibility 混乱
- query protocol 混乱
- `AJ_RD` 定义漂移

### 4.2 输入

- `TAP-Vid DAVIS` 原始数据
- `strided+original` cache
- 当前 `CoTracker3 offline` cache
- 当前 `Track-On2` cache

### 4.3 必须实现

执行者需要把以下能力落成统一入口：

- `to_xy_pixel(...)`
- `to_yx_norm(...)`
- `to_xy_256(...)`
- `assert_coord_range(...)`
- `roundtrip_coord_test(...)`

其中，所有坐标转换必须走单一 helper。

### 4.4 必须产出的脚本

- `fspt/coords.py`
- `scripts/eval_aj_rd_from_cache.py`
- `scripts/audit_cache_schema.py`
- `scripts/audit_visibility_reentry_events.py`

### 4.5 具体任务

1. 统一 `query_points / pred_tracks / gt_tracks / visibility` 的坐标语义
2. 对 `strided+original` 做 roundtrip 测试
3. 对统一 cache 做 schema 校验
4. 复算 `AJ_RD`
5. 复算 `re-entry first-frame error`
6. 复算 `long-occ` 分桶指标
7. 确认 `n_reappearance_events` 与独立 re-entry 统计脚本一致

### 4.6 Go / Stop

#### Go

满足以下条件才允许进入 P1：

- 三个 cache 都能稳定复算
- 同一 cache 重跑 2 次，核心指标差异 `<= 1e-6`
- 坐标 roundtrip 通过
- `n_reappearance_events` 与独立统计一致
- `AJ_RD` 和 re-entry 口径完全对齐

#### Stop

出现任一情况就只修 P0：

- 任一模型指标无法复现
- 坐标转换存在歧义
- `AJ_RD` 与 first re-entry 使用了不同 visibility 定义
- cache schema 不一致

### 4.7 P0 输出

- `outputs/canonical_redetection_2026-06-18/p0_metric_audit/manifest.json`
- `outputs/canonical_redetection_2026-06-18/p0_metric_audit/metrics_summary.json`
- `outputs/canonical_redetection_2026-06-18/p0_metric_audit/per_video_metrics.json`
- `outputs/canonical_redetection_2026-06-18/p0_metric_audit/summary.md`
- `outputs/canonical_redetection_2026-06-18/p0_metric_audit/decision.md`

---

## 5. P1 — Teacher ceiling / oracle selection 审计

### 5.1 目标

确认 teacher distillation 是否还有 headroom。

这里不训练，不改 tracker，只比较已有或可接入 teacher 的 re-entry 表现。

### 5.2 关键定义

```text
oracle_teacher(event) = argmin_teacher reentry_first_frame_error(event)
```

```text
oracle_gain = oracle_teacher - fixed_best_teacher
```

### 5.3 候选 teacher

优先比较这些 teacher：

- `cotracker3_offline`
- `trackon2`
- `alltracker`（若可得）
- `ReTracker` / `TAPTRv3`（若可得）
- `HeFT` / `Video-DiT`（若可得）

### 5.4 必报指标

每个 teacher 都必须报告：

- `overall AJ / OA / delta_avg`
- `AJ_RD`
- `re-entry first-frame median / mean / p95`
- `re-entry <4px / <8px / <16px`
- `long-occ <4px / <8px / <16px`
- `teacher usage distribution`
- `hard-tail examples`

### 5.5 必须产出的脚本

- `scripts/eval_teacher_reentry_audit.py`
- `scripts/oracle_teacher_selection.py`

### 5.6 Go / Stop

#### Go

满足任意一条可以进入 P2：

- `oracle teacher selection` 在 `AJ_RD` 或 `long-occ <4px` 上比固定最佳 teacher 至少 `+5pp`
- 某个非 `cotracker3_offline` teacher 在 hard long-occ 上显著优于 `cotracker3_offline`

#### Weak Go

如果只有 `+2pp ~ +5pp` 的小幅收益，可以继续做 P2，但不要提前承诺 multi-teacher distillation 会成功。

#### Stop

满足任一条时，停止 teacher distillation 主线：

- `oracle teacher selection gain < 2pp`
- 所有 teacher 在 hard tail 上同错
- teacher 分布没有明显分化

### 5.7 P1 输出

- `outputs/canonical_redetection_2026-06-18/p1_teacher_audit/manifest.json`
- `outputs/canonical_redetection_2026-06-18/p1_teacher_audit/metrics_summary.json`
- `outputs/canonical_redetection_2026-06-18/p1_teacher_audit/per_event_metrics.json`
- `outputs/canonical_redetection_2026-06-18/p1_teacher_audit/summary.md`
- `outputs/canonical_redetection_2026-06-18/p1_teacher_audit/decision.md`

---

## 6. P2 — 零样本特征分水岭审计

### 6.1 目标

验证是否存在一个时序一致特征源，能够把 GT re-entry 重新拉回 top-k。

这一步不训练 student，只做特征审计。

### 6.2 两级执行

#### P2a — 资源 smoke

先用极小规模确认 feature extraction 不是工程死路。

要求：

- 只跑 `2 videos / 64 re-entry events`
- 记录 `feature extraction time / video`
- 记录 `GPU peak memory`
- 记录 `feature grid size`
- 记录 `crash / OOM`
- 记录 `是否能缓存到磁盘`

#### P2b — top-k recall audit

再跑一个更完整的 recall audit：

- `5 videos / 128 queries` smoke
- 评估 `top1 / top5 / top10` 在 `4px / 8px / 16px` 下的 recall

### 6.3 特征源

优先比较：

- `DINOv2 baseline`
- `Chrono` 风格特征
- `VDiT / HeFT` 风格特征
- `DINO-Tracker` style 特征
- `CoTracker3 offline` coarse prior（如果有）

### 6.4 必报指标

每个特征源都必须报告：

- `n`
- `top1 median px`
- `top5 best median px`
- `top1@4/8/16px`
- `top5@4/8/16px`
- `top10@4/8/16px`
- `top1 miss but top5 hit @16px`
- `long-occ` 子集的同组指标
- `FB consistency pass rate`
- `accepted top5@16 after FB`

### 6.5 必须产出的脚本

- `scripts/eval_redetection_feature_audit.py`
- `scripts/extract_feature_cache.py`
- `scripts/audit_vdit_resource_smoke.py`
- `scripts/audit_fb_consistency.py`

### 6.6 Go / Stop

#### Strong Go

满足以下任意两条，允许进入 `P3A`：

- `overall top5@16px >= 20%`
- `long-occ top5@16px >= 10%`
- `top1 miss but top5 hit @16px >= 10%`

#### Branch Go

如果 feature audit 只是中等成功，但仍有明显信号，则可以进入 `P3A`：

- feature-based pseudo labels

#### Stop

满足任一条时，停止 feature-based route：

- `overall top5@16px < 10%`
- `long-occ top5@16px < 5%`
- `FB consistency` 之后候选几乎全被过滤
- 资源开销不可接受

### 6.7 P2 输出

- `outputs/canonical_redetection_2026-06-18/p2_feature_audit/manifest.json`
- `outputs/canonical_redetection_2026-06-18/p2_feature_audit/metrics_summary.json`
- `outputs/canonical_redetection_2026-06-18/p2_feature_audit/per_query_metrics.json`
- `outputs/canonical_redetection_2026-06-18/p2_feature_audit/summary.md`
- `outputs/canonical_redetection_2026-06-18/p2_feature_audit/decision.md`

---

## 7. P3 — Re-entry pseudo-label 数据审计

### 7.1 目标

把问题从“候选是否够强”切到“训练数据与监督分布是否够强”。

P3 不训练 student，只生成并审计伪标签。

### 7.2 两个分支

#### P3A — feature-based pseudo labels

**状态：oracle-only / stop.**

当前 DAVIS 上的 feature-based pseudo-label 路线已被复核为方法学泄漏，且实测样本量与质量均不足以支撑训练链。该分支只保留为诊断审计，不再作为可训练主线。

如果后续要恢复同类路线，必须先满足以下前提：

- 输入切换到外部无标注数据源，不再使用 DAVIS GT 造标签
- `FB consistency` / `flow consistency` 真正接入
- `train/val split leakage audit` 通过
- 样本规模达到 `n_reentry_events >= 10k`

在满足上述前提之前，不再推进：

- feature-based candidate generation
- `cost-level` 筛选
- `FB consistency` 筛选
- `teacher selection`

#### P3B — data-first pseudo labels

如果 P2 没过门槛，就走：

- `AnthroTAP-style` 结构化监督
- `CoTracker3 offline` / `CT-offline-centered` 伪标签
- fallback `Point Prompting`

### 7.3 数据源分层

伪标签来源建议按如下分层：

1. `ct_offline`
2. `anthrotap`
3. `vdit`
4. `alltracker`
5. `point_prompting`

### 7.4 伪标签 schema

每条 pseudo-label 必须至少包含：

```json
{
  "video_id": "...",
  "track_id": 0,
  "frame": 123,
  "xy": [0.0, 0.0],
  "coord_format": "xy_pixel",
  "visible": true,
  "teacher": "cotracker3_offline",
  "teacher_conf": 0.0,
  "fb_error": 0.0,
  "flow_consistency_error": 0.0,
  "occ_run_len": 32,
  "is_reentry_frame": true,
  "source_bucket": "ct_offline|anthrotap|vdit|alltracker|point_prompting",
  "quality_flags": ["fb_pass", "flow_pass"]
}
```

### 7.5 必报数据质量指标

- `n_tracks`
- `n_frames`
- `n_reentry_events`
- `n_long_occ_events`
- `occ length histogram`
- `teacher distribution`
- `FB pass rate`
- `flow consistency pass rate`
- `pseudo-label <4px / <8px / <16px on DAVIS GT subset`
- `per-video distribution`
- `train/val split leakage audit`

### 7.6 质量门槛

满足以下条件才允许进入 P4：

- `n_reentry_events >= 10k`
- `occ_run_len >= 16` 的比例 `>= 20%`
- DAVIS GT subset 上 `reentry <16px >= 80%`
- 单一视频贡献 `< 20%`
- `FB / flow` 通过的样本数量足够，不依赖大量未过滤标签

### 7.7 Stop

以下任一条成立时，P3 停止：

- re-entry 事件不足
- `pseudo-label <16px < 60%`
- 数据几乎全是短遮挡
- 单一 domain / human 视频过度支配
- `FB/flow` 过滤后样本过少

### 7.8 必须产出的脚本

- `scripts/build_reentry_pseudo_labels.py`
- `scripts/audit_pseudo_label_quality.py`
- `scripts/filter_pseudo_labels_fb_flow.py`
- `scripts/split_pseudo_label_train_val.py`

### 7.9 P3 输出

- `outputs/canonical_redetection_2026-06-18/p3_pseudo_label_audit/manifest.json`
- `outputs/canonical_redetection_2026-06-18/p3_pseudo_label_audit/labels_train.jsonl`
- `outputs/canonical_redetection_2026-06-18/p3_pseudo_label_audit/metrics_summary.json`
- `outputs/canonical_redetection_2026-06-18/p3_pseudo_label_audit/summary.md`
- `outputs/canonical_redetection_2026-06-18/p3_pseudo_label_audit/decision.md`

---

## 8. P4 — Student 训练

### 8.1 目标

训练一个 online student，使它在训练中真正见过 `long-occ / re-entry`，而不是推理时靠补丁修。

### 8.2 候选 student

优先沿用现有 `Track-On2 / CoTracker-style` 训练链，不要重新发明主干。

### 8.3 建议配置

建议新建：

- `configs/fspt_redetection_student_smoke.yaml`
- `configs/fspt_redetection_student_full.yaml`

### 8.4 训练分层

建议从小到大：

1. `P4a synthetic smoke`
2. `P4b pseudo-label smoke`
3. `P4c mixed training`
4. `P4d hard-tail oversampling`

### 8.5 训练数据组合

- `periodic roll`
- `synthetic occluder`
- `long crop jump`
- `random re-entry displacement`
- `variable aspect ratio crop`

### 8.6 Loss 设计

基础 loss：

- `visible coordinate loss`
- `visibility / confidence loss`
- `standard TAP loss`

新增 re-entry loss：

- `reentry_frame_weight = 2.0`
- `post_reentry_window_weight = 1.5`
- `occluded_inframe_coord_weight = 0.2`
- 仅对可信 hidden coordinate 数据启用 occluded coordinate supervision

Teacher / consistency：

- `teacher_distill_loss`
- `fb_consistency_loss`

### 8.7 训练 smoke

先跑小训练，不要一上来 full run。

建议单卡 smoke 命令骨架：

```bash
python train.py \
  --config configs/fspt_redetection_student_smoke.yaml \
  --output outputs/canonical_redetection_2026-06-18/p4_student_training/smoke_v1
```

如果要多卡，再切到 `scripts/train_distributed.py`。

### 8.8 Smoke 指标

必须同时看：

- loss 是否收敛
- train / val re-entry error 是否同时下降
- visibility calibration 是否恶化
- overall validation 是否掉太多

### 8.9 Go / Stop

#### Smoke Go

满足以下条件才允许进入 full train：

- smoke loss 收敛
- train / val re-entry error 同时下降
- visibility calibration 没有崩坏
- overall validation 下降不超过 `1pp`

#### Full Train Go

满足以下条件才允许进入 P5：

- `full DAVIS AJ_RD +2pp`
- `long-occ <4px +5pp`
- `re-entry median error` 下降 `>=10%`
- `overall AJ` 下降 `<0.5pp`
- `per-video long-occ win rate >= 60%`

#### Stop

满足任一条时，停止 student 主线：

- 只提升 train，不提升 val
- `overall AJ` 下降 `>1pp`
- `re-entry` 提升来自单一视频
- `visibility head` 崩坏

### 8.10 P4 输出

- `outputs/canonical_redetection_2026-06-18/p4_student_training/smoke_v1/`
- `outputs/canonical_redetection_2026-06-18/p4_student_training/full_v1/`
- 对应的 `manifest.json` / `metrics_summary.json` / `summary.md`

---

## 9. P5 — Full DAVIS + 跨数据验证

### 9.1 目标

确认 P4 的增益不是 smoke 假象。

### 9.2 必跑评测

主评测：

- `TAP-Vid DAVIS strided+original`

兼容参考：

- `TAP-Vid DAVIS first+input`，只作历史参考，不作主结论

有条件再跑：

- `Kinetics`
- `RoboTAP`
- `PointOdyssey`

### 9.3 主表指标

每个模型必须报告：

- `AJ_RD`
- `long-occ <4px`
- `re-entry median error`
- `re-entry mean error`
- `re-entry p95`
- `overall AJ / OA / delta_avg`
- `per-video long-occ win rate`

### 9.4 成功判定

#### Paper-track Go

满足以下条件才算 paper-track 成功：

- `AJ_RD +2pp`
- `long-occ <4px +5pp`
- `re-entry median error` 下降 `>=10%`
- `overall AJ` 下降 `<0.5pp`
- `per-video long-occ win rate >= 60%`

#### Diagnostic-only

满足以下情况之一时，仍可保留为诊断贡献，但不能进入主方法：

- `re-entry` 有提升但 `overall AJ` 下降 `0.5–1.0pp`
- 增益集中在单一 domain
- 只有 train 好看，val 不动

#### Stop

满足任一条时，停止当前方法线：

- `AJ_RD < +1pp`
- `long-occ <4px < +2pp`
- `overall AJ` 下降 `>1pp`
- `re-entry` 没有实质改善

### 9.5 P5 输出

- `outputs/canonical_redetection_2026-06-18/p5_full_eval/manifest.json`
- `outputs/canonical_redetection_2026-06-18/p5_full_eval/metrics_summary.json`
- `outputs/canonical_redetection_2026-06-18/p5_full_eval/per_video_metrics.json`
- `outputs/canonical_redetection_2026-06-18/p5_full_eval/summary.md`
- `outputs/canonical_redetection_2026-06-18/p5_full_eval/decision.md`

---

## 10. P6 — Online probe

### 10.1 前提

只有 P5 过 `Paper-track Go`，才允许进入 P6。

### 10.2 目标

验证训练后的 student 在 causal / online 设定下，是否能复现离线指标，不使用 `CoTracker3 offline` runtime。

### 10.3 明确禁止

P6 里不要做这些事：

- 不能在线调用 `CoTracker3 offline`
- 不能使用未来帧
- 不能用 GT trigger
- 不能把 verifier 直接接回轨迹写入

### 10.4 允许的在线信号

允许的 online 信号只限于：

- student 自身 visibility / uncertainty 触发 lost state
- student 自身 re-entry head 输出
- 基于当前与历史可用信息的 FB consistency
- teacher 只在训练期使用

### 10.5 必报指标

- `online AJ / OA / delta_avg`
- `online AJ_RD`
- `trigger precision / recall`
- `false re-entry rate`
- `no-op rate`
- `latency`
- `memory`

### 10.6 Go / Stop

#### Go

满足以下条件才算在线 probe 成功：

- `online AJ_RD` 保留 P5 离线提升的 `>=70%`
- `overall AJ` no-harm
- `trigger false positive` 不导致明显 drift

#### Stop

以下任一条成立时，停止在线路线：

- 离线有效，但在线不触发
- 在线误触发严重
- online 写回导致 overall collapse

### 10.7 P6 输出

- `outputs/canonical_redetection_2026-06-18/p6_online_probe/manifest.json`
- `outputs/canonical_redetection_2026-06-18/p6_online_probe/metrics_summary.json`
- `outputs/canonical_redetection_2026-06-18/p6_online_probe/summary.md`
- `outputs/canonical_redetection_2026-06-18/p6_online_probe/decision.md`

---

## 11. 论文产出路线

### 11.1 若 P5 / P6 成功

论文主线写成：

> Re-entry-aware point tracking via structured pseudo-label supervision and re-detection-first training.

可提的贡献：

1. `AJ_RD / re-entry` 审计协议，面向 `DAVIS strided+original`
2. 证明 raw `DINO` / `local refiner` / `verifier gate` 不足
3. 提出 `teacher-distilled re-entry training`
4. 用 `periodic roll + trusted occluded coordinate supervision + FB-filtered pseudo labels` 形成训练配方
5. 训练出的 online student 在 `long-occ / AJ_RD` 上有提升且 `overall no-harm`

### 11.2 若训练不成功但审计强

论文降级为 diagnostic / benchmark：

- 系统性证明 `long-occ re-entry failure` 来自候选 / 监督分布
- 提供 `AJ_RD` 与 re-entry cache
- 提供 teacher ceiling / feature audit
- 给出 negative results map，避免重复 `raw DINO` / `local refiner` 失败路线

### 11.3 若 P2 / P3 都失败

停止方法线，只保留失败分析：

> Current public teachers and foundation features do not provide reliable re-entry supervision under DAVIS strided+original; stronger data or a new benchmark is required.

---

## 12. 失败记录（必须保留）

以下失败记录要保留在仓库文档中，不能删除：

- `DINOv2` whole-frame retrieval 失败
- `learned offset regression refiner` 失败
- `learned local grid verifier` 失败
- `Track-On2 memory hygiene oracle` 不足以支撑继续投入

这些失败都不是废结果，而是下一步切换主线的依据。

---

## 13. 给执行者的最终修改清单

执行者需要做的事，按优先级排序如下：

1. 新建本文件对应的执行版本，所有 Notion 的 canonical 结论必须落成可运行 checklist
2. 新建 P0-P6 需要的脚本骨架，并把真实入口接到仓库现有数据流
3. 新建 canonical coord helper，统一坐标语义
4. 新建 P0 / P1 / P2 / P3 / P4 / P5 / P6 的 `.json + .md` 输出约定
5. 新建对应 `configs/` 中的 smoke / full 配置
6. 每个阶段都先跑 smoke，再决定是否进入下一阶段
7. 每个 stop 条件都必须写成实际 decision，不许口头跳过

---

## 14. 复核时我会检查什么

我复核时只看这些点：

- 是否还在偷偷用 `first+input` 支撑主线
- 是否还在混用旧的补丁路线
- 是否每个阶段都产出 `.json + .md`
- 是否每个 Go / Stop 都真的按门槛执行
- 是否 `P2 -> P3 -> P4` 的分支逻辑清楚
- 是否 `P5` 之前没有在线泄漏
- 是否所有失败记录都保留了

如果这些点对了，再看具体数值。
