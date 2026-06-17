# Attempt 0 / Track-On-R Checklist Review (2026-06-13)

## 0. 结论先行

三份新增文档的主判断总体成立：

1. 不应继续在当前 `CoTracker3 + 后处理 recovery/rerank` 空间里找主指标增益。
2. 必须先做统一复现与统一重算，不能直接相信各 repo README / paper 的原生表格。
3. `Track-On-R / Track-On2 -> TAPNext++ -> AllTracker` 作为当前最快推进顺序是合理的。

但现有清单需要补强几个协议风险：

- `Track-On-R Week 1` 应定义为 **Attempt 0 的 Track-On 子阶段**，不是完整 Attempt 0 的替代品。
- 统一重算不能只说“导出 pred_tracks / pred_visibility”，必须写死坐标、可见性、query protocol、dataset item 对齐和 evaluator sanity check。
- “用本仓库 evaluator 重算”之前，应验证它与官方 TAP-Vid evaluator 在同一输入上是否一致，否则统一表仍可能有系统偏差。
- Track-On-R vs Track-On2 的增益可能很小，kill criterion 需要区分“Track-On-R fine-tune 是否值得继续”和“Track-On family 是否值得成为新 baseline”。

## 1. 合理且应保留的部分

### 1.1 方向排序合理

`docs/new_direction_ranking_2026-06-13.md` 的排序合理：

1. `Track-On-R / Track-On2`
2. `TAPNext++`
3. `AllTracker`

原因成立：Track-On repo 提供 Track-On2、Track-On-R、verifier、teacher ensemble、evaluation wrappers；TAPNext++ 对 long-sequence / re-detection 机制最强但训练链更重；AllTracker 是 dense high-res 正交方向，更适合作为 baseline / teacher / 第二阶段资产。

### 1.2 Attempt 0 强制统一重算是必要的

`docs/attempt_0_unified_reproduction_checklist_2026-06-13.md` 要求产出 repo-native reproduction table、unified rescoring table、manifest、prediction caches 和 go/no-go decision，这个原则必须保留。不同 repo 的 `query-first / strided / delta_avg / AJ / resolution` 很容易混用；没有 unified rescoring table，后续方向选择不可信。

### 1.3 Week 1 先跑 Track-On-R 是可执行的

`docs/trackonr_execution_checklist_2026-06-13.md` 的 Day-by-Day 计划整体可执行，尤其是 Day 1 解决 DINOv3 和 mmcv，Day 2/3 跑 repo-native DAVIS/Kinetics，Day 4/5/6 做导出和统一重算，Week 1 不做训练、不改 verifier、不扩 teacher pool。

## 2. 需要修正或补充的问题

### 2.1 Attempt 0 与 Track-On-R Week 1 的关系需要写清楚

当前存在轻微冲突：Attempt 0 文档要求统一跑完 5 个对象；Track-On-R 清单则像是在 Week 1 后就决定是否进入 Attempt 1。

建议解释为：

> Week 1 Track-On-R checklist 是 Attempt 0 的 Phase A/B，不是完整 Attempt 0 的替代品。它可以先做，因为 Track-On 优先级最高；但除非 Track-On-R 在 unified rescoring 中已经远强于其它候选，否则最终“押主方向”的决策仍应等 TAPNext++ 和 AllTracker 的统一表补齐。

执行建议：

- Week 1 先完成 `CoTracker3 + Track-On2 + Track-On-R`。
- Week 1 结束只做 `provisional go / hold / stop`。
- 正式方向锁定应叫 `Attempt 0 final decision`，需要 TAPNext++ / AllTracker 表补齐。
- 如果 Track-On-R 明显领先 CoTracker3，可并行启动 Attempt 1 准备，但后台仍补 TAPNext++ / AllTracker。

### 2.2 必须增加 evaluator parity check

本仓库 `datasets_code/metrics.py` 约定：

- `pred_tracks`: `(N, T, 2)`，归一化坐标 `[0,1]`，顺序 `[y, x]`
- `gt_tracks`: `(N, T, 2)`，归一化坐标 `[0,1]`，顺序 `[y, x]`
- `query_points`: `(N, 3)`，顺序 `[t, y, x]`
- `pred_visibility`: bool 或 score，内部用 `>0.5`
- `resolution`: `(H, W)`，按 original size 转像素阈值

Track-On README 中 Predictor 输出是：

- queries: `(t, x, y)` in pixel coordinates
- traj: `(x, y)` in pixels
- vis: `{0,1}`

所以 unified rescoring 前必须检查：

1. Track-On `(x,y)` pixel -> FSPT `(y,x)` normalized。
2. query `(t,x,y)` -> `(t,y,x)`。
3. original size 必须是 `(H,W)`，不是 `(W,H)`。
4. pred_visibility 要确认 True=visible；若是 prob/logit，要确认阈值方向。

新增强制工件：

- `adapter_sanity_report.json`
- `metric_parity_report.json`

### 2.3 需要与官方 TAP-Vid evaluator 做 parity 验证

仅用本仓库 evaluator 作为统一裁判仍有风险：统一但未必 official。建议 Attempt 0 增加 `Evaluator Parity`：

- 选择 2-3 个 DAVIS videos。
- 同一份 GT + 同一份 prediction cache。
- 同时跑：
  - `datasets_code/metrics.py`
  - `google-deepmind/tapnet` 官方 TAP-Vid metrics
- 对比 `AJ / OA / <avg / <4px`。

通过标准：每个指标差异应小于 `1e-4` 到 `1e-3`。如果更大，先修 evaluator / adapter，不进入模型比较。

### 2.4 统一导出格式缺少关键字段

当前最小字段不够。建议统一导出 schema 至少包含：

```text
model_name: str
repo_commit: str
checkpoint_path: str
checkpoint_sha256: str
dataset_name: davis | kinetics
split: str
protocol: str
sequence_name: str
sequence_index: int
frame_count: int
original_size_hw: [H, W]
model_input_size_hw: [H_in, W_in]
query_points_tyx_norm: float32 [N,3]
pred_tracks_yx_norm: float32 [N,T,2]
pred_visibility_bool: bool [N,T]
raw_coordinate_note: str
adapter_version: str
```

尤其必须保留 query frame、sequence index、coordinate format、visibility threshold、resize policy、checkpoint hash 和 repo commit。

### 2.5 Track-On-R kill criterion 需要拆成两层

当前清单容易误杀 Track-On family。建议拆成：

#### A. Track-On family baseline replacement

继续条件：

- `max(Track-On2, Track-On-R)` 在 DAVIS/Kinetics unified 主指标上明显优于当前 CoTracker3 baseline。

如果成立，即使 Track-On-R 没赢 Track-On2，Track-On2 也仍可成为新 baseline。

#### B. Track-On-R real-world fine-tune 主线

继续条件：

- Track-On-R 在至少一个核心数据集上稳定优于 Track-On2；或
- Track-On-R 在 Kinetics / real-world-heavy benchmark 上更强，且 DAVIS 不显著下降。

如果不成立，应停止的是 `Track-On-R fine-tuning 主线`，不是整个 Track-On2 baseline replacement。

### 2.6 不能只看 DAVIS AJ/OA，也要同步看 `<avg` / `<4px`

Attempt 0 中 Track-On2/R 继续条件目前偏重 DAVIS AJ 和 OA。建议改为主指标组合：

- DAVIS: `AJ` 必须不低，`<avg` 和 `<4px` 至少一个明确更好，`OA` 不明显下降。
- Kinetics: 同样记录，不能出现 DAVIS 赢但 Kinetics 明显输。

建议门槛：

- `AJ >= CoTracker3 + 0.3` percentage point，或至少不低于 CoTracker3；
- `<avg >= CoTracker3 + 0.3` 或 `<4px >= CoTracker3 + 0.3`；
- `OA >= CoTracker3 - 0.3`；
- DAVIS 和 Kinetics 不出现方向相反的大幅退化。

### 2.7 Kinetics 数据准备风险需要前置

Week 1 Day 3 要跑 Kinetics，但 Kinetics TAP-Vid 下载/处理更容易卡。建议 Day 1 同时做：

- DAVIS path smoke；
- Kinetics path smoke；
- 随机读取 1 个 sample，打印 video shape、query shape、target shape、occlusion shape、original size。

如果 Kinetics 未准备好，DAVIS Track-On eval 可以继续，但 Week 1 decision 必须标注 `DAVIS-only provisional`，不能作最终 go decision。

### 2.8 Track-On repo branch / commit 固定要更严格

Track-On repo 主分支同时包含 Track-On-R / Track-On2，且 README 提到 earlier branches。Week 1 必须记录：

- branch；
- commit hash；
- checkpoint URL；
- checkpoint sha256；
- DINOv3 model revision；
- HuggingFace cache path。

否则后面结果不可复现。

### 2.9 DINOv3 权限失败不应中止整个方向

当前 kill criterion 写 DINOv3 权限拿不到就停止 Track-On-R 第一优先级。建议保留但加 fallback：

1. 先尝试拿 DINOv3 权限。
2. 若 24-48h 内拿不到，检查 DINOv2 branch / previous Track-On2 branch，或并行启动 TAPNext++ checkpoint 复现。
3. DINOv3 权限卡住只 kill `Track-On-R Week 1 execution`，不代表 Track-On 方法无效。

### 2.10 TAPNext++ / AllTracker 不应无限后置

如果 Week 1 只跑 Track-On，可能形成路径依赖。建议明确：

- TAPNext++ repo-native + unified DAVIS 至少应在 Week 2 前半完成。
- AllTracker repo-native DAVIS 至少应在 Week 2 完成。
- 如果 Track-On-R 只比 CoTracker3 小幅赢，不应过早进入 Track-On-R 训练。

## 3. 建议新增到 Attempt 0 的强制检查

### 3.1 Adapter unit tests

为每个外部模型导出适配器做最小单元测试：

1. 构造完美预测：`pred_tracks = gt_tracks`，`pred_visibility = gt_visibility`。
2. 通过 export -> import -> metric pipeline。
3. 确认 `AJ ~= 1.0`、`OA ~= 1.0`、`<avg ~= 1.0`、`<4px ~= 1.0`。
4. 构造坐标交换错误 case，确认指标明显下降，用来证明测试能抓出 `(x,y)/(y,x)` 问题。

### 3.2 Visibility threshold sweep 只作诊断

统一重算默认使用模型原始 visibility，但建议额外记录 threshold `0.3 / 0.5 / 0.7` 的诊断表，确认 OA/AJ 差异不是单纯 visibility threshold artifact。主表仍固定用 repo recommended threshold 或 raw bool。

### 3.3 Per-dataset and pooled tables 分开

必须分别给：

- DAVIS table
- Kinetics table
- optional mean table

不要只给 pooled mean。

### 3.4 Runtime / memory 只做辅助

保留 runtime 和 memory，但不要让它影响第一阶段方向判断。当前目标是官方主指标提升，不是部署。

## 4. 建议的执行补丁

### 4.1 给 `attempt_0_unified_reproduction_checklist` 增补

建议新增章节：

1. `Evaluator Parity Check`
2. `Adapter Sanity Tests`
3. `Unified Export Schema v1`
4. `Partial vs Final Attempt 0 Decision`
5. `Track-On family vs Track-On-R fine-tune decision split`

### 4.2 给 `trackonr_execution_checklist` 增补

建议修改 Week 1 输出：

- `track_on_env_manifest.md`
- `track_on_checkpoint_manifest.md`
- `trackon_adapter_sanity_report.json`
- `trackon_metric_parity_report.json`
- `track_on2_repo_native_metrics.json`
- `track_onr_repo_native_metrics.json`
- `track_on_unified_rescoring.json`
- `week1_partial_decision.md`

其中 `week1_partial_decision.md` 只能输出：

- `provisional go Track-On family`
- `hold Track-On, investigate protocol/adapter gap`
- `stop Track-On-R fine-tune, keep or drop Track-On2 separately`
- `stop Track-On family, move to TAPNext++`

### 4.3 给 `new_direction_ranking` 增补

建议在排序结论后加一句：

> 排序是执行优先级，不是免比较结论。Track-On-R 可以先跑，但最终主 baseline 仍由 Attempt 0 unified rescoring table 决定。

## 5. 当前最终建议

可以开始执行 `docs/trackonr_execution_checklist_2026-06-13.md`，但执行口径应改成：

1. 它是 Attempt 0 的 Track-On 子阶段。
2. Day 1 同时做 DINOv3、mmcv、DAVIS/Kinetics dataset smoke。
3. Day 4 前必须实现 adapter sanity check。
4. Day 5/6 前必须做本仓库 evaluator 与官方 TAP-Vid metrics 的 parity check。
5. Week 1 只能做 partial decision；完整 Attempt 0 仍需补 TAPNext++ / AllTracker。

如果这些补丁落实，当前路线没有根本问题；最大风险不是方向选错，而是统一评测适配细节出错导致错误决策。
