# Experiment Review Table (2026-06-09)

这份文档用于给外部 AI 或合作者快速审阅仓库中的实验资产、结果口径和研究边界。

重点回答两个问题：

1. 仓库里已经做过哪些实验，哪些结果可信。
2. 当前工作到底是不是“真正接入点追踪模型后的第一帧恢复能力”。


## 1. 先给结论

当前仓库里实际上并行存在三条不同实验线：

| 线 | 本质 | 当前状态 | 是否已经是“真实点追踪恢复” |
|---|---|---|---|
| RouteA / relocal | 真正接入 tracker 的集成路线 | 有代码和多次 smoke / eval，但几乎没有稳定增益 | 是，但当前没有有效正结果 |
| PRT benchmark / selector | 基于 oracle-defined recovery event 的后处理恢复线 | 有清晰 benchmark、诊断和稳定的 hand-crafted baseline | 不是 |
| world-state diagnosis | 用 GT 3D / noisy depth 分析问题上限与瓶颈 | 诊断证据很强，学习方法还弱 | 不是 |

最重要的判断：

- 当前**最硬的正结果**来自 `PRT benchmark + hand-crafted support-memory selector`。
- 当前**还没有**一个“真实接入 tracker 推理流程、并稳定改善第一帧恢复”的方法结果。
- 所以如果对外表述，必须把当前工作定位成：
  - `benchmark + diagnosis + strong post-hoc recovery baseline`
- 不能表述成：
  - `an integrated point tracker with verified recovery capability`


## 2. 三条实验线的关系

### 2.1 用户最初想做的方向

用户原始目标更接近：

- 点追踪模型本身在连续视频流中，遇到遮挡后，能否在重新出现第一帧真实恢复到正确位置。

这个目标至少包含三件事：

1. 模型要自己知道何时发生了恢复问题。
2. 模型要在真实推理流程中触发恢复分支。
3. 最终指标要在真实 tracker 输出上体现改善。

### 2.2 当前 PRT 线实际回答的问题

当前 PRT 线回答的是：

> 如果我已经知道这里发生了一个 recovery event，并且已经围绕它构造了 baseline 和 top-k candidates，那么 support memory 能否帮助我更好地选择恢复位置？

因此它不是完整 tracker 恢复能力，而是：

- `oracle-defined event benchmark`
- `post-hoc recovery selector`

### 2.3 当前 RouteA 线实际回答的问题

RouteA / relocal 线是真正更接近用户原目标的代码路线，因为它试图：

- 在 tracker 内部做 relocal / verifier / refinement
- 直接在 DAVIS / TAP-Vid 类评测里看整体 AJ/OA 变化

但目前的问题是：

- 这条线虽然是“真的集成了”，但当前几乎没有稳定有效的实验结果。


## 3. 总实验表

### 3.1 PRT benchmark / post-hoc recovery 主线

| 类别 | 实验/脚本 | 主要文件 | 当前结论 | 可否作为主证据 |
|---|---|---|---|---|
| benchmark 定义 | [docs/prt_benchmark_definition.md](/gemini/code/FSPT/docs/prt_benchmark_definition.md) | benchmark 定义文档 | 已明确 `occlusion recovery event`、分层与指标 | 是 |
| split 构建 | [scripts/build_prt_splits.py](/gemini/code/FSPT/scripts/build_prt_splits.py) | [docs/prt_build_status.md](/gemini/code/FSPT/docs/prt_build_status.md) | PRT query 定义可执行，样本量充足 | 是 |
| candidate cache | [scripts/build_prt_candidate_dataset.py](/gemini/code/FSPT/scripts/build_prt_candidate_dataset.py) | [docs/prt_candidate_dataset_status.md](/gemini/code/FSPT/docs/prt_candidate_dataset_status.md) | 已能稳定构建 baseline/candidate/support cache | 是 |
| 正式多序列分层数据 | [scripts/build_prt_serial.py](/gemini/code/FSPT/scripts/build_prt_serial.py) | [build_stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/build_stats.json) | 15 seq, stratified, `n=9627`，当前正式主数据 | 是 |
| 规则选择器 | [scripts/eval_prt_hybrid_selector.py](/gemini/code/FSPT/scripts/eval_prt_hybrid_selector.py) | [summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v2/summary.json) | 当前最强、最可信的正结果 | 是 |
| AUC 审计 | 同上 | [auc.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v2/auc.json) | `support_margin` 是唯一稳定强信号 | 是 |
| bootstrap 稳定性 | 同上 | [bootstrap_stability.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v2/bootstrap_stability.json) | 大样本下结果稳定 | 是 |
| learnable ranker 诊断 | [scripts/diagnose_prt_recoverability.py](/gemini/code/FSPT/scripts/diagnose_prt_recoverability.py) | [diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_ani10_stratified_effective_quick/diagnostic_results.json) | 旧 train_v2 泛化失败，但 overfit 证明模型并非完全学不会 | 是，但属于诊断而非主方法结果 |

### 3.2 RouteA / integrated tracker relocal 线

| 类别 | 实验/脚本 | 主要文件 | 当前结论 | 可否作为主证据 |
|---|---|---|---|---|
| RouteA 计划文档 | [docs/routeA_eval256_training_and_paper_plan.md](/gemini/code/FSPT/docs/routeA_eval256_training_and_paper_plan.md) | 计划文档 | 这条线面向真实 tracker 集成 | 否，属计划/执行记录 |
| 运行脚本 | [scripts/run_routeA_serial.py](/gemini/code/FSPT/scripts/run_routeA_serial.py) | 各 RouteA 输出目录 | 已有多轮 smoke/eval | 否 |
| quickcheck relocal | 现有 checkpoint 评估 | [summary.json](/gemini/code/FSPT/outputs/quickcheck_relocal_stats/summary.json) | `AJ_delta = 0.0` | 否 |
| dual reloc smoke | 各 RouteA smoke | [summary.json](/gemini/code/FSPT/outputs/fspt_routeA_allpair_dual_reloc_twoview_smoke256/summary.json) | `AJ_delta ≈ -3e-06`，实质无变化 | 否 |
| visiblebank / posterior relocal | Stage3 relocal | [summary.json](/gemini/code/FSPT/outputs/fspt_routeA_stage3_relocal_accept_visiblebank_posterior_l30_eval256/summary.json) | 最好也只有 `AJ_delta ≈ +3.5e-06` | 否 |
| verifier 集成 | Stage3 verifier | [summary.json](/gemini/code/FSPT/outputs/fspt_routeA_stage3_verifier_v1_from_posterior/summary.json) | `AJ_delta ≈ +3.18e-05`，量级太小，不能算有效改善 | 否 |

### 3.3 world-state / oracle 诊断线

| 类别 | 实验/脚本 | 主要文件 | 当前结论 | 可否作为主证据 |
|---|---|---|---|---|
| Stage0 GT 3D oracle | [scripts/eval_world_state_stage0.py](/gemini/code/FSPT/scripts/eval_world_state_stage0.py) | [outputs/stage0_world_state_oracle.json](/gemini/code/FSPT/outputs/stage0_world_state_oracle.json) | GT world-state 对长遮挡重入优势极大 | 是，属诊断证据 |
| Stage0 结论摘要 | - | [outputs/stage0_world_state_conclusion.md](/gemini/code/FSPT/outputs/stage0_world_state_conclusion.md) | `occ20_cam0.30`: 2D hold `186.29px` vs 3D hold `7.00px` | 是，属诊断证据 |
| Stage1 predicted depth | [scripts/eval_world_state_stage1.py](/gemini/code/FSPT/scripts/eval_world_state_stage1.py) | [outputs/stage1_predicted_depth.json](/gemini/code/FSPT/outputs/stage1_predicted_depth.json) | 已有结果资产，但摘要口径需单独整理 | 暂不直接进主表 |
| Stage2 learnable world-state refiner | [scripts/train_world_state_stage2.py](/gemini/code/FSPT/scripts/train_world_state_stage2.py) | [metrics.json](/gemini/code/FSPT/outputs/world_state_stage2_smoke/metrics.json) | smoke 结果弱，尚不能支撑方法结论 | 否 |


## 4. 当前最值得审阅的正式结果

### 4.1 正式主数据集

| 项目 | 数值 | 文件 |
|---|---:|---|
| n_sequences | 15 | [build_stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/build_stats.json) |
| per_seq_samples | 1000 budget | [build_stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/build_stats.json) |
| sampling_mode | stratified | [build_stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/build_stats.json) |
| total_samples | 9627 | [build_stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/build_stats.json) |
| merged cache | root merged | [dataset_cache.npz](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/dataset_cache.npz) |

说明：

- 这是当前仓库里 PRT 主结果唯一应该使用的正式多序列数据版本。
- 旧的 `15seq/1500` 结果实际上是单序列，不应再当正式多序列证据。

### 4.2 当前最强的 PRT 规则方法结果

来源：

- [summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/summary.json)
- [auc.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/auc.json)
- [bootstrap_stability.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/bootstrap_stability.json)
- [significance.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/significance.json)

| 项目 | 数值 |
|---|---:|
| 公式 | `10.0 * support_margin` |
| baseline median | `39.32` |
| oracle median | `21.95` |
| threshold=0 coverage | `0.790` |
| threshold=0 median | `37.01` |
| threshold=0 `<4px` | `0.078` |
| threshold=0 `better_frac` | `0.391` |
| 强制 100% coverage median | `39.07`（比 baseline 差） |

显著性（equal-weight sequence-level）：

- Sample-level paired permutation p = 0.010
- Sequence-level permutation p = 0.446（不显著）
- Sequence-grouped bootstrap mean diff CI90: [-1.68, +0.41]
- Subgroup 分析为 exploratory / uncorrected

正式表述：

> threshold=0 (79% coverage) 下 median 从 39.32 降到 37.01，sample-level p=0.01，但 sequence-level 不显著。

### 4.3 当前最重要的信号结论

来源：

- [auc.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/auc.json)

| 特征 | AUC | 结论 |
|---|---:|---|
| `support_margin` | `0.653` | 稳定有效 |
| `cand_score` | `0.442` | 弱且不稳定 |
| `cand_ncc` | `0.492` | 接近噪声 |

当前最可信判断：

- `support_margin` 是唯一稳定强信号。
- 当前 PRT 线最本质的发现不是“我们有个很强学习模块”，而是：
  - `support memory` 确实能帮助恢复第一帧，
  - 但这种帮助目前主要体现为 post-hoc selector 中的稳定信号。


## 5. 历史结果与正式结果的区分

### 5.1 早期小样本正结果

来源：

- [summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/summary.json)
- [significance.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/significance.json)
- [outputs/prt_adaptive_selector_final.json](/gemini/code/FSPT/outputs/prt_adaptive_selector_final.json)

| 项目 | 数值 |
|---|---:|
| n_val | `300` |
| baseline median | `36.23` |
| hybrid median | `32.97` |
| adaptive median | `31.78` |
| oracle median | `25.90` |

这批结果的定位：

- 历史上重要，证明过方向有潜力。
- 但现在不能再作为论文主结果，因为规模太小。

### 5.2 已废弃的“15seq/1500”主张

来源：

- [stats.json](/gemini/code/FSPT/outputs/prt_candidate_val_v2_15seq_wrong_single_seq/stats.json)
- [outputs/prt_15seq_audited_summary.json](/gemini/code/FSPT/outputs/prt_15seq_audited_summary.json)
- [outputs/prt_15seq_stratified.json](/gemini/code/FSPT/outputs/prt_15seq_stratified.json)
- [outputs/prt_adaptive_15seq.json](/gemini/code/FSPT/outputs/prt_adaptive_15seq.json)

问题：

- 名义写的是 `15 seq`。
- 实际是单序列 `ani10_new_f` 大样本。

因此：

- 这些结果只能做历史分析或分层启发。
- 不能再作为正式“多序列主表”。


## 6. 学习方法实验表

### 6.1 PRT 学习方法

| 方法 | 文件 | 结果摘要 | 判断 |
|---|---|---|---|
| confidence gate | [summary.json](/gemini/code/FSPT/outputs/prt_confidence_gate_v1/summary.json) | `median 33.40` vs baseline `36.23` on old 300-sample val | 早期有改善，但规模太小，不是正式主证据 |
| temporal verifier v2 clean | [best.json](/gemini/code/FSPT/outputs/prt_temporal_verifier_v2_clean/best.json) | `40.71` vs baseline `37.97` | 有效负结果 |
| sequence verifier smoke | [best.json](/gemini/code/FSPT/outputs/prt_sequence_verifier_smoke/best.json) | `39.66` vs baseline `37.97` | 比 temporal verifier 好一点，但仍输 baseline |
| support-memory ranker v1 | [best.json](/gemini/code/FSPT/outputs/prt_support_ranker_hybrid_v1/best.json) | `38.52` vs baseline `37.97` | 最接近主线，但仍未超过 hand-crafted baseline |
| recoverability diagnostic | [diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_ani10_stratified_effective_quick/diagnostic_results.json) | quick 模型泛化时塌缩为偏向 baseline | 说明数据/目标设计仍有问题 |
| train-overfit diagnostic | [diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_train_overfit_eff/diagnostic_results.json) | same-distribution overfit 可学 | 说明“完全不可学”这个判断不成立 |

### 6.2 RouteA / world-state 学习方法

| 方法 | 文件 | 结果摘要 | 判断 |
|---|---|---|---|
| world_state_selector_v1 | [metrics.json](/gemini/code/FSPT/outputs/world_state_selector_v1/metrics.json) | `30.32` vs baseline `21.79` | 负结果 |
| world_state_ranker_v1 | [metrics.json](/gemini/code/FSPT/outputs/world_state_ranker_v1/metrics.json) | `44.64` vs baseline `37.25` | 负结果 |
| world_state_dino_selector_v2 | [metrics.json](/gemini/code/FSPT/outputs/world_state_dino_selector_v2/metrics.json) | 大多选择 baseline，本质无效 | 负结果 |
| world_state_acceptor_v2 | [metrics.json](/gemini/code/FSPT/outputs/world_state_acceptor_v2/metrics.json) | 最好也未优于 baseline | 负结果 |
| world_state_dino_top1_acceptor_v2 | [metrics.json](/gemini/code/FSPT/outputs/world_state_dino_top1_acceptor_v2/metrics.json) | accept_rate 常为 0 | 负结果 |
| world_state_stage2_smoke | [metrics.json](/gemini/code/FSPT/outputs/world_state_stage2_smoke/metrics.json) | `reproj_median_px = 17.21` | 仅 smoke，尚不能支持方法贡献 |


## 7. 哪些结果最值得其他 AI 重点复核

建议外部审阅优先检查以下 6 个点：

1. [summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v2/summary.json) 的 operating point 命名是否需要重写。
2. `oracle gap utilization` 在旧文稿中是否存在错误复用。
3. [build_stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/build_stats.json) 的多序列分层采样是否满足正式论文要求。
4. 当前 PRT 工作是否应严格定位为 `benchmark + diagnosis + strong post-hoc baseline`。
5. RouteA 线是否还有继续投入价值，还是应暂时让位于 PRT benchmark 论文路线。
6. learned selector 的下一步是否应改成：
   - sequence split
   - effective zone only
   - candidate-vs-baseline ranking
   - support-margin-centered low-dimensional model


## 8. 关于“这是不是点追踪恢复方法”的明确回答

答案分两层。

### 8.1 如果问当前最强正结果

不是。

原因：

- 当前最强正结果来自 PRT/post-hoc 线。
- 它依赖 oracle-defined recovery events。
- 它不是在 tracker 连续推理过程中自动触发恢复。

所以它证明的是：

- 第一帧恢复是个真实问题；
- support memory 是重要线索；
- 在已知 recovery event 的条件下，post-hoc selector 可以稳定改善 baseline。

它**没有**证明：

- 一个真实 tracker 已经具备了更强的一体化恢复能力。

### 8.2 如果问仓库里是否存在“真实集成”的尝试

有。

就是 RouteA / relocal / verifier 这一条线。

但当前事实是：

- 它是真正集成进 tracker 的；
- 但它几乎没有给出可用的性能提升。

所以当前仓库的真实状态是：

- “集成线更真实，但没结果。”
- “PRT 线结果最强，但它不是完整 tracker 集成方法。”


## 9. 当前最合理的对外定位

如果现在让其他 AI 或合作者复核方向，最诚实、最稳的说法应是：

> 该项目目前最成熟的部分是一个面向点追踪遮挡恢复第一帧的 benchmark 与诊断框架，并在 oracle-defined recovery events 上得到一个稳定有效的 support-memory 后处理基线。它尚未证明一个真实集成到 tracker 推理流程中的恢复模块可以稳定提升性能。

一句话压缩版：

- `强 benchmark / 强诊断 / 有效 post-hoc baseline / 尚无稳定 integrated tracker result`


## 10. 相关文档索引

建议配套一起给外部 AI：

- [docs/prt_experiment_inventory_2026-06-09.md](/gemini/code/FSPT/docs/prt_experiment_inventory_2026-06-09.md)
- [docs/prt_recovery_handoff_2026-06-05.md](/gemini/code/FSPT/docs/prt_recovery_handoff_2026-06-05.md)
- [docs/prt_benchmark_definition.md](/gemini/code/FSPT/docs/prt_benchmark_definition.md)
- [docs/routeA_eval256_training_and_paper_plan.md](/gemini/code/FSPT/docs/routeA_eval256_training_and_paper_plan.md)
- [outputs/stage0_world_state_conclusion.md](/gemini/code/FSPT/outputs/stage0_world_state_conclusion.md)
