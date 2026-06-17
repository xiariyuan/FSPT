# PRT Experiment Inventory

## 1. 这份文档的用途

这份文档用于给其他 AI 或合作者快速审阅当前仓库中与 PRT / recovery 相关的实验资产、结果口径和研究边界。

目标不是写论文，而是回答三个问题：

- 仓库里到底做过哪些实验
- 哪些结果可信，哪些已经废弃
- 当前工作到底是不是“真实接入点追踪模型”的恢复方法


## 2. 研究问题当前定义

当前问题已经不再建议表述为“单次 re-entry”。

更准确的定义是：

- `occlusion recovery event`
- 即一个点在遮挡结束后第一次重新可见的那一帧恢复任务
- 一条轨迹可以贡献多个 recovery events

当前 PRT 管线的实际形式是：

1. 给定一个 recovery event
2. 用 baseline 几何传播得到一个恢复位置
3. 在 baseline 周围做 top-k 局部候选检索
4. 再用 support memory 决定是否用候选替换 baseline


## 3. 当前工作边界

这一点非常关键。

### 3.1 当前工作已经做到的

- 已经定义了一个 recovery benchmark
- 已经构建了候选数据集
- 已经证明 support memory 是稳定主信号
- 已经在大样本多序列上证明 hand-crafted support-memory selector 能改善 baseline

### 3.2 当前工作还没有做到的

- 还没有把这个 recovery module 真正接入一个完整点追踪模型的在线/端到端预测流程
- 训练与评测仍然基于“上帝视角抽取出来的 recovery events”
- support memory 使用的是 recovery event 已知后回溯出来的遮挡前支持帧
- 候选池也是围绕 oracle-defined recovery event 单独构建的

因此，当前工作更准确的定位是：

- `point-tracking recovery benchmark + support-memory-based post-hoc recovery selector`

而不是：

- `an end-to-end point tracker with integrated recovery capability`

这会直接影响论文定位和写法。


## 4. 数据构建实验

### 4.1 Recovery event / candidate dataset builder

核心脚本：

- [scripts/build_prt_splits.py](/gemini/code/FSPT/scripts/build_prt_splits.py)
- [scripts/build_prt_candidate_dataset.py](/gemini/code/FSPT/scripts/build_prt_candidate_dataset.py)
- [scripts/build_prt_serial.py](/gemini/code/FSPT/scripts/build_prt_serial.py)
- [scripts/build_prt_parallel.py](/gemini/code/FSPT/scripts/build_prt_parallel.py)

当前推荐使用的是：

- `build_prt_candidate_dataset.py`
- `build_prt_serial.py`

并行脚本曾经存在口径问题，不建议作为当前主入口。


### 4.2 旧版小规模 train / val cache

训练缓存：

- [outputs/prt_candidate_train_v2/stats.json](/gemini/code/FSPT/outputs/prt_candidate_train_v2/stats.json)

验证缓存：

- [outputs/prt_candidate_val_v2_3seq/stats.json](/gemini/code/FSPT/outputs/prt_candidate_val_v2_3seq/stats.json)

主要特点：

- train：`n=200`
- val：`n=300`
- 用于早期规则选择器和学习模块 smoke test

限制：

- 样本太少
- train / val 分布不稳
- 不足以支撑正式 learned module 结论


### 4.3 错误的“15seq”旧缓存

已确认错误口径：

- [outputs/prt_candidate_val_v2_15seq_wrong_single_seq/stats.json](/gemini/code/FSPT/outputs/prt_candidate_val_v2_15seq_wrong_single_seq/stats.json)
- [outputs/prt_candidate_val_v2_15seq_wrong_single_seq/preview.json](/gemini/code/FSPT/outputs/prt_candidate_val_v2_15seq_wrong_single_seq/preview.json)

问题：

- 名义上写 `15 seq`
- 实际是单序列大样本

状态：

- 可用于历史分析
- 不能再当正式多序列主结果


### 4.4 当前正式多序列分层采样数据

正式数据目录：

- [outputs/prt_val_15seq_1000each_stratified/build_stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/build_stats.json)
- [outputs/prt_val_15seq_1000each_stratified/dataset_cache.npz](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/dataset_cache.npz)
- [outputs/prt_val_15seq_1000each_stratified/samples_meta.jsonl](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/samples_meta.jsonl)

总统计：

| 项目 | 数值 |
|---|---:|
| 序列数 | 15 |
| 总样本数 | 9627 |
| 采样模式 | stratified |

单序列 stats 位于：

- [outputs/prt_val_15seq_1000each_stratified/ani10_new_f/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/ani10_new_f/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/ani14_new_f/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/ani14_new_f/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/ani16_new_/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/ani16_new_/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/ani1_new_/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/ani1_new_/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/ani1_new_f/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/ani1_new_f/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/ani3_new_/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/ani3_new_/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/cnb_dlab_0215_ego2/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/cnb_dlab_0215_ego2/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/cnb_dlab_0225_ego2/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/cnb_dlab_0225_ego2/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/egobody_3rd/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/egobody_3rd/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/r1_new_/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/r1_new_/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/r1_new_f/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/r1_new_f/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/scene_d78_3rd/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/scene_d78_3rd/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/scene_d78_ego1/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/scene_d78_ego1/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/scene_d78_ego2/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/scene_d78_ego2/stats.json)
- [outputs/prt_val_15seq_1000each_stratified/scene_recording_20210910_S05_S06_0_ego1/stats.json](/gemini/code/FSPT/outputs/prt_val_15seq_1000each_stratified/scene_recording_20210910_S05_S06_0_ego1/stats.json)

当前正式主数据集应以这一版为准。


## 5. 规则方法实验

### 5.1 3-seq / 300 样本早期正结果

文件：

- [outputs/prt_hybrid_selector_v3_3seq/summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/summary.json)
- [outputs/prt_hybrid_selector_v3_3seq/significance.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/significance.json)
- [outputs/prt_adaptive_selector_final.json](/gemini/code/FSPT/outputs/prt_adaptive_selector_final.json)

主要数字：

| 设置 | Median |
|---|---:|
| Baseline | 36.23 |
| Hybrid full | 32.97 |
| Adaptive | 31.78 |
| Oracle | 25.90 |

状态：

- 历史上重要
- 但现在不应再作为论文主结果


### 5.2 错误口径下的“15seq”旧分析

文件：

- [outputs/prt_15seq_audited_summary.json](/gemini/code/FSPT/outputs/prt_15seq_audited_summary.json)
- [outputs/prt_15seq_stratified.json](/gemini/code/FSPT/outputs/prt_15seq_stratified.json)
- [outputs/prt_adaptive_15seq.json](/gemini/code/FSPT/outputs/prt_adaptive_15seq.json)

状态：

- 只能作为历史分析
- 不再用于正式主表


### 5.3 当前正式主结果：15seq stratified v3

文件：

- [outputs/prt_hybrid_selector_15seq_stratified_v3/summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/summary.json)
- [outputs/prt_hybrid_selector_15seq_stratified_v3/auc.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/auc.json)
- [outputs/prt_hybrid_selector_15seq_stratified_v3/bootstrap_stability.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/bootstrap_stability.json)
- [outputs/prt_hybrid_selector_15seq_stratified_v3/threshold_sweep_val.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/threshold_sweep_val.json)
- [outputs/prt_hybrid_selector_15seq_stratified_v3/significance.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_15seq_stratified_v3/significance.json)

当前最优公式：

- `10.0 * support_margin`（不含 cand_score / cand_ncc）

Operating point 说明：

- summary.json 中 `full_coverage` 的 threshold=0，实际 coverage=0.79（不是 100%）
- 强制 100% coverage（threshold=-1.35）时 median=39.07，比 baseline 39.32 更差
- 最优工作点是 threshold=0, coverage=79%

当前正式主表数字：

| 方法 | Coverage | Median | <4px | Better frac |
|---|---:|---:|---:|---:|
| Baseline | 1.00 | 39.32 | 0.150 | - |
| Oracle | 1.00 | 21.95 | 0.283 | - |
| Hybrid @ thr=0 | 0.79 | 37.01 | 0.078 | 0.391 |
| Hybrid @ train thr | 0.92 | 37.48 | 0.072 | 0.415 |

特征 AUC：

| Feature | AUC |
|---|---:|
| support_margin | 0.653 |
| cand_score | 0.442 |
| cand_ncc | 0.492 |

Bootstrap：

| 方法 | Median mean ± std | P5 | P95 |
|---|---:|---:|---:|
| Hybrid @ 79% | 37.11 ± 0.91 | 35.67 | 38.71 |
| Baseline | 39.32 | - | - |

Significance（来自 significance.json）：

- Sample-level paired permutation p = 0.010（显著）
- Equal-weight sequence-level permutation p = 0.446（不显著）
- Sequence-grouped bootstrap mean diff CI90: [-1.68, +0.41]（包含 0）
- Subgroup 分析为 exploratory / uncorrected，不作为确认性结论

正式表述：

> On the 15-sequence stratified validation set (n=9627), the support-margin hybrid selector reduces median error from 39.32 to 37.01 px at threshold=0 (79% coverage), but the gain is not significant under equal-weight sequence-level testing (p=0.45).

这一组是当前仓库里最值得当主结果的规则方法结果。


## 6. 学习方法实验

### 6.1 Learned confidence gate

脚本：

- [scripts/train_prt_confidence_gate.py](/gemini/code/FSPT/scripts/train_prt_confidence_gate.py)

结果：

- [outputs/prt_confidence_gate_v1/summary.json](/gemini/code/FSPT/outputs/prt_confidence_gate_v1/summary.json)

状态：

- 负结果
- 不建议继续投入


### 6.2 Support-memory learned ranker

脚本：

- [scripts/train_prt_support_memory_ranker.py](/gemini/code/FSPT/scripts/train_prt_support_memory_ranker.py)

结果：

- [outputs/prt_support_ranker_hybrid_v1/metrics.json](/gemini/code/FSPT/outputs/prt_support_ranker_hybrid_v1/metrics.json)
- [outputs/prt_support_ranker_hybrid_v1/best.json](/gemini/code/FSPT/outputs/prt_support_ranker_hybrid_v1/best.json)
- [outputs/prt_support_ranker_hybrid_v1/selective.json](/gemini/code/FSPT/outputs/prt_support_ranker_hybrid_v1/selective.json)

状态：

- 目前还没有超过 hand-crafted support-memory baseline
- 仍可继续，但不能作为当前论文主结果


### 6.3 Recoverability diagnostics

脚本：

- [scripts/diagnose_prt_recoverability.py](/gemini/code/FSPT/scripts/diagnose_prt_recoverability.py)

结果文件：

- [outputs/prt_recoverability_diag_smoke/diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_smoke/diagnostic_results.json)
- [outputs/prt_recoverability_diag_ani10_quick/diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_ani10_quick/diagnostic_results.json)
- [outputs/prt_recoverability_diag_ani10_effective_quick/diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_ani10_effective_quick/diagnostic_results.json)
- [outputs/prt_recoverability_diag_ani10_stratified_effective_quick/diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_ani10_stratified_effective_quick/diagnostic_results.json)
- [outputs/prt_recoverability_diag_train_overfit_eff/diagnostic_results.json](/gemini/code/FSPT/outputs/prt_recoverability_diag_train_overfit_eff/diagnostic_results.json)

目前结论：

- 用旧 `train_v2` 泛化到新分层序列，quick MLP 在 effective zone 上塌缩为“全信 baseline”
- 但用 train=val 的 overfit 测试，MLP 能学到有效行为

这说明：

- 不是代码完全坏了
- 而是旧训练集太小、分布不匹配、目标设计不对


### 6.4 Temporal / sequence verifier

脚本：

- [scripts/train_prt_temporal_verifier.py](/gemini/code/FSPT/scripts/train_prt_temporal_verifier.py)
- [scripts/train_prt_sequence_verifier.py](/gemini/code/FSPT/scripts/train_prt_sequence_verifier.py)

结果：

- [outputs/prt_temporal_verifier_v1/metrics.json](/gemini/code/FSPT/outputs/prt_temporal_verifier_v1/metrics.json)
- [outputs/prt_temporal_verifier_v2_clean/metrics.json](/gemini/code/FSPT/outputs/prt_temporal_verifier_v2_clean/metrics.json)
- [outputs/prt_sequence_verifier_smoke/metrics.json](/gemini/code/FSPT/outputs/prt_sequence_verifier_smoke/metrics.json)
- [outputs/prt_sequence_verifier_smoke/feature_audit.json](/gemini/code/FSPT/outputs/prt_sequence_verifier_smoke/feature_audit.json)

状态：

- 目前不是主线
- 可以视为探索性旁支


## 7. 负结果 / 低 ROI 方向

### 7.1 CoTracker3 smoke test

结果：

- [outputs/cotracker3_prt_online_smoke.json](/gemini/code/FSPT/outputs/cotracker3_prt_online_smoke.json)

状态：

- 负结果
- 仅可作为 sanity check，不足以支撑正式对比


### 7.2 Pooling ablation

结果：

- [outputs/prt_pooling_ablation.json](/gemini/code/FSPT/outputs/prt_pooling_ablation.json)

结论：

- `mean / max / softmax / per-frame-max` 差异极小
- 不是主瓶颈


## 8. 当前最可信的实验结论

### 8.1 已成立的

1. `support_margin` 是唯一稳定有效的信号
2. hand-crafted support-memory selector 在 15 序列、`9627` 样本上能稳定优于 baseline
3. 候选池里经常已有更好答案，瓶颈主要在 selection
4. 单靠 `cand_score` 和 `cand_ncc` 不能解释改善

### 8.2 尚未成立的

1. learned selector 超过 hand-crafted selector
2. end-to-end tracker 中真实集成 recovery module 后还能保持改善
3. 在真实视频数据上也成立


## 9. 当前最关键的研究问题

用户原始目标更接近：

- 点追踪模型本身在遮挡恢复第一帧是否能真正恢复

而当前工作实际上做到的是：

- 在 oracle-defined recovery events 上，构造 post-hoc support-memory selector，证明 recovery 问题存在、support memory 有用、简单 selector 可以改善 baseline

这两者不等价。

### 9.1 为什么不等价

当前数据构建使用了“上帝视角”信息：

- recovery event 是先从 GT 轨迹里抽出来的
- support frames 是遮挡前 GT 可见帧
- baseline 是围绕已知 recovery event 构建的
- candidate search 也是围绕该 event 的局部窗口

这说明当前系统回答的是：

- “如果我已经知道这里发生了一个恢复事件，并且已经围绕它构造了候选，那么 support memory 能否帮助我选得更好？”

它还没有回答：

- “一个真实点追踪器在连续视频流里，能否自动知道此时需要恢复、自动触发 recovery branch、并真正把点在第一帧找回来？”

### 9.2 当前工作更接近什么

更准确的定位应是：

- `benchmark + diagnosis + post-hoc recovery selector`

而不是：

- `integrated point tracker with recovery capability`


## 10. 推荐给其他 AI 的审阅重点

建议其他模型优先复核以下问题：

1. 当前 `n=9627` 主结果的统计口径是否合理
2. `oracle gap utilization` 的计算是否在所有文稿中一致
3. 当前工作是否应定位为：
   - benchmark + diagnosis + strong baseline
   - 而不是 learned method paper
4. 如果要往“真实点追踪方法”推进，最小还需要补什么：
   - event trigger
   - integrated tracker inference
   - real-data validation
5. learned selector 下一步最合理的训练形式：
   - sequence split
   - candidate-vs-baseline ranking
   - support-margin-centered low-dimensional model


## 11. 当前建议的项目定位

如果今天就要定主线，最稳的定位是：

- `Occlusion Recovery Benchmark for Point Tracking`
- `Signal Diagnosis`
- `Strong Support-Memory Baseline`

learned module 应继续尝试，但不应成为论文成立的唯一前提。
