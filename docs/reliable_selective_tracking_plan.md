# Calibrated Selective Point Tracking

## 1. 这次真正的主线

我们不再把论文主线放在“继续把平均 AJ 往上抠一点”的方向上。
当前最适合继续推进的主线是：

**可校准的选择性点追踪（calibrated selective point tracking）**

核心问题不是“模型能不能输出轨迹”，而是：

- 模型能不能判断这条轨迹值不值得信。
- 在长遮挡、重现、域偏移时，模型能不能给出可靠的置信度。
- 当不确定时，模型能不能做出可控的 abstain / fallback，而不是硬输出错误结果。

这条线比继续做纯 relocalization / TTA / 微弱 AJ 提升更像一篇真正能落地的论文。

## 2. 为什么要 pivot

### 2.1 现有 TTA 方向上限太低

当前全量 DAVIS 的 TTA 汇总几乎是零增益，见：

- [tta_temporal_only_full_davis_seed_summary.md](/gemini/code/FSPT/outputs/tta_temporal_only_full_davis_seed_summary.md)

长遮挡子集虽然能看到一些信号，但结果对阈值非常敏感，而且会出现位置精度变差的情况，见：

- [tta_longocc_temporal_only_t30_seed_summary.md](/gemini/code/FSPT/outputs/tta_longocc_temporal_only_t30_seed_summary.md)
- [tta_temporal_only_full_davis_t10_seed42_summary.md](/gemini/code/FSPT/outputs/tta_temporal_only_full_davis_t10_seed42_summary.md)

这说明 TTA 这条线不是“还差一点点”，而是它对当前强基线的边际收益已经很小。

### 2.2 MegaDepth 预训练还没有形成可用迁移收益

几何 pair 预训练可以保留为辅助资产，但现有结果还不足以把它作为主论文核心。

- [fspt_megadepth_pairs_pretrain_full/epoch_metrics.jsonl](/gemini/code/FSPT/outputs/fspt_megadepth_pairs_pretrain_full/epoch_metrics.jsonl)
- [eval_fspt_megadepth_pairs_pretrain_all/davis_results.json](/gemini/code/FSPT/outputs/eval_fspt_megadepth_pairs_pretrain_all/davis_results.json)

现阶段更合理的做法，是把它降级成辅助手段，而不是继续押主线。

### 2.3 verifier / gate 方向有真实信号

当前 verifier 不是“没东西”，而是“它能分出好坏，只是我们还没有把它提升成一个完整问题定义”。

现有结果里，verifier 已经能看到明显的 oracle-gap 相关信号，以及 precision / recall / coverage 的结构性差异，见：

- [fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/summary.json](/gemini/code/FSPT/outputs/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/summary.json)
- [fspt_routeA_stage3_verifier_v1_from_posterior/summary.json](/gemini/code/FSPT/outputs/fspt_routeA_stage3_verifier_v1_from_posterior/summary.json)

这比继续追求一个几乎看不见的 AJ 提升更有研究价值。

## 3. 论文应该怎么讲

### 3.1 核心问题

强 point tracker 在 easy case 上已经很强，但在以下场景里会不稳定：

- 长遮挡后重现
- 外观变化明显
- real video domain shift
- 候选轨迹和 base 轨迹很接近，但实际好坏并不明显

因此，真正缺的不是“再造一个更复杂的 tracker”，而是：

**一个能校准置信度、控制风险、并在不确定时选择 abstain 的 tracking 系统。**

### 3.2 主贡献

建议把贡献收成 3 点：

1. **Reliability head**
   预测 candidate 是否真的比 base 更好，而不是只给一个粗糙 gate。

2. **Selective tracking protocol**
   用 risk-coverage / selective AJ / calibration 指标来评估 tracker 的“可信度”，而不只看平均 AJ。

3. **Long-occlusion application**
   在 long-occ 场景中展示它能稳定提升可靠性，并且不会伤害 easy cases。

如果后面还能把可靠性分数用于 real-video pseudo-label filtering，那可以作为第二阶段贡献。

## 4. 最小可行实验顺序

### Stage 1: 固定一个 base 和一个 candidate bank

不要再同时试太多路线。先固定：

- base tracker: `CoTracker3`
- candidate source: 当前最稳定的 visible-bank / posterior relocalization 路线
- verifier: 当前 `verifier` / `policy_gate` 里最成熟的分支

### Stage 2: 把问题改写成“谁更可信”

训练目标不再是单纯的轨迹回归，而是：

- candidate 是否优于 base
- 置信度是否可校准
- 选择性输出的 risk-coverage 曲线是否合理

### Stage 3: 评估要换成“可信度”视角

除了 TAP-Vid 官方指标，还要补：

- selective AJ
- coverage vs risk 曲线
- AURC / risk-coverage summary
- ECE / calibration error
- acceptance precision / recall
- oracle gap closed

### Stage 4: 只把 pseudo-label filtering 当第二阶段

如果 verifier/calibrator 真能稳定筛掉坏样本，再把它用于 real-video pseudo-label filtering。

这个步骤能把“可靠性论文”升级成“能带来训练收益的论文”。

## 5. 这条线和已有工作的关系

我们不再试图和下面这些工作正面拼“更强 tracker”：

- CoTracker3
- TAPTRv3
- Track-On / Track-On2
- ReTracker
- MFTIQ

这些工作已经把性能主线占得很满。

我们的差异点是：

- 不做单纯更强匹配
- 不做单纯更长记忆
- 不做单纯更大数据
- 而是做 **可校准的可靠性控制**

这比“安全门控”更完整，因为它是一个可测量、可校准、可报告风险覆盖的 tracking 问题。

## 6. 投稿策略

### 6.1 今年下半年的现实判断

ECCV 2026 已经错过，官方页面显示：

- paper registration deadline: 2026-02-26
- submission deadline: 2026-03-05

因此，当前更现实的路线是：

- **先按 journal 标准做厚**
- 目标：`Pattern Recognition` / `IJCV` / `TPAMI`

### 6.2 后续会议窗口

如果后续实验做到足够强，再看下一轮会议窗口，不把当前进度押在已经过期的 2026 会议截点上。

## 7. 现在不要再做什么

- 不要继续把 TTA 当主线。
- 不要继续把 MegaDepth 预训练当主线。
- 不要继续把“更强 relocalization”当唯一目标。
- 不要只盯着 AJ 单点提升。

现在最该做的是把问题定义改成：

**tracking 的输出是否可靠，以及这种可靠性能否被校准和利用。**

