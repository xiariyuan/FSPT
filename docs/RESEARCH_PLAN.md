# FSPT 研究计划

## 1. 当前总主线

旧的 TTA / 纯 relocalization 主线已经不再作为主论文方向。

**当前主线：verifier-guided pseudo-label filtering + calibrated selective tracking**

更直白地说：

- 让 verifier 判断哪些轨迹值得信。
- 让这个可靠性信号同时服务于测试时的选择性输出和训练时的伪标签筛选。
- 把“可靠性”从辅助指标变成数据质量控制和推理决策的核心。

这比单纯的后置门控更强，因为它不仅决定“要不要输出”，还决定“哪些样本值得训练”。

## 2. 为什么要 pivot

### 2.1 TTA 线已经接近上限

现有 TTA 结果在全量 DAVIS 上几乎没有有效增益，长遮挡子集虽然有信号，但对阈值极其敏感，甚至会出现位置精度变差的情况。

相关汇总见：

- [tta_temporal_only_full_davis_seed_summary.md](/gemini/code/FSPT/outputs/tta_temporal_only_full_davis_seed_summary.md)
- [tta_longocc_temporal_only_t30_seed_summary.md](/gemini/code/FSPT/outputs/tta_longocc_temporal_only_t30_seed_summary.md)
- [tta_temporal_only_full_davis_t10_seed42_summary.md](/gemini/code/FSPT/outputs/tta_temporal_only_full_davis_t10_seed42_summary.md)

结论很明确：这条线可以作为分析材料，但不适合作为主论文继续重押。

### 2.2 MegaDepth 预训练还没有形成稳定迁移收益

几何 pair 预训练可以保留为辅助模块，但现有实验还不足以把它作为主贡献。

相关结果见：

- [fspt_megadepth_pairs_pretrain_full/epoch_metrics.jsonl](/gemini/code/FSPT/outputs/fspt_megadepth_pairs_pretrain_full/epoch_metrics.jsonl)
- [eval_fspt_megadepth_pairs_pretrain_all/davis_results.json](/gemini/code/FSPT/outputs/eval_fspt_megadepth_pairs_pretrain_all/davis_results.json)

当前更合理的定位是：

- 几何预训练作为备选增强手段
- 不是论文主故事

### 2.3 verifier / gate 方向有真实结构性信号

现有 verifier / policy gate 已经能看到：

- candidate vs base 的可分性
- coverage / precision / recall 的结构
- long-occlusion 上更强的 oracle-gap 分解

相关结果见：

- [fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/summary.json](/gemini/code/FSPT/outputs/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/summary.json)
- [fspt_routeA_stage3_verifier_v1_from_posterior/summary.json](/gemini/code/FSPT/outputs/fspt_routeA_stage3_verifier_v1_from_posterior/summary.json)

这说明真正值得深化的不是“更强回归”，而是“更好的可靠性建模”。

## 3. 论文应该怎么讲

### 3.1 核心问题

强 tracker 在 easy case 上已经接近饱和，但在以下场景仍然容易失控：

- 长遮挡后的重现
- 外观变化和域偏移
- candidate 和 base 都合理，但谁更好并不显然

因此，我们关注的不是“再造一个更大的 tracker”，而是：

**如何对 point tracking 的输出做可靠性估计，并在不确定时进行选择性输出。**

### 3.2 核心贡献

建议收敛成 3 个贡献点：

1. **Reliability head**
   - 学习判断 candidate 是否比 base 更可信
   - 不只是输出一个 gate，而是输出可分析的置信分数

2. **Selective tracking evaluation**
   - 引入 risk-coverage / selective AJ / calibration 指标
   - 把可靠性作为一等公民，而不是只看平均 AJ

3. **Long-occlusion application**
   - 在长遮挡子集上证明可靠性建模有用
   - 同时保证 easy case 不被伤害

如果后续能把可靠性分数进一步用于 real-video pseudo-label filtering，那可以作为第二阶段扩展。

## 4. 最小可行实验顺序

### Stage 1: 固定 base 和 candidate

先固定一个稳定组合，不再横跳：

- base tracker: `CoTracker3`
- candidate source: 当前最稳定的 visible-bank / posterior / relocalization 路线
- reliability module: 现有 `verifier` 或 `policy_gate`

### Stage 2: 把监督目标改成“谁更好”

训练目标不再只是位置误差，而是：

- candidate 是否优于 base
- 这种判断能否被稳定校准
- 在不同阈值下能否形成合理的 coverage / risk 曲线

### Stage 3: 统一评估协议

除了 TAP-Vid 官方指标，还要补充：

- selective AJ
- coverage vs risk 曲线
- AURC / risk-coverage summary
- calibration error，例如 ECE 或 Brier score
- acceptance precision / recall
- oracle gap closed

### Stage 4: 再考虑 pseudo-label filtering

如果 verifier 稳定，可进一步用于：

- real-video pseudo-label 过滤
- reliable track mining
- 作为第二阶段训练增益

这个阶段是加分项，不是当前主线的前提。

## 5. 相关工作如何对齐

这条线不再正面和下面这些工作拼“更强 tracker”：

- CoTracker3
- TAPTRv3
- Track-On
- ReTracker
- TAPNext++
- MFTIQ

我们的差异点是：

- 不是只做更强匹配
- 不是只做更长上下文
- 不是只做更大规模数据
- 而是做 **可校准的可靠性控制**

这条线和最近的 verifier-guided pseudo-labeling、MFTIQ 的质量估计、以及 selective classification / conformal prediction 的方法论是相通的，但我们要把它落实到 point tracking 的可信输出上。

## 6. 实验与投稿策略

### 6.1 实验优先级

1. 先把 reliability head 的监督目标做对。
2. 再把 selective metrics 做全。
3. 再看 long-occlusion 是否稳定提升。
4. 最后才考虑 pseudo-label filtering 或 real-video adaptation。

### 6.2 投稿策略

今年下半年更现实的选择是先按期刊标准做厚，目标优先级建议：

- `Pattern Recognition`
- `IJCV`
- `TPAMI`

如果后续实验非常扎实，再看下一轮会议窗口。

## 7. 当前不要再做什么

- 不要继续把 TTA 当主线。
- 不要继续把几何预训练当主线。
- 不要继续把“只要再抠一点 AJ”当目标。
- 不要把主故事写成“安全决策”。

现在应该讲的是：

**point tracking 的可靠性估计、选择性输出和风险控制。**
