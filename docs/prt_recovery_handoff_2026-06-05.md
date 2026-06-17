# PRT Recovery Research Handoff

## 1. 研究方向

当前研究的问题是：

- 在点跟踪任务里，一个点被遮挡较长时间后，重新出现的第一帧如何恢复到正确位置。
- 关注的是 `re-entry first frame recovery`，不是整段视频平均误差。
- 当前系统的基础流程是：
  - 用遮挡前状态做几何传播，得到 baseline 恢复位置。
  - 在 baseline 周围做局部 DINO 检索，拿到 top-k 候选。
  - 再决定是否用某个候选替换 baseline。

核心观察：

- 候选池里经常已经存在比 baseline 更好的答案。
- 真正瓶颈主要不是“搜不到候选”，而是“选不出正确候选”。


## 2. 当前结论概览

可信结论和不可信结论必须分开。

### 2.1 可信结论

- `support memory` 是目前唯一稳定有信号的线索。
- 简单显式规则在小规模验证集上能改善 baseline。
- learned gate 基本无效，问题不在 decision boundary。
- pooling 改动几乎无效，问题不在 support 聚合方式。
- off-screen return 不是当前局部恢复框架能解决的主要方向。
- 未来最值得继续的是：做一个 `history-conditioned learnable recovery module`。

### 2.2 不可信或暂不能当最终论文证据的结论

- 之前所谓 “15 sequences / 1500 samples” 的那版结果，实际是单序列数据，不是真多序列结论。
- 因此，任何基于那版数据写进论文主张的“整体有效”结论，都不能当最终证据。


## 3. 数据与结果口径审计

### 3.1 小规模 3-seq / 300 样本结果

这是当前最干净、最明确的一版正结果。

相关文件：

- [outputs/prt_candidate_val_v2_3seq/stats.json](/gemini/code/FSPT/outputs/prt_candidate_val_v2_3seq/stats.json)
- [outputs/prt_hybrid_selector_v3_3seq/summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/summary.json)
- [outputs/prt_hybrid_selector_v3_3seq/significance.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/significance.json)
- [outputs/prt_adaptive_selector_final.json](/gemini/code/FSPT/outputs/prt_adaptive_selector_final.json)

数据集统计：

- val 样本数：`300`
- re-entry 类型：
  - in-frame occlusion：`271`
  - off-screen return：`29`
- baseline median：`36.23 px`
- best candidate median：`26.73 px`
- candidate beats baseline frac：`0.78`

说明：

- 候选池里有明显可利用空间。
- top-k 候选不是问题根源，选择器才是主要瓶颈。

#### 3-seq hybrid selector

文件：

- [outputs/prt_hybrid_selector_v3_3seq/summary.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/summary.json)
- [outputs/prt_hybrid_selector_v3_3seq/significance.json](/gemini/code/FSPT/outputs/prt_hybrid_selector_v3_3seq/significance.json)

公式：

- `4.0 * support_margin + 0.5 * cand_score + 0.0 * cand_ncc`

结果：

- baseline：`36.23`
- oracle：`25.90`
- hybrid full coverage：`32.97`
- selective 75%：`31.30`

显著性：

- full coverage paired permutation p mean：`0.0494`
- full coverage paired permutation p median：`0.0386`
- selective 75% paired permutation p mean：`0.00042`

#### 3-seq adaptive selector

文件：

- [outputs/prt_adaptive_selector_final.json](/gemini/code/FSPT/outputs/prt_adaptive_selector_final.json)

策略：

- `occ_length >= 100` 时用 hybrid
- 否则保留 baseline

结果：

- baseline：`36.23`
- hybrid always：`32.97`
- adaptive：`31.78`
- improvement vs baseline：`12.3%`
- oracle gap utilization：`43.1%`

显著性：

- paired permutation p mean：`0.0028`
- bootstrap mean diff CI：`[-2.98, -0.83]`

解释：

- 小规模上这个结果是正的。
- 但样本规模仍然偏小，不能单独支撑高把握论文结论。


### 3.2 所谓 15-seq / 1500 样本结果，实际是单序列

这个问题非常关键。

错误来源对应文件：

- [outputs/prt_candidate_val_v2_15seq_wrong_single_seq/stats.json](/gemini/code/FSPT/outputs/prt_candidate_val_v2_15seq_wrong_single_seq/stats.json)
- [outputs/prt_candidate_val_v2_15seq_wrong_single_seq/preview.json](/gemini/code/FSPT/outputs/prt_candidate_val_v2_15seq_wrong_single_seq/preview.json)

事实：

- 配置里写的是 `max_sequences_per_split = 15`
- 但实际样本 `seq_name` 全来自 `ani10_new_f`
- 所以这版 `n=1500` 不是多序列结果，而是单序列大样本结果

这版结果仍然有分析价值，但不能当最终主结果。

相关汇总文件：

- [outputs/prt_15seq_audited_summary.json](/gemini/code/FSPT/outputs/prt_15seq_audited_summary.json)
- [outputs/prt_15seq_stratified.json](/gemini/code/FSPT/outputs/prt_15seq_stratified.json)
- [outputs/prt_adaptive_15seq.json](/gemini/code/FSPT/outputs/prt_adaptive_15seq.json)

#### 单序列 1500 样本总体结果

- baseline：`39.12`
- oracle：`29.27`
- best candidate only：`30.53`
- hybrid always：`38.90`
- adaptive occ>=100：`38.52`
- permutation p：`0.33 ~ 0.35`
- bootstrap diff CI：`[-0.90, +0.24]`

结论：

- overall 不显著。
- 说明 3-seq 小样本里的乐观结论不能直接外推。

#### 单序列 1500 样本分层结果

最关键文件：

- [outputs/prt_15seq_stratified.json](/gemini/code/FSPT/outputs/prt_15seq_stratified.json)

最重要观察：

- effective zone：`in-frame occlusion + 100-500 frames`
- 该区间样本数：`1175`
- baseline：`36.56`
- hybrid：`34.39`
- oracle：`26.48`
- delta：`2.16 px`

按类型：

- in-frame occlusion：
  - n=`1350`
  - baseline=`34.79`
  - hybrid=`32.96`
  - adaptive=`32.32`
- off-screen：
  - n=`150`
  - 基本无改善

按遮挡长度：

- `20-100`：hybrid 反效果
- `100-200`：有改善
- `200-500`：有改善
- `500+`：样本太少，不应强下结论

这版数据支持的真正说法是：

- support memory 在 `in-frame + 100-500` 这个区间内有用。
- 它不是一个整体稳健通杀的方案。


## 4. 已做过的实验

下面按“继续做价值”来分。

### 4.1 数据构建与问题定义

核心脚本：

- [scripts/build_prt_splits.py](/gemini/code/FSPT/scripts/build_prt_splits.py)
- [scripts/build_prt_candidate_dataset.py](/gemini/code/FSPT/scripts/build_prt_candidate_dataset.py)
- [scripts/build_prt_parallel.py](/gemini/code/FSPT/scripts/build_prt_parallel.py)

这些脚本负责：

- 识别 re-entry queries
- 生成 baseline 恢复位置
- 在 baseline 周围构建 top-k 局部候选
- 提取 query / baseline / candidate / support patches
- 缓存训练和评估所需数据

重要实现事实：

- `per-seq-samples=300` 或 `1000` 只是人工采样预算，不是数据上限。
- 底层采样是先 `shuffle` 候选，再按预算截断，不是顺序取前 N 个。


### 4.2 纯规则 hybrid selector

脚本：

- [scripts/eval_prt_hybrid_selector.py](/gemini/code/FSPT/scripts/eval_prt_hybrid_selector.py)

作用：

- 根据 support memory 相似度和候选分数，显式打分并选候选
- 支持：
  - 系数搜索
  - 阈值搜索
  - coverage-risk 曲线
  - bootstrap
  - 单特征 AUC 审计

主要发现：

- `support_margin` 是唯一稳定主信号
- `cand_score` 很弱
- `cand_ncc` 在小样本里偶尔有点用，但放大后趋近噪声


### 4.3 adaptive selector

文件：

- [outputs/prt_adaptive_selector_final.json](/gemini/code/FSPT/outputs/prt_adaptive_selector_final.json)
- [outputs/prt_adaptive_15seq.json](/gemini/code/FSPT/outputs/prt_adaptive_15seq.json)

思想：

- 短遮挡不替换
- 长遮挡再启用 support-memory selector

发现：

- 在 3-seq / 300 上提升明显
- 在单序列 1500 上 overall 不显著，但在 `100-500` 长遮挡区间仍有局部价值


### 4.4 learned confidence gate

脚本：

- [scripts/train_prt_confidence_gate.py](/gemini/code/FSPT/scripts/train_prt_confidence_gate.py)

结果文件：

- [outputs/prt_confidence_gate_v1/summary.json](/gemini/code/FSPT/outputs/prt_confidence_gate_v1/summary.json)

结论：

- 这条线价值不高。
- 典型结果：
  - accept_rate：`0.513`
  - median：`33.40`
  - better_frac：`0.263`
- 说明学一个 gate 不是主要增益来源。

更早的分析也显示：

- LR gate 的 CV AUC 约 `0.558`
- 这接近随机，说明当前特征不足以可靠判断“该不该替换”


### 4.5 support-memory learned ranker

脚本：

- [scripts/train_prt_support_memory_ranker.py](/gemini/code/FSPT/scripts/train_prt_support_memory_ranker.py)

结果文件：

- [outputs/prt_support_ranker_hybrid_v1/metrics.json](/gemini/code/FSPT/outputs/prt_support_ranker_hybrid_v1/metrics.json)
- [outputs/prt_support_ranker_hybrid_v1/selective.json](/gemini/code/FSPT/outputs/prt_support_ranker_hybrid_v1/selective.json)

作用：

- 输入 support patches、baseline patch、candidate patches 和元信息
- 学习在 `{baseline, cand1..K}` 中做多分类选择

现状判断：

- 这是未来最值得继续深做的脚本基础。
- 但现有版本还比较弱，尚未形成比显式 hybrid 明显更强的可靠结果。


### 4.6 temporal / sequence verifier

脚本：

- [scripts/train_prt_temporal_verifier.py](/gemini/code/FSPT/scripts/train_prt_temporal_verifier.py)
- [scripts/train_prt_sequence_verifier.py](/gemini/code/FSPT/scripts/train_prt_sequence_verifier.py)

结果文件：

- [outputs/prt_temporal_verifier_v1/metrics.json](/gemini/code/FSPT/outputs/prt_temporal_verifier_v1/metrics.json)
- [outputs/prt_temporal_verifier_v2_clean/metrics.json](/gemini/code/FSPT/outputs/prt_temporal_verifier_v2_clean/metrics.json)
- [outputs/prt_sequence_verifier_smoke/metrics.json](/gemini/code/FSPT/outputs/prt_sequence_verifier_smoke/metrics.json)
- [outputs/prt_sequence_verifier_smoke/feature_audit.json](/gemini/code/FSPT/outputs/prt_sequence_verifier_smoke/feature_audit.json)

判断：

- 做过，但目前不是最优主线。
- 更像探索过的旁支，还没有证明比 support-memory 主线更强。


### 4.7 pooling ablation

文件：

- [outputs/prt_pooling_ablation.json](/gemini/code/FSPT/outputs/prt_pooling_ablation.json)

结论：

- `mean / max / softmax / per-frame-max` 基本没差别
- AUC 差异极小
- best median 也几乎相同

结论很明确：

- support pooling 不是当前瓶颈


### 4.8 CoTracker3 / 框架替换 smoke test

脚本：

- [scripts/eval_cotracker3_prt.py](/gemini/code/FSPT/scripts/eval_cotracker3_prt.py)

结果文件：

- [outputs/cotracker3_prt_online_smoke.json](/gemini/code/FSPT/outputs/cotracker3_prt_online_smoke.json)

结果：

- `n=4`
- cotracker median：`164.8`
- hold_2d median：`39.85`

结论：

- 基于该 smoke test，纯在线 memory 框架没有表现出解决 re-entry 的能力。
- 这不是值得继续投入的主线。


## 5. 当前最重要的事实判断

### 5.1 哪些方向已经基本可以停

- 继续调 `alpha / beta / gamma`
- 继续调 threshold
- learned confidence gate
- pooling 变体
- 试图用当前局部框架解决 off-screen return
- 再做一次纯框架替换 smoke test

这些方向已经给过明确信号，继续投入边际收益很低。

### 5.2 哪些方向还有真正潜力

- 真正的多序列验证
- 做一个可学习的 `history-conditioned recovery module`
- 先只针对 `in-frame + 100-500` 的有效区间建模
- 未来若要做完整系统，再单独做 off-screen/global re-detection 分支


## 6. 正在运行的实验

当前运行状态：

- 正在跑的是 `1000/seq` 真多序列构建，不是旧的 `300/seq`
- 进程示例：
  - `scripts/build_prt_candidate_dataset.py --max-samples 1000 --output-dir outputs/prt_val_15seq_1000each/ani10_new_f`

当前目录：

- [outputs/prt_val_15seq_1000each](/gemini/code/FSPT/outputs/prt_val_15seq_1000each)

当前观察到的状态：

- 目录存在
- 暂时只看到一个子目录：
  - `ani10_new_f`
- 目前该目录还是空的，`build_stats.json` 尚未生成

因此当前不能宣称真多序列结果已经拿到。


## 7. 论文与文档现状

论文文件：

- [papers/latex/main.tex](/gemini/code/FSPT/papers/latex/main.tex)
- [papers/tcsvt_prt_draft_v1.md](/gemini/code/FSPT/papers/tcsvt_prt_draft_v1.md)
- [papers/related_work_positioning_notes.md](/gemini/code/FSPT/papers/related_work_positioning_notes.md)

图表资产：

- [outputs/paper_assets/main_table.json](/gemini/code/FSPT/outputs/paper_assets/main_table.json)
- [outputs/paper_assets/feature_auc.json](/gemini/code/FSPT/outputs/paper_assets/feature_auc.json)
- [outputs/paper_assets/coverage_risk_curve.json](/gemini/code/FSPT/outputs/paper_assets/coverage_risk_curve.json)
- [outputs/paper_assets/stratified_by_type.json](/gemini/code/FSPT/outputs/paper_assets/stratified_by_type.json)
- [outputs/paper_assets/stratified_by_occ.json](/gemini/code/FSPT/outputs/paper_assets/stratified_by_occ.json)
- [outputs/paper_assets/stratified_by_cam.json](/gemini/code/FSPT/outputs/paper_assets/stratified_by_cam.json)
- [outputs/paper_assets/stratified_2d.json](/gemini/code/FSPT/outputs/paper_assets/stratified_2d.json)

重要提醒：

- `papers/latex/main.tex` 是在错误的 “15 seq” 理解基础上扩展过的。
- 现在不应把它视为最终可信论文主稿。
- 在真多序列验证完成前，不建议继续做深度论文包装。


## 8. 建议给其他 AI 的复核重点

如果让其他模型复核，优先请它们检查下面这些点。

### 8.1 先核数据口径

- 核查 `outputs/prt_candidate_val_v2_15seq_wrong_single_seq/preview.json` 中 `seq_name`
- 确认之前的 1500 样本是否真的是单序列
- 审核当前 `outputs/prt_val_15seq_1000each` 的真实运行进度

### 8.2 再核研究判断

- support memory 是否确实是唯一稳定信号
- candidate pool 是否真的足够强，瓶颈是否主要在 ranking
- off-screen 是否应从当前主线里剥离

### 8.3 最后核未来方法

- 现有 `train_prt_support_memory_ranker.py` 是否适合作为 learnable recovery module 的起点
- 是否应把训练目标先限制在：
  - `reentry_type_id == in-frame`
  - `100 <= occ_length < 500`
- 是否需要把支持信息从“手工 pooled memory”升级成“带时间顺序的 history encoder”


## 9. 推荐的下一步计划

### 9.1 第一优先级

- 等待并审计真多序列 `1000/seq` 缓存构建
- 检查：
  - 实际完成了多少个序列
  - 每个序列有多少样本
  - 是否真正多序列
  - 是否存在严重分布偏斜

### 9.2 第二优先级

- 在真多序列数据上重跑：
  - baseline
  - hybrid always
  - adaptive
  - effective-zone 分层结果

### 9.3 第三优先级

- 继续方法开发，但不要再做简单规则修补
- 直接做 `learnable history-conditioned recovery module`

建议的第一版模块定义：

- 输入：
  - 遮挡前若干帧 support patches
  - support 的时间顺序
  - baseline patch
  - top-k candidate patches
  - 基础元信息：occ length / camera motion / local score
- 输出：
  - `{baseline, cand1..K, optional abstain}` 上的 logits
- 首先只打有效区间：
  - `in-frame`
  - `100 <= occ_length < 500`

### 9.4 明确不建议的下一步

- 再做更多 threshold sweep
- 再做 pooling ablation
- 再做 learned gate
- 再尝试用当前局部 selector 硬解 off-screen
- 在真多序列结果出来前继续包装论文主结论


## 10. 一句话总结

这个方向还活着，但当前真正有价值的下一步不是继续调规则，而是：

- 先拿到真实多序列证据，
- 再把 `support memory` 升级成一个可学习的恢复模块，
- 并只在已经证明有机会的有效区间内先做强。


## 11. 审阅整合后的正式方案

这一节吸收了多轮外部审阅意见，并结合当前项目状态做取舍。

### 11.1 正式问题定义

推荐把问题从“单次 re-entry”升级为：

- `occlusion recovery event`
- 定义为：一个点在遮挡结束后第一次重新可见的那一帧恢复任务
- 一条轨迹可以包含多个 recovery events

但要保留最小遮挡长度阈值：

- 建议继续使用 `occ_length >= 20`

原因：

- 太短的遮挡会让 baseline 本身已经足够准
- 会稀释 long-occlusion recovery 的信号


### 11.2 推荐论文定位

最合理的定位不是纯 benchmark，也不是纯理论框架，而是：

- `Benchmark + Diagnosis + Lightweight Learned Recovery Module`

推荐叙事：

- 社区已经意识到 long-occlusion re-detection 是盲点
- 我们把它细化成 causal point tracking 中的 recovery events
- 我们系统分析信号来源
- 然后提出一个轻量、可插拔、history-conditioned 的恢复模块


### 11.3 与 TAPNext++ 的关系

后续写作时应明确把工作放在最新社区语境里：

- TAPNext++ 已经提出 `AJ_RD`，说明 re-detection after occlusion 是当前 point tracking 的重要盲点
- 但 TAPNext++ 主要是“指出问题 + 定义评估”，不是专门做 recovery module
- 我们的定位应是：
  - formalize recovery events
- diagnose why current trackers fail
- provide a dedicated recovery mechanism

这会比“我们自己定义了一个新问题”更容易被接受。

补充定位建议：

- 不要把自己写成“另起炉灶定义新问题”
- 更合理的表述是：
  - 社区已经用 `AJ_RD` 这类指标确认这是盲点
  - 我们把它进一步细化为 `occlusion recovery events`
  - 并提供专门的诊断和恢复模块


### 11.4 数据与评估原则

必须新增一个明确的数据审计层。

建议新增审计脚本，自动输出：

- 实际序列数
- 每序列样本数
- 每序列 recovery event 数
- in-frame / off-screen 比例
- 遮挡长度分布
- 相机运动分布

评估时要注意事件不独立：

- 同一视频中的多个 recovery events 不是严格独立样本

因此建议：

- 结果至少按序列分组报告一次
- bootstrap 优先用 sequence-grouped / clustered bootstrap
- split 也应按 sequence 分组，而不是按 event 随机打散


### 11.5 真实数据优先级

真实数据优先级建议调整为：

1. `PointOdyssey`
2. `TAP-Vid-DAVIS`
3. `RGB-Stacking`

原因：

- `PointOdyssey` 已有基础设施，适合快速迭代
- `TAP-Vid-DAVIS` 更主流，更适合作为真实世界对照
- `RGB-Stacking` 可以作为补充验证


### 11.6 可学习模块的正式开发策略

建议分两步做，不要一开始把模块设计得过重。

#### Step 1: 最简 learnable recovery module

目标：

- 只验证“可学习模块是否能超过手工 hybrid”

建议输入：

- support patches
- baseline patch
- top-k candidate patches

建议结构：

- 简单 history encoder
- candidate-to-history cross attention
- baseline 作为第 0 个候选一起打分

先不要急着塞太多附加结构。

#### Step 2: 完整版本

在 Step 1 有效后，再加入：

- support 的时间顺序编码
- `occ_length`
- `camera_motion`
- 几何 / 相对位置信息
- 更强的 ranking-aware loss

这里有一个方法层面的根本判断需要始终记住：

- 当前框架主要是在固定候选池里做 `selector`
- 它的性能上限受候选池质量限制

因此后续方法线应分成两级：

- 一级目标：
  - 先把 `selector` 做强
  - 证明 learned selector 明显优于 hand-crafted hybrid
- 二级延伸：
  - 再考虑 `refiner`
  - 即在选中候选后，再预测一个局部残差修正

当前不建议一开始就直接做 refiner，原因是：

- 会显著增加建模和评估复杂度
- 当前还没有证明 selector 本身已经吃满了候选池价值

但在论文 discussion 中应明确：

- refinement 是自然下一步
- 它能突破固定候选池带来的上限约束


### 11.7 训练策略建议

除了模型结构，训练采样方式也要改。

当前 learned ranker 容易学成“优先相信 baseline”，需要从 batch 构造层面纠正。

建议按三类样本做平衡采样：

- `baseline-best`
- `candidate-clearly-better`
- `hard-negative candidate`

更一般地说，需要避免模型学成“保守地永远相信 baseline”。

建议把 `occ_length` 视为显式输入的天然不确定性信号，而不是继续依赖手工阈值：

- 短遮挡时，模型应更保守
- 长遮挡时，模型应更愿意利用 support memory 覆盖 baseline

也就是说，`occ>=100` 当前有效，不应被理解为 magic number，
而应被理解为一个更一般规律的离散近似。

建议损失由三部分构成：

- `L_ce`
- `L_rank`
- `L_cost`

其中 `L_rank` 建议显式比较：

- 好候选 vs baseline
- 坏候选 vs baseline

也就是让模型学的是“相对排序”，而不是单纯类别预测。


### 11.8 选择性与可靠性层的定位

不建议现在把主线切成 conformal / selective theory paper。

更合理的顺序是：

1. 先做 recovery module
2. 再做 calibration
3. 再做 selective override
4. 最后才考虑 conformal guarantee

原因：

- 当前最缺的是恢复能力，不是理论包装
- 没有一个够强的 base module，理论层不会变成主贡献

另外还有一个更具体的问题：

- conformal prediction 通常依赖较强的 exchangeability 假设
- 而 recovery events 在同一视频内部往往高度相关

因此当前更稳妥的做法仍然是：

- bootstrap CI
- grouped / sequence-level statistical testing
- coverage-risk curve


### 11.9 Off-screen 的处理原则

建议明确拆成两类问题：

- `local recovery`
- `global re-detection`

当前论文只主打：

- `in-frame`
- 尤其是 `100 <= occ_length < 500`

off-screen 不要强行和当前局部 selector 绑在一起。


### 11.10 停机条件

建议分两级设置。

#### 黄色预警

如果满足任意条件，转为 benchmark / analysis 论文：

- 多序列上 learned module 改善 `< 3 px`
- 统计检验 `p > 0.1`
- 超过一半序列上无效

#### 红色预警

如果满足任意条件，停止继续堆方法：

- 多序列上 learned module 改善 `< 1 px`
- 稳定无法超过 hand-crafted hybrid
- 候选池 oracle gap 已经很小


### 11.11 当前推荐执行顺序

1. 修复并完成真实多序列构建
2. 新增数据审计脚本
3. 在真多序列上重跑 baseline / hybrid / adaptive / stratified
4. 用 sequence-grouped 方式重新做统计检验
5. 实现 Step 1 最简 learnable recovery module
6. 若 Step 1 成立，再做 Step 2 完整版
7. 再补 `TAP-Vid-DAVIS`
8. 最后再考虑 calibration / selective / conformal
