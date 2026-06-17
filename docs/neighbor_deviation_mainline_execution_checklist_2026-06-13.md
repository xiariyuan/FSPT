# Neighbor-Deviation Mainline Execution Checklist (2026-06-13)

## 0. 目标

把主线正式收口到：

**neighbor-deviation reliability + calibrated selective point tracking + pseudo-label filtering preparation**

这份清单的目标不是继续发明 recovery 方法，而是把现有真实信号做成一条能投稿的、口径一致的论文线。

## 1. 立即冻结的范围

### 1.1 永久停止

- 停止所有新的 `online recovery` 方法发散。
- 停止所有新的 `ranker / reranker / candidate selector` 变体。
- 停止 `DiReCT-lite`、`cross-attention recovery`、`dense heatmap recovery`。
- 停止 `geometry rerank`、`patch verifier v2`、`new DINO fusion trick`。
- 停止再把 `Stage-2c` 当成需要继续突破的方法主线。

### 1.2 保留但降级

- [local_selector_stage2c_threshold_sweep_status_2026-06-12.md](/gemini/code/FSPT/docs/local_selector_stage2c_threshold_sweep_status_2026-06-12.md)
  作用：failure analysis / historical baseline
- [local_geometry_rerank_status_2026-06-12.md](/gemini/code/FSPT/docs/local_geometry_rerank_status_2026-06-12.md)
  作用：negative result evidence
- [direct_recovery_attention_audit_v1_1_status_2026-06-12.md](/gemini/code/FSPT/docs/direct_recovery_attention_audit_v1_1_status_2026-06-12.md)
  作用：negative result evidence

## 2. 当前唯一活跃主线

### 2.1 主结论

- `neighbor deviation` 是当前最强的 trajectory-only reliability signal。
- 该信号在跨序列 LOOCV 下稳定。
- 它对 `long_occ` 子集有更明确的 selective ranking 价值。
- calibration 的价值是把它转成风险概率，而不是改善排序。

### 2.2 对应核心工件

- [trajectory_manifold_verifier_v1_status_2026-06-12.md](/gemini/code/FSPT/docs/trajectory_manifold_verifier_v1_status_2026-06-12.md)
- [trajectory_manifold_verifier_v1_loocv_status_2026-06-12.md](/gemini/code/FSPT/docs/trajectory_manifold_verifier_v1_loocv_status_2026-06-12.md)
- [neighbor_deviation_selective_v1_1_status_2026-06-13.md](/gemini/code/FSPT/docs/neighbor_deviation_selective_v1_1_status_2026-06-13.md)

## 3. 第一优先级任务：统一结果口径

### 3.1 文档清理

- 更新所有主计划文档中的 selective tracking 结论，统一以 `v1_1` 为准。
- 明确写出：
  - selective ranking 由原始 `neighbor deviation` 提供
  - calibration 提供 risk probability，不提供排序增益
- 在主文档里标注 `v1` selective 结果存在方向性错误，不再引用。

### 3.2 需要更新的文档

- [RESEARCH_PLAN.md](/gemini/code/FSPT/docs/RESEARCH_PLAN.md)
- [reliable_selective_tracking_plan.md](/gemini/code/FSPT/docs/reliable_selective_tracking_plan.md)
- [neighbor_deviation_selective_v1_status_2026-06-13.md](/gemini/code/FSPT/docs/neighbor_deviation_selective_v1_status_2026-06-13.md)

### 3.3 成功标准

- 所有主文档只引用 `v1_1` selective 结论。
- 不再出现“Platt 带来 selective AJ 增益”的表述。

## 4. 第二优先级任务：做严格的外部校准评估

当前 `v1_1` calibration 仍然是同集拟合/同集评估。论文里这还不够。

### 4.1 要做什么

- 做 `grouped leave-one-video-out calibration`：
  - train videos 拟合 calibrator
  - held-out video 上评估 ECE / Brier
- 至少比较：
  - raw-risk
  - Platt-risk
  - isotonic-risk
  - visibility baseline

### 4.2 实现建议

- 新脚本：
  - `scripts/eval_neighbor_deviation_grouped_calibration.py`
- 输入：
  - [per_sample_predictions.jsonl](/gemini/code/FSPT/outputs/trajectory_manifold_verifier_v1/per_sample_predictions.jsonl)
- 输出：
  - `outputs/neighbor_deviation_grouped_calibration_v1/`
  - `docs/neighbor_deviation_grouped_calibration_v1_status_2026-06-13.md`

### 4.3 必报指标

- pooled ECE
- grouped-LOOV ECE
- pooled Brier
- grouped-LOOV Brier
- per-video calibration table

### 4.4 成功标准

- `neighbor deviation` 在 held-out video 上依然保持合理 ECE / Brier。
- 即使 Platt 不能继续提升，也要明确 raw-risk 已经够强。

### 4.5 失败解释

- 如果外部 calibration 崩掉，不影响 reliability 主结论。
- 那时就把 calibration 降级成“potential extension”，主论文收回到 ranking + risk ordering。

## 5. 第三优先级任务：规范化 selective evaluation 协议

### 5.1 当前问题

现在的 `selective_metrics_overall.json` 已经能看排序能力，但还不够像论文标准协议。

### 5.2 要补的内容

- 明确区分：
  - `risk score`
  - `reliability score`
  - `selection threshold`
- 给出 coverage-driven threshold table：
  - coverage = 0.1 / 0.3 / 0.5 / 0.7 / 0.9
  - 对应 threshold
  - 对应 mean error / median error / high-error fraction
- 给出 fixed-risk 视角：
  - 例如 `predicted risk <= 0.05 / 0.10 / 0.20`
  - 对应 coverage 和实际 high-error rate

### 5.3 实现建议

- 在 [eval_neighbor_deviation_selective.py](/gemini/code/FSPT/scripts/eval_neighbor_deviation_selective.py) 基础上扩展，不重写。
- 增加两个输出：
  - `threshold_sweep_by_coverage.json`
  - `threshold_sweep_by_predicted_risk.json`

### 5.4 成功标准

- selective protocol 可以直接变成论文正文的实验节。
- 后续做 conformal / abstain 时不需要重新组织数据格式。

## 6. 第四优先级任务：长遮挡主场景分析

### 6.1 当前已知

- overall 上 `neighbor deviation` 与 `visibility` 差距不算特别大
- `long_occ` 子集上差距更有叙事价值

### 6.2 要做什么

- 把 `long_occ` 作为主分析子集单独写厚：
  - AUC
  - Spearman
  - coverage sweep
  - top-10% / top-20% coverage qualitative cases
- 分析哪些视频贡献了主要信号，哪些视频是失败样例

### 6.3 输出

- `outputs/neighbor_deviation_longocc_analysis_v1/`
- `docs/neighbor_deviation_longocc_analysis_v1_status_2026-06-13.md`

### 6.4 成功标准

- 能明确展示：
  - 为什么 `neighbor deviation` 在 `long_occ` 更像“异常轨迹检测器”
  - 为什么 `visibility` 在这个子集不够可靠

## 7. 第五优先级任务：失败案例与边界条件

这部分必须做，不然论文会显得像只报好结果。

### 7.1 要分析的失败类型

- smooth but wrong
- low deviation but high error
- high deviation but GT 仍然较准
- 邻居轨迹本身整体漂移

### 7.2 需要输出

- 每类失败案例至少 5 个
- 每个案例包含：
  - video name
  - point id
  - occ length
  - mean error
  - visibility
  - neighbor deviation

### 7.3 输出位置

- `outputs/neighbor_deviation_failure_cases_v1/`
- `docs/neighbor_deviation_failure_cases_v1_status_2026-06-13.md`

### 7.4 成功标准

- 论文 discussion 能明确写出 signal 的边界，不夸大。

## 8. 第六优先级任务：learned head 的正确定位

### 8.1 不要做的事

- 不要把 `learned head` 改写成新的主方法。
- 不要为了追一点 AUC 再开一轮模型调参。

### 8.2 正确做法

- 只把它作为 appendix / ablation：
  - raw neighbor deviation
  - learned head
  - 两者泛化差距
  - fold 波动

### 8.3 需要补的表

- per-fold AUC variance
- worst fold / best fold 对比
- 与 hand-crafted 方法的鲁棒性比较

### 8.4 成功标准

- 结论清楚：
  - learned head 有潜在增益
  - 但 hand-crafted signal 更稳定，因此主论文用 hand-crafted

## 9. 第七优先级任务：pseudo-label filtering 预备实验

这是第二阶段，不是立刻主打，但现在就该把接口准备好。

### 9.1 要做什么

- 用 `neighbor deviation` 或 calibrated risk 做伪标签过滤阈值
- 先不做大训练，只做数据筛选统计

### 9.2 最小实验

- 在已有 rollout / pseudo-track 数据上统计：
  - 过滤前样本数
  - 过滤后样本数
  - 平均误差变化
  - 高误差比例变化
- 比较：
  - visibility threshold
  - neighbor deviation threshold
  - calibrated risk threshold

### 9.3 输出

- `outputs/neighbor_deviation_pseudolabel_filtering_v1/`
- `docs/neighbor_deviation_pseudolabel_filtering_v1_status_2026-06-13.md`

### 9.4 成功标准

- 能证明 `neighbor deviation` 不只是测试时可用，训练时也能更稳地筛数据。

## 10. 第八优先级任务：论文骨架收口

### 10.1 标题方向

不要再写 recovery / relocalization。

建议围绕：

- reliability
- calibrated risk
- selective point tracking
- long-occlusion robustness

### 10.2 摘要必须包含的 4 点

1. 强 tracker 在 easy case 饱和，但在 long-occ 和域偏移下仍会输出高风险错误。
2. 提出 `neighbor deviation` 作为 trajectory-local reliability signal。
3. 证明该信号跨序列泛化，并支持 selective tracking / risk control。
4. 该信号可进一步用于 pseudo-label filtering。

### 10.3 论文实验结构

1. 主结果：reliability detection
2. 泛化：LOOCV / held-out calibration
3. selective tracking
4. long-occ 专项分析
5. failure cases
6. pseudo-label filtering 预备实验

## 11. 执行顺序

建议严格按下面顺序执行，不要跳：

1. 修正文档口径，统一切到 `v1_1`
2. 做 grouped calibration
3. 补 selective threshold protocol
4. 做 long-occ 深挖
5. 做 failure cases
6. 写 learned head appendix 表
7. 做 pseudo-label filtering 预备统计
8. 收论文骨架

## 12. 每一步的停止条件

### 12.1 grouped calibration

- 如果 held-out calibration 崩掉：
  - calibration 降级
  - reliability 主线保留

### 12.2 selective protocol

- 如果新的阈值协议没有额外信息：
  - 保留当前 coverage sweep 即可
  - 不继续打磨 fancy 指标

### 12.3 pseudo-label filtering

- 如果只带来极小统计差异：
  - 作为 future work
  - 不拖慢主线收口

## 13. 当前最重要的判断

现在最该做的不是“再找一个 recovery idea”，而是把下面这条线做扎实：

**neighbor deviation 是一个可泛化的 trajectory reliability / risk signal；它支持 selective point tracking，并有潜力进一步指导 pseudo-label filtering。**

只要这条线的 calibration、subset、failure boundary 和 protocol 收干净，论文主线就成立。
