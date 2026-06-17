# Online Recovery Module Design (2026-06-09)

## 1. Decision

如果目标从 `oracle-defined post-hoc selector` 切换到 `真实在线恢复模块`，当前最合理的开发基线是：

- `CoTracker3 online` 作为主开发 baseline

同时保留以下对照方法作为论文 comparison baselines：

- `CoTracker3 online`
- `Track-On2`
- `TAPNext++`
- `ReTracker`（如果代码和算力允许）

当前不建议把 `Track-On2` 作为第一开发基线。


## 2. Why CoTracker3 Online

### 2.1 工程理由

仓库里已经有现成接入点：

- [models/cotracker_refiner.py](/gemini/code/FSPT/models/cotracker_refiner.py)
- [scripts/eval_cotracker3_prt.py](/gemini/code/FSPT/scripts/eval_cotracker3_prt.py)

其中 [models/cotracker_refiner.py](/gemini/code/FSPT/models/cotracker_refiner.py) 里已经存在以下在线恢复雏形：

- `relocalization`
- `template_mode = visible_bank`
- `relocal_acceptor / verifier`
- `relocalization_residual`
- `retracking splice`

这意味着我们不是从零设计一个新系统，而是把现有启发式模块收敛成一条真正可训练的在线恢复路径。

### 2.2 方法理由

`CoTracker3` 的优势不是“它在论文里最适合 recovery”，而是：

1. 强 baseline 足够有说服力
2. 有明确 online 接口
3. 中间特征、局部相关、visibility/confidence 都能取到
4. 当前仓库已经深度依赖它

### 2.3 为什么不把 Track-On2 作为第一开发基线

`Track-On2` 的 memory 机制在概念上更接近 recovery，但它不适合作为第一开发基线，原因是：

1. 你当前仓库只有评测封装，没有完整内嵌改造路径
2. 它本身 memory 更强，改造失败时难以定位问题是在 recovery module 还是在原始架构耦合
3. 如果 recovery 成功，也更难证明收益来自你的模块，而不是原模型 memory 已经很强

因此：

- `Track-On2` 更适合作为外部 comparison baseline
- `CoTracker3 online` 更适合作为第一开发和方法实现基线


## 3. Relevant Recent Work

需要对齐的最新公开工作：

- `TAPNext++`
  - 价值：明确把 `re-detection / re-entry` 提为核心盲点，并提供 `AJ_RD`
  - 来源：
    - https://arxiv.org/abs/2604.10582
    - https://tap-next-plus-plus.github.io/

- `Track-On2`
  - 价值：在线 memory tracking 架构参考
  - 来源：
    - https://arxiv.org/abs/2509.19115
    - https://kuis-ai.github.io/track_on2/
    - https://github.com/gorkaydemir/track_on

- `CoTracker`
  - 价值：当前主开发基线
  - 来源：
    - https://arxiv.org/abs/2410.11831
    - https://github.com/facebookresearch/co-tracker

- `ReTracker`
  - 价值：提供 global matching / rematching 思路，适合 recovery search 分支借鉴
  - 来源：
    - https://openaccess.thecvf.com/content/ICCV2025/papers/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.pdf

- `Real-World Point Tracking with Verifier-Guided Pseudo-Labeling`
  - 价值：说明 verifier 不是 tracker 本体，而是候选质量判断器
  - 来源：
    - https://arxiv.org/abs/2603.12217

- `CoWTracker`
  - 价值：2026 年更新的强 tracking baseline，说明 point tracking 仍在快速演化
  - 来源：
    - https://openaccess.thecvf.com/content/CVPR2026/html/Lai_CoWTracker_Tracking_by_Warping_instead_of_Correlation_CVPR_2026_paper.html


## 4. Core Shift From PRT To Online Recovery

当前 PRT 线的根本问题是：

- 先由 GT 抽取 recovery event
- 再围绕该 event 构建 support frames 和 candidate pool

这不是在线恢复。

真正的在线恢复必须满足：

1. 恢复触发由 tracker 自身状态决定，不由 GT 事件决定
2. support memory 在线维护，而不是 GT 回溯抽取
3. 候选搜索在真实推理流程中触发
4. 输出直接写回 tracker 轨迹

一句话：

- `PRT = event-conditioned post-hoc selection`
- `online recovery = tracker-conditioned causal intervention`


## 4.1 Stage 0 Gate

在实现任何新的 online recovery 模块之前，先做 `Stage 0`：

- 输入：`CoTracker3 fnet` 特征图 + GT 定义的 recovery events
- 目标：判断 `CoTracker3` 自身特征是否已经包含足够的 `support-conditioned re-entry signal`
- 脚本：
  - [scripts/diagnose_cotracker_feature_recovery_signal.py](/gemini/code/FSPT/scripts/diagnose_cotracker_feature_recovery_signal.py)

判据事先固定，不允许事后改口径：

- `gt_beats_bg_max_frac >= 0.65`
- `gt_minus_bg_max_mean >= 0.5 * std(gt_minus_bg_max)`
- `rank1_frac >= 0.30`

其中：

- `gt_beats_bg_max_frac` 只表示 GT 相似度是否超过随机背景采样中的最强干扰点
- `rank1_frac` 必须按当前帧 `dense similarity map` 的全图 `argmax` 定义，不能用随机背景近似替代

决策规则：

- 三条都通过：继续走 `CoTracker3-feature-only SCORE`
- 只通过部分：标记为 `warning`，允许继续做小规模 `M0`，但必须准备更强 feature fallback
- 三条都不通过：停止 `CoTracker3-feature-only` 路线，切到 `DINOv2 recovery branch`

额外要求：

- 必须按 occlusion length 分桶报告：
  - `20-99`
  - `100-199`
  - `200-499`
  - `500+`
- 如果只有短遮挡桶有效，那么后续方法必须诚实限定 effective range，而不是宣称通用 recovery


## 5. Recommended Architecture

建议将新模块命名为：

- `SCORE`
- `Support-Conditioned Online Recovery`

其目标不是替代 tracker，而是在长遮挡后第一帧重出现时，作为一个轻量恢复适配器插入。

### 5.1 Overall Pipeline

每帧处理流程：

1. `base tracker`
   - 用 `CoTracker3 online` 正常输出：
     - `tracker_pos`
     - `tracker_vis`
     - `tracker_conf`
     - `feature_map`
     - `per-point feature`

2. `recovery trigger`
   - 判断当前点是否需要启动恢复搜索

3. `support memory bank`
   - 从该点历史高置信可见帧中提取 memory entries

4. `support-conditioned search`
   - 用 memory bank 在当前帧 feature map 上做全局或半全局搜索

5. `verifier + residual refiner`
   - 对 `base` 与 `recovered candidate` 做比较
   - 预测是否覆盖、覆盖到哪里、置信度多高

6. `tail retracking`
   - 若本帧接受恢复结果，则从该点重新初始化 base tracker，拼接后续轨迹


## 6. Module Breakdown

### 6.1 Base Tracker

第一版不改 `CoTracker3` 主体，只外挂 recovery adapter。

好处：

- 基线清晰
- 调试边界清晰
- 如果失败，可以明确归因于 recovery adapter，而不是 base tracker 被破坏

### 6.2 Support Memory Bank

每个点维护一个固定大小的 memory bank：

- `frame_idx`
- `coord`
- `track_feat`
- `optional pooled patch feat`
- `visibility/confidence at write time`

更新规则：

- 只有在 `tracker_vis` 和 `tracker_conf` 都高时写入
- 遮挡阶段冻结 bank
- 使用 FIFO 或 reservoir 维持固定容量
- 需要 novelty check，避免 bank 全是连续重复帧

第一版容量建议：

- `M = 8`

### 6.3 Recovery Trigger

触发必须先用简单规则，不要一开始就学。

第一版规则：

1. 当前帧 `tracker_vis` 从低变高
2. 过去一段时间有连续低可见段
3. memory bank 非空

输入只用 tracker 自身状态：

- `tracker_vis`
- `tracker_conf`
- `corr_conf`
- `occlusion_streak`
- `motion_jump`

第一版可以做成 hysteresis state machine，而不是神经网络。

原因：

- 先验证 search + verifier 是否有效
- 不把失败归因混到 trigger 上

### 6.4 Support-Conditioned Search

搜索不能再只围绕当前 base 预测做极小局部邻域修补。

建议做：

- 以 `tracker_pos` 为中心的大窗口搜索，作为第一版
- 后续扩展到全局 coarse-to-fine 搜索

输入：

- `support bank features`
- `current frame feature map`
- `current tracker prediction`

融合策略：

- 多 memory entry 对当前位置做相似度
- 逐位置取 `max` 或 learnable weighted max

第一版推荐：

- `max over support entries`

原因：

- 一个点在遮挡前外观是变化的
- 重现时只要有一个历史 template 匹配即可

### 6.5 Similarity Refiner

这是整个方案里最值得学习的部分。

不建议继续做：

- `baseline / cand1..K` 离散分类

原因：

- 这类目标已经在 PRT 线里表现出容易塌缩为“总选 baseline”

建议改成：

- 从密集 similarity map 直接预测连续位置

第一版结构：

- 输入：
  - `sim_map`
  - `current feature patch`
  - `optional base position prior`
- 网络：
  - 小型 ConvNet / U-Net style head
- 输出：
  - `refined_sim_map`
  - `soft-argmax position`

这会把任务从“离散候选选择”改成“连续重定位”。

这是和当前 PRT learned ranker 最大的结构性差异。

### 6.6 Verifier / Confidence Head

`Similarity Refiner` 给位置，`Verifier` 决定是否替换。

输入建议包含：

- refined heatmap peak
- entropy
- top1-top2 margin
- distance to base prediction
- base visibility/confidence
- support consensus score
- occlusion length

输出：

- `override confidence`

决策：

- 若 `override_conf > threshold`
  - 使用 recovered position
- 否则
  - 保持 tracker 原始结果

### 6.7 Tail Retracking

如果当前帧接受 recovered position，只修这一帧通常不够。

因此建议复用当前已有逻辑：

- 从 accepted re-entry frame 重新初始化 base tracker
- splice 后续轨迹

对应代码入口已存在于：

- [models/cotracker_refiner.py](/gemini/code/FSPT/models/cotracker_refiner.py)


## 7. What To Reuse From Existing Repo

优先复用以下已有实现，而不是重写：

1. `relocalization`
   - 复用全局/半全局匹配 step

2. `template_mode = visible_bank`
   - 作为 online memory bank 的起点

3. `relocal_acceptor / verifier`
   - 但要替换成真正的 memory-conditioned confidence head

4. `relocalization_residual`
   - 保留 residual refinement 思路

5. `retracking`
   - 作为 accept 后的 tail recovery

当前不建议继续保留的大量启发式：

- 多层 gate power / shift threshold / margin threshold 的手工组合
- 复杂但不可解释的局部开关堆叠

需要把这条线收敛到：

- `trigger`
- `search`
- `verify`
- `retrack`


## 8. Training Strategy

### 8.1 Data Construction

训练时允许用 GT 做 supervision，但不能让 GT 决定推理输入。

正确数据构建方式：

1. 在训练视频上跑完整 `CoTracker3 online rollout`
2. 在线记录：
   - `feature_map`
   - `track_feat`
   - `tracker_pos`
   - `tracker_vis`
   - `tracker_conf`
   - memory bank 状态
3. 再用 GT 仅生成标签：
   - `trigger label`
   - `target position`
   - `better-than-baseline label`
   - `success < 4px label`

### 8.2 Stage-Wise Training

推荐三阶段：

#### Stage A

只验证 `search + refiner` 是否有效。

- trigger 暂时用 GT event 或 GT visibility teacher
- 目的是验证：如果时机给对了，恢复分支能否真的找回位置

#### Stage B

切换到 tracker-driven trigger。

- 用 tracker 自己的 visibility/confidence 启动恢复
- 评估真正在线条件下是否仍有提升

#### Stage C

训练 learned trigger。

- 只有在 A/B 有效时才值得做


## 9. Loss Recommendation

第一版建议 4 项损失：

1. `L_heatmap`
   - GT 位置热图监督

2. `L_pos`
   - `soft-argmax` 输出位置到 GT 的回归损失

3. `L_conf`
   - 预测本次恢复是否成功（例如 `<4px`）

4. `L_override`
   - 若 recovery 优于 base，则鼓励 override
   - 若 recovery 不优于 base，则惩罚 override

不建议第一版直接退回：

- 多类 CE over `{baseline, cand1..K}`

原因：

- 这条线在当前 PRT 里已经暴露出塌缩风险


## 10. Evaluation Protocol

主评测不应该再只看 overall AJ。

建议分三层：

### 10.1 Recovery-Specific

在 re-entry frame 上评测：

- `reentry_median_px`
- `better_frac_vs_base`
- `AJ_RD`（若能对齐 TAPNext++）

### 10.2 Sequence-Level Statistics

必须做：

- grouped bootstrap by sequence
- sequence-level paired test

不要只做 event-level pooled bootstrap。

### 10.3 Overall Safety

同时报告：

- overall AJ
- overall OA
- 是否退化

论文叙事应该是：

- first-frame recovery improves
- overall does not degrade materially


## 11. Risks

### Risk 1

`CoTracker3` 的 128-d feature 不够判别。

应对：

- 从 center feature 扩展到 pooled local patch feature
- 再不够，才考虑引入更强外观 backbone

### Risk 2

tracker visibility 不可靠，trigger 触发不到位。

应对：

- 先做 teacher-trigger stage
- 再切 tracker-trigger

### Risk 3

即使 online recovery 有改善，幅度也很小。

应对：

- 重点报告 recovery-specific metrics
- 不强行用 overall AJ 讲故事

### Risk 4

learned verifier 仍然偏向保守，不敢 override。

应对：

- 使用连续定位监督，而不是纯离散候选分类
- 加入 success-aware confidence supervision


## 12. Final Recommendation

当前最建议的路线是：

1. 基于 `CoTracker3 online` 开发
2. 在现有 [models/cotracker_refiner.py](/gemini/code/FSPT/models/cotracker_refiner.py) 上收敛成单一在线 recovery 路径
3. 第一版只做：
   - rule-based trigger
   - online visible memory bank
   - support-conditioned dense search
   - heatmap refiner
   - verifier gate
   - optional retracking splice
4. `Track-On2` 只作为 comparison baseline，不作为第一开发基线

一句话：

- `CoTracker3 online + SCORE-style recovery adapter` 是当前最现实、最可落地的在线恢复主线
