# 新会话交接文档：FSPT / V9-A2 Dynamic Horizon

更新日期：2026-07-10

> 新会话必须先完整阅读本文，再读取本文列出的关键结果文档。不要从 V9-A2.1 重新开始，不要重复已经完成的 smoke、oracle、feature build、controller 和 robust audit。

---

## 1. 新会话第一条指令建议

把下面这句话直接发给新会话：

```text
请先完整阅读服务器文件：
/gemini/code/FSPT/docs/NEW_CHAT_HANDOFF_V9A2_2026-07-10.md

然后读取文档中标记为“必须阅读”的结果文件和 CURRENT_MAINLINE.md，检查当前 Git 分支和 HEAD。不要重复已完成实验，从文档规定的下一步 V9-A2.4 开始，执行过程中每一步都认真复核。
```

---

## 2. 仓库与 Git 当前状态

项目目录：

```text
/gemini/code/FSPT
```

当前分支：

```text
mainline-pivot-foundation-20260626
```

当前 HEAD：

```text
368fa7e Fix V9-A2 robust threshold and video stability audits
```

当前本地分支与远端同步：

```text
HEAD = origin/mainline-pivot-foundation-20260626 = 368fa7e
```

最近相关提交：

```text
368fa7e Fix V9-A2 robust threshold and video stability audits
ef242fa Add reentry viscalibrator model and MCP service scripts
9d84db0 Add TAPNext++ evaluation, ReEntry cross-baseline comparison, and V9-A2 controller experiments
fa22457 Add V9-A2 dynamic horizon controller experiments
```

截至交接时，tracked worktree 是干净的。

注意：大型 `.npz/.pt/.pkl` 特征和缓存文件保存在服务器本地，不应随意提交 Git；代码、文档和小型 JSON 报告已同步到当前远端分支。

---

## 3. 研究主问题

研究目标：提高在线点追踪器在长时间遮挡后的 re-entry / reacquisition 能力，同时控制错误重捕获、身份漂移和可见性误报。

当前基座：

```text
CoTracker3 online native tracker
+
TrackOn2 online candidate bridge
```

当前论文级方向不是继续堆规则，而是：

```text
ReEntryBeliefTrack / BeliefTrack
```

核心概念：

```text
Track / Coast / Reacquire phase
uncertainty-aware belief propagation
identity-aware reacquisition
multi-hypothesis recovery
safe memory write
calibrated dynamic recovery horizon
```

当前 V9-A2 是这个大方向的轻量验证：

```text
保留稳定的 W8 recovery action，
学习何时把 recovery horizon 扩展到 W16。
```

---

## 4. 冻结的基础结果

### 4.1 Native CoTracker3 online

缓存：

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt
```

Native metrics：

```text
AJ        65.2366
OA        90.8186
delta_avg 77.9458
delta_4px 85.8670
AJ_RD     0.3534
AJ_RD_256 0.5333
```

### 4.2 CVRRM W8 主基线

配置：

```text
TrackOn2 bridge
dist_nc <= 64
W = 8
candidate-visible frame-level recovery
output-level writeback
```

相对 Native：

```text
AJ        +0.0964
OA        +0.9462
delta_avg +0.3971
delta_4px +0.5085
AJ_RD     +0.0153
AJ_RD_256 +0.0274
positive/negative/zero videos = 16/4/5
```

### 4.3 CVRRM W16

相对 Native：

```text
AJ        +0.0810
OA        +0.9701
delta_avg +0.4339
delta_4px +0.5593
AJ_RD     +0.0162
AJ_RD_256 +0.0286
positive/negative/zero videos = 15/5/5
```

解释：

```text
W8 更稳；
W16 re-entry 指标略高，但负视频更多。
```

### 4.4 评估口径

允许表述：

```text
TAP-Vid-DAVIS-first metric-compatible reproduction
```

不允许表述：

```text
超过 CoTracker3 原论文完整 Table 1
```

---

## 5. V9-A1 已完成，不要重做

V9-A1 只在 W8 touched rows 上做 learned accept/reject controller。

关键结果：

```text
learned gate 可以提高 AJ/OA、减少风险，
但一旦过滤 W8，AJ_RD_256 会低于 CVRRM W8 accept-all。
```

结论：

```text
W8-only filtering 的 action space 被 CVRRM accept-all 上限锁住。
不能靠继续调 threshold 解决。
```

结果文档：

```text
docs/v9a1_controller_calibration_aware_prototype_result_2026-07-08.md
```

---

## 6. V9-A2 已完成的阶段

### 6.1 V9-A2.1：action-space audit

W8/W16 row 数：

```text
W8 rows        = 1456
W16 rows       = 2013
Common rows    = 1456
W16-only rows  = 557
```

W16-only：

```text
candidate_good = 118 / 557
candidate_good_mean = 0.2118
damage16_mean = 0.0018
```

结论：

```text
W16 extension 确实包含新的 recovery opportunity；
V9-A2 必须做 W8 + W16 dynamic horizon，不能再做 W8-only filter。
```

结果文档：

```text
docs/v9a2_action_space_audit_result_2026-07-08.md
```

### 6.2 V9-A2.2：oracle upper-bound

相对 Native：

```text
W8 accept-all AJ_RD_256 Δ                 +0.0274
W16 accept-all AJ_RD_256 Δ                +0.0286
W8 + W16-only candidate_good oracle       +0.0321
Event-level utility-positive W16 oracle   +0.0320
```

Oracle 的 Pos/Neg/Zero：

```text
17/3/5
```

结论：

```text
selective W16 extension 有真实 trajectory-level 上限；
event-level dynamic horizon 值得继续。
```

结果文档：

```text
docs/v9a2_action_space_oracle_audit_result_2026-07-08.md
```

### 6.3 V9-A2.3a：feature builder 修正

已完成：

```text
坐标不再预先 clip；
显式 oob features；
anchor reliability features；
label overlap audit；
strict pre-occlusion anchor。
```

重要标签结论：

```text
candidate_good == utility_positive
Jaccard = 1.0
```

所以它们不能当两个独立 supervision target。

主目标应使用：

```text
candidate_good
```

辅助风险目标：

```text
good_not_worse
candidate_bad
candidate_worse_px
false_visible
```

结果文档：

```text
docs/v9a2_feature_builder_correction_result_2026-07-08.md
docs/v9a2_preocc_anchor_strict_fix_result_2026-07-08.md
```

### 6.4 V9-A2.3b：full feature build

Full features：

```text
base dim   = 28
anchor dim = 115
all dim    = 143
```

W8：

```text
1456 rows
finite_rate = 1.0
```

W16：

```text
2013 rows
finite_rate = 1.0
```

重要发现：同一个 common row 在 W8/W16 中可能绑定不同 `first_event_t / max_event_age / touch_source_count`，因此 W16 common rows 不能直接替代 W8 canonical context。

最终 canonical joint dataset：

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz
```

构成：

```text
W8 canonical common rows = 1456
W16-only extension rows   = 557
Total                     = 2013
Feature dim               = 143
```

结果文档：

```text
docs/v9a2_full_feature_build_result_2026-07-08.md
```

### 6.5 V9-A2.3c：dynamic horizon controller

默认策略：

```text
Preserve all W8 common rows.
Only learn which W16-extension events/rows to add.
```

训练协议：

```text
video-group-heldout OOF
feature sets: base / anchor / all
models: logreg / hgb / extratrees
modes: frame / event_max / event_mean
```

OOF 分类本身不强：

```text
best row-level OOF approximately:
base / extratrees AP 0.3233, AUC 0.6420
```

RGB patch anchor 没有形成强 ranking signal：

```text
anchor / extratrees AUC 0.5773
all / extratrees AUC 0.5855
```

但 event-level apply-back 有正向结果。

结果文档：

```text
docs/v9a2_dynamic_horizon_controller_result_2026-07-08.md
```

---

## 7. 当前最重要的最新结果：V9-A2.3c-R

### 7.1 Robust threshold audit

该审计使用预声明、紧凑阈值，不依赖 dense trajectory best sweep：

```text
event_max only
fixed thresholds / OOF-F1
W8 common rows always preserved
only W16 extension controlled
```

主 learned policy：

```text
all_logreg + event_max + fixed threshold 0.05
```

相对 Native：

```text
AJ Δ        +0.1025
OA Δ        +0.9711
AJ_RD_256 Δ +0.0294
Accepted total = 1937
Pos/Neg/Zero = 17/3/5
```

接受的 W16 extension 质量：

```text
accepted extension = 481 / 557
good               = 109
bad                = 141
false-visible      = 61
candidate-worse    = 190
```

对比：

```text
W8 AJ_RD_256 Δ   +0.0274, Pos/Neg/Zero 16/4/5
W16 AJ_RD_256 Δ  +0.0286, Pos/Neg/Zero 15/5/5
Learned fixed0.05 +0.0294, Pos/Neg/Zero 17/3/5
Oracle             +0.0321, Pos/Neg/Zero 17/3/5
```

另一个无需 fixed threshold 微调的结果：

```text
all_logreg event_max OOF-F1:
AJ_RD_256 Δ +0.0293
AJ Δ +0.0975
OA Δ +0.9746
Pos/Neg/Zero 17/3/5
```

结果文档：

```text
docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-09.md
```

JSON：

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_robust_threshold_audit.json
```

### 7.2 Video stability audit

Learned fixed 0.05 与 W16 比较：

```text
AJ_RD_256 defined videos = 25
better = 4
worse  = 2
equal  = 19
mean learned-minus-W16 = +0.000665
```

主要提升视频：

```text
drift-straight  +0.010417
bike-packing    +0.004325
dance-twirl     +0.002278
car-shadow      +0.001873
```

退化视频：

```text
bmx-trees  -0.001871
parkour    -0.000389
```

结果文档：

```text
docs/v9a2_dynamic_horizon_video_stability_audit_result_2026-07-09.md
```

JSON：

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_video_stability_audit.json
```

---

## 8. 当前科学结论

### 已经可以支持的结论

```text
1. W8-only learned filtering 不够；扩展 action space 是必要的。
2. W16-only rows 中存在真实 recovery opportunity。
3. event-level dynamic horizon 可以在 conservative threshold 下超过 W16 accept-all。
4. learned fixed0.05 同时改善 AJ、AJ_RD_256 和正负视频结构。
5. 当前结果不是只由 dense trajectory threshold sweep 得到；robust threshold audit 已通过。
```

### 不能夸大的部分

```text
1. 提升幅度真实但较小：相对 W16 的 AJ_RD_256 只增加约 +0.0008。
2. 视频级提升集中：25 个 defined videos 中 4 better、2 worse、19 equal。
3. RGB patch anchor 特征本身较弱，不能宣称已经解决 identity-aware reacquisition。
4. 当前模型仍依赖外部 TrackOn2 candidate provider。
5. 不能宣称超过原始 CoTracker3 完整多数据集 Table 1。
```

### 最重要反思

```text
当前成功主要来自“正确的 action space + event-level horizon policy”，
而不是强大的 RGB identity representation。
```

因此后续有两个方向：

```text
A. 先把 V9-A2 做成 paper-ready ablation / prototype；
B. 再升级 semantic/internal identity feature，缩小 oracle gap。
```

---

## 9. 必须阅读的文件

新会话开始后，按顺序读取：

```text
1. CURRENT_MAINLINE.md
2. docs/v9a2_anchor_uncertainty_reacquisition_design_2026-07-08.md
3. docs/v9a2_action_space_oracle_audit_result_2026-07-08.md
4. docs/v9a2_full_feature_build_result_2026-07-08.md
5. docs/v9a2_dynamic_horizon_controller_result_2026-07-08.md
6. docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-09.md
7. docs/v9a2_dynamic_horizon_video_stability_audit_result_2026-07-09.md
```

核心脚本：

```text
scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py
scripts/v9a2_action_space_oracle_audit.py
scripts/v9a2_build_joint_w8_w16_dataset.py
scripts/v9a2_dynamic_horizon_controller.py
scripts/v9a2_dynamic_horizon_robust_audit.py
scripts/v9a2_dynamic_horizon_video_stability_audit.py
```

---

## 10. 下一步：V9-A2.4 Paper-Ready Ablation（推荐立即执行）

不要再做大范围 threshold tuning。

### 10.1 冻结主 learned policy

主 policy 固定为：

```text
all features
logistic regression
event_max
fixed threshold = 0.05
preserve W8 common rows
control W16 extensions only
```

OOF-F1 policy 作为无人工固定阈值对照。

### 10.2 需要形成统一 paper table

至少比较：

```text
Native
CVRRM W8
CVRRM W16
V9-A1 best diagnostic
V9-A2 learned fixed0.05
V9-A2 learned OOF-F1
V9-A2 oracle
```

指标：

```text
AJ
OA
delta_avg
delta_4px
AJ_RD
AJ_RD_256
accepted extension count
good/bad/false-visible/candidate-worse extension count
positive/negative/zero videos
```

### 10.3 必做统计稳健性

由于只有 25 个 AJ_RD_256 defined videos，必须增加 paired video-level uncertainty：

```text
paired bootstrap confidence interval
或 paired permutation / sign test
```

重点比较：

```text
V9-A2 fixed0.05 vs W16
V9-A2 fixed0.05 vs W8
```

要报告：

```text
mean delta
median delta
95% CI
better/worse/equal count
```

不要只报告 aggregate mean。

### 10.4 必做 ablation

至少：

```text
feature set:
base / anchor / all

controller mode:
frame / event_max / event_mean

threshold protocol:
fixed0.05 / OOF-F1 / accept-all W16

anchor groups:
query-only / last-only / preocc-only / all anchors
```

最后一项可能需要按 feature names 做列屏蔽，不必重建 NPZ。

### 10.5 必做效率报告

记录：

```text
feature extraction time
controller inference time
event activation frequency
accepted extension percentage
p50 / p95 overhead（若可测）
```

### 10.6 V9-A2.4 输出建议

```text
scripts/v9a2_paper_ready_ablation.py
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_paper_ready_ablation.json
docs/v9a2_paper_ready_ablation_result_2026-07-10.md
```

完成后更新：

```text
CURRENT_MAINLINE.md
```

---

## 11. V9-A2.4 之后的决策门

### 如果统计稳健性成立

如果：

```text
fixed0.05 vs W16 的 paired CI 不明显跨负区间，
且 AJ/OA、AJ_RD_256、Pos/Neg/Zero 同时保持优势，
```

则：

```text
冻结 V9-A2 为 paper-ready dynamic horizon prototype。
```

下一步进入：

```text
方法写作 + 多数据集验证 + candidate-provider expansion
```

### 如果统计稳健性不成立

如果收益主要依赖少数视频，CI 跨 0 很宽，则：

```text
V9-A2 保留为正向诊断，不作为最终方法终点。
```

立即进入更强 identity 路线：

```text
V9-A2.5 semantic/internal identity feature
```

优先级：

```text
1. TrackOn2 internal descriptor / correlation feature
2. DINO dense feature aligned to anchor/candidate patches
3. learned identity verifier
```

先做小型 OOF ranking audit，不要直接重训全 tracker。

如果 semantic identity 仍无法缩小 oracle gap，则转：

```text
V9-A3 internal candidate head / multi-hypothesis ReEntryBeliefTrack
```

---

## 12. 工程与实验注意事项

```text
1. 不要使用 W16 common rows 替代 W8 canonical common rows。
2. canonical joint dataset 必须是 W8 common + W16-only extension。
3. 默认不要过滤 W8；W8 veto 只能作为 ablation。
4. 不要把 candidate_good 和 utility_positive 当成独立标签。
5. 不要只优化 damage16；主要风险是 fine candidate degradation。
6. 所有训练/阈值选择必须按 video group heldout，避免同视频泄漏。
7. 区分开发型 dense sweep 与 predeclared robust threshold 结果。
8. RGB patch identity 弱，不要把当前结果包装成完整 identity-aware solution。
9. 大型 NPZ/PT 缓存不提交 Git，除非明确需要并检查仓库策略。
10. 仓库可能存在其他研究方向提交，修改前先检查 git status，避免覆盖无关工作。
```

---

## 13. MCP / Git 状态说明

此前 MCP shell 曾卡住，重启 `fileSystemMCP.py` 后恢复。

GitHub SSH 最初因当前容器缺失 key/agent而失败，但截至本交接文档创建时：

```text
当前分支 HEAD 与 origin 已同步到 368fa7e。
```

新会话仍应先执行：

```bash
git status -sb --untracked-files=no
git branch -vv
git log --oneline -5
```

确认状态后再修改。

---

## 14. 一句话交接

```text
V9-A2 已证明：保留 W8、用 event-level controller 选择 W16 extension，在预声明 fixed0.05 阈值下可将 AJ_RD_256 Δ 从 W16 的 +0.0286 提到 +0.0294，并把 Pos/Neg/Zero 从 15/5/5 改善到 17/3/5；但收益 modest 且集中，RGB anchor 弱。下一步不是继续调阈值，而是做 V9-A2.4 paper-ready ablation、paired video-level uncertainty 和效率报告，再决定冻结方法还是升级 semantic/internal identity。
```

---

## 15. V9-A2.4 已完成（2026-07-10 补充）

Artifacts:

```text
scripts/v9a2_paper_ready_ablation.py
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_paper_ready_ablation.json
docs/v9a2_paper_ready_ablation_design_2026-07-10.md
docs/v9a2_paper_ready_ablation_result_2026-07-10.md
```

Frozen main policy:

```text
all-logreg + event_max + fixed threshold 0.05
preserve W8 common rows
control W16 extension rows only
```

Aggregate results:

```text
W8 AJ_RD_256 Δ       +0.027411
W16 AJ_RD_256 Δ      +0.028587
V9-A2 fixed0.05      +0.029379
Oracle               +0.032107
```

Ablation findings:

```text
event_max > frame and event_mean on AJ_RD_256
all features > base-only and anchor-only at fixed0.05
all anchors > individual query/last/preocc groups at fixed0.05
RGB anchor ranking remains weak despite trajectory-level complementary value
```

Paired video-level fixed0.05 vs W16:

```text
N=25
mean +0.000665
median 0
bootstrap 95% CI [-0.000051, +0.001681]
better/worse/equal = 4/2/19
exact sign-flip p = 0.15625
```

Decision:

```text
V9-A2 is a positive paper-ready prototype/diagnostic but not a statistically established final method on DAVIS. Do not continue threshold tuning. Next experiment is V9-A2.5 semantic/internal identity feature audit under the frozen fixed0.05 dynamic-horizon protocol.
```

---

## 16. V9-A2.5 DINOv3 semantic identity pilot 已完成

Key artifacts:

```text
outputs/paper_discovery_2026-07-05/v9a25_dinov3_identity/v9a25_dinov3_identity_eval.json
outputs/paper_discovery_2026-07-05/v9a25_dinov3_identity/v9a25_dinov3_late_fusion_audit.json
outputs/paper_discovery_2026-07-05/v9a25_dinov3_identity/v9a25_dinov3_feature_family_audit.json
docs/v9a25_dinov3_identity_feature_pilot_result_2026-07-10.md
docs/v9a25_dinov3_late_fusion_audit_result_2026-07-10.md
docs/v9a25_dinov3_feature_family_audit_result_2026-07-10.md
```

Result:

```text
DINO-only logreg OOF: AP 0.3637, AUC 0.6429
all+DINO primary AJ_RD_256 Δ: +0.029439
frozen V9-A2 AJ_RD_256 Δ: +0.029379
paired primary-vs-frozen mean: +0.000035
95% CI: [-0.000227,+0.000277]
exact sign-flip p: 0.875
```

Late fusion and feature-family audit:

```text
Frozen and DINO row scores are almost uncorrelated, but predeclared fusion weights do not beat frozen V9-A2.
Historical candidate identity is real (AP/AUC 0.3703/0.6532), and local distinctiveness is the strongest ranking family (AP/AUC 0.3967/0.6537).
Every feature family's paired trajectory CI versus frozen V9-A2 crosses zero.
```

Decision:

```text
DINO identity is informative for row ranking, including genuine historical-anchor and local-distinctiveness signal, but does not provide a robust trajectory improvement. Stop DINO concatenation/fusion/family tuning. Next: TrackOn2 internal matching/memory feature feasibility audit; then V9-A3 if needed.
```
