# FSPT / V9-A2 新会话交接文档

更新时间：2026-07-10  
项目根目录：`/gemini/code/FSPT`  
当前主线：V9-A2 ReEntryBeliefTrack / Dynamic Horizon

---

## 0. 给新会话的第一条指令

新会话开始后，请先完整阅读：

```text
/gemini/code/FSPT/docs/SESSION_HANDOFF_V9A2_2026-07-10.md
/gemini/code/FSPT/CURRENT_MAINLINE.md
```

然后重点阅读：

```text
/gemini/code/FSPT/docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-09.md
/gemini/code/FSPT/docs/v9a2_dynamic_horizon_video_stability_audit_result_2026-07-09.md
/gemini/code/FSPT/docs/v9a2_dynamic_horizon_controller_result_2026-07-08.md
/gemini/code/FSPT/docs/v9a2_full_feature_build_result_2026-07-08.md
```

不要重新从 V9-A1 或 V9-A2 smoke 开始。当前应从 **V9-A2.4 paper-ready ablation / event-level objective refinement** 继续。

---

## 1. Git 与环境状态

仓库：

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

远端：

```text
origin git@github.com:xiariyuan/FSPT.git
```

截至写本文档前：

```text
origin/mainline-pivot-foundation-20260626 == HEAD
cached ahead/behind = 0/0
tracked working tree clean
```

最近相关提交：

```text
368fa7e Fix V9-A2 robust threshold and video stability audits
ef242fa Add reentry viscalibrator model and MCP service scripts
9d84db0 Add TAPNext++ evaluation, ReEntry cross-baseline comparison, and V9-A2 controller experiments
fa22457 Add V9-A2 dynamic horizon controller experiments
```

Git 本地身份当前设置为：

```text
user.name  = ChatGPT
user.email = chatgpt@openai.local
```

执行工具：

```text
点追踪.shell
workdir = FSPT
```

注意：曾出现 `fileSystemMCP.py` 卡住，表现为连 `date` 都超时。若再次出现，先在服务器终端检查并重启 MCP，而不是持续重复 Git/实验命令。

---

## 2. 项目问题与研究主线

目标：改进在线点追踪在长遮挡后的重入恢复，主干 tracker 为 CoTracker3 online，外部候选来源主要是 TrackOn2。

当前论文级主线不是简单 learned gate，而是：

```text
ReEntryBeliefTrack / Dynamic Horizon
Track → Coast → Reacquire
uncertainty-aware belief
identity-aware reacquisition
safe memory write
```

当前已验证的最重要经验：

```text
W8-only filtering 无法突破 CVRRM 的 recall 上限。
必须保留 W8 common recovery，并选择性扩展 W16-only action space。
```

---

## 3. 冻结基线与核心指标

Native CoTracker3 online cache：

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/
  cotracker3_v7a4_raw_visconf_full30.pt
```

TrackOn2 candidate cache：

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/
  trackon2_dinov3_davis_first_input_bridge.pt
```

Native 指标：

```text
AJ         65.2366
OA         90.8186
delta_avg  77.9458
delta_4px  85.8670
AJ_RD      0.3534
AJ_RD_256  0.5333
```

CVRRM W8：

```text
AJ Δ         +0.0964
OA Δ         +0.9462
delta_avg Δ  +0.3971
delta_4px Δ  +0.5085
AJ_RD Δ      +0.0153
AJ_RD_256 Δ  +0.0274
Pos/Neg/Zero 16/4/5
accepted     1456
```

CVRRM W16：

```text
AJ Δ         +0.0810
OA Δ         +0.9701
delta_avg Δ  +0.4339
delta_4px Δ  +0.5593
AJ_RD Δ      +0.0162
AJ_RD_256 Δ  +0.0286
Pos/Neg/Zero 15/5/5
accepted     2013
```

Oracle W8 + W16-only candidate-good：

```text
AJ Δ         +0.1955
OA Δ         +1.0204
AJ_RD_256 Δ  +0.0321
Pos/Neg/Zero 17/3/5
accepted     1574
```

Oracle 的意义：证明 W16 extension 存在真实 trajectory-level headroom；不是可部署方法。

---

## 4. V9-A1 结论：不要回到 W8-only controller

V9-A1 使用原 V8-C0.2 的 28 维 tabular feature，做 video-heldout learned gate。

最佳 OOF classifier 的 AP/AUC 不差，但 apply-back 无法超过 CVRRM W8 的 AJ_RD_256：

```text
V9-A1 extratrees AJ_RD_256 Δ ≈ +0.0212
CVRRM W8 AJ_RD_256 Δ         = +0.0274
```

结论：

```text
过滤 W8 会损失 recovery recall。
V9-A1 只能作为风险诊断，不是主方法。
```

不要继续调 V9-A1 threshold。

---

## 5. V9-A2 已完成阶段

### 5.1 V9-A2.1 Action-Space Audit

W8 / W16 touched rows：

```text
W8 rows       = 1456
W16 rows      = 2013
common        = 1456
W16-only      = 557
```

W16-only：

```text
candidate_good = 118
candidate_good_mean = 0.2118
damage16_mean ≈ 0.0018
candidate_worse_px_mean ≈ 0.4452
```

结论：W16-only 有机会，但不能 accept-all，需要 dynamic horizon。

### 5.2 V9-A2.2 Oracle Upper Bound

```text
W8 + selective W16 oracle AJ_RD_256 Δ = +0.0321
Event-level oracle AJ_RD_256 Δ         = +0.0320
```

事件级动作几乎达到逐帧 oracle，支持 event-level horizon selector。

### 5.3 Feature Builder 修正

已完成：

```text
- normalized yx 不再预先 clip
- 显式 OOB features
- query / last / strict-preocc / candidate RGB patch features
- anchor reliability features
- strict preocc_t < first_event_t
```

Feature 维度：

```text
base   = 28
anchor = 115
all    = 143
```

Strict preocc audit：

```text
W8: 1456/1456 strict, fallback=0
W16: 2013/2013 strict, fallback=0
```

### 5.4 Label Overlap

当前标签定义下：

```text
candidate_good == utility_positive
Jaccard = 1.0
```

因此它们不是两个独立任务。

训练目标建议：

```text
primary: candidate_good
secondary: good_not_worse
risk: candidate_bad / false_visible / candidate_worse_px
damage16: 只用于安全审计，不适合作为主标签
```

### 5.5 Canonical Joint Dataset

W8/W16 common rows 的 event context 可能不同。662 个 common rows 在 feature 上不一致，根因是 W16 会绑定更早的 `first_event_t` / 更大的 event age，而不是 patch extraction bug。

因此已经构建 canonical joint dataset：

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/
  v9a2_joint_w8_common_plus_w16_extension_v3.npz
```

组成：

```text
W8 canonical common rows = 1456
W16-only extension rows  = 557
total                     = 2013
feature_dim_all           = 143
finite_rate               = 1.0
```

必须继续使用此 canonical joint dataset，不要直接用 W16 full dataset 训练所有行。

---

## 6. V9-A2.3c Learned Dynamic Horizon 最终结果

控制策略：

```text
W8 common rows 默认全部保留
仅控制 W16-extension rows
frame / event_max / event_mean 三种聚合
video-heldout OOF scores
```

初始 classifier OOF 结果：

```text
base / extratrees: AP 0.3233, AUC 0.6420
anchor / extratrees: AP 0.2435, AUC 0.5773
all / extratrees: AP 0.2528, AUC 0.5855
```

RGB anchor 特征单独不强，`all` 也没有在 row-level AP/AUC 上明显优于 base。

但 trajectory apply-back 的 robust、预声明阈值结果通过：

主冻结变体：

```text
all_logreg + event_max + fixed threshold 0.05
```

结果：

```text
AJ Δ         +0.1025
OA Δ         +0.9711
AJ_RD_256 Δ  +0.0294
accepted     1937
Pos/Neg/Zero 17/3/5
```

对比：

```text
W8 AJ_RD_256 Δ             +0.0274
W16 AJ_RD_256 Δ            +0.0286
Learned fixed 0.05         +0.0294
Oracle                     +0.0321
```

绝对增益：

```text
Learned - W8  = +0.0020 AJ_RD_256
Learned - W16 = +0.0008 AJ_RD_256
Oracle - W8   = +0.0047 AJ_RD_256
```

大约吃到 W8→oracle gap 的 42.6%，吃到 W16→oracle 剩余 gap 的约 22.9%。这些比例是辅助解释，不能替代原始指标。

Robust threshold audit 中同样表现稳定的变体：

```text
all_logreg event_max OOF-F1      AJ_RD_256 Δ +0.0293
all_logreg event_max fixed 0.01  AJ_RD_256 Δ +0.0293
all_logreg event_max fixed 0.02  AJ_RD_256 Δ +0.0293
anchor_logreg event_max 0.01     AJ_RD_256 Δ +0.0293
```

因此 +0.0294 不是只来自 dense trajectory best-sweep。

---

## 7. 视频级稳定性

冻结变体：

```text
learned_all_logreg_event_max_fixed_0.05
```

相对 W16，在 AJ_RD_256 有定义的视频上：

```text
defined videos = 25
undefined/skipped = 5
better = 4
worse  = 2
equal  = 19
mean learned-minus-W16 = +0.000665
```

主要正向视频：

```text
drift-straight  +0.010417
bike-packing    +0.004325
dance-twirl     +0.002278
car-shadow      +0.001873
```

负向视频：

```text
bmx-trees  -0.001871
parkour    -0.000389
```

结论：

```text
收益是真实但较小，并集中在少量视频。
不能宣称广泛、均匀提升。
```

---

## 8. 关键反思

### 8.1 路线正确，但当前模块仍是轻量原型

V9-A2 成功证明：

```text
preserve W8 + selectively extend W16
```

优于：

```text
W8-only filtering
W16 accept-all
```

但提升相对 W16 只有 +0.0008 AJ_RD_256，应诚实表述为 modest gain。

### 8.2 Row-level AP/AUC 与 trajectory metric 不完全一致

最佳 trajectory 结果来自 `all_logreg event_max`，但其 row-level AP/AUC 并不是最高。

这说明：

```text
candidate_good frame classification 不是完全 metric-aligned 的训练目标。
event aggregation 和 action structure 比单纯提高 row AP/AUC 更重要。
```

下一步不应只盲目换更强 classifier。

### 8.3 RGB identity anchor 较弱

Anchor-only OOF 不强，说明 deterministic RGB patch 不足以处理：

```text
形变、光照、尺度变化、遮挡后外观变化、非刚性目标
```

更强 identity 可能来自：

```text
DINOv3 features
TrackOn2 internal memory/descriptor
learned candidate descriptor
internal multi-hypothesis candidate head
```

但在上更强 feature 前，建议先完成 event-level objective / paper-ready ablation。

---

## 9. 当前关键文件

### 设计与结果文档

```text
docs/v9a2_anchor_uncertainty_reacquisition_design_2026-07-08.md
docs/v9a2_action_space_audit_result_2026-07-08.md
docs/v9a2_action_space_oracle_audit_result_2026-07-08.md
docs/v9a2_feature_builder_correction_result_2026-07-08.md
docs/v9a2_preocc_anchor_strict_fix_result_2026-07-08.md
docs/v9a2_full_feature_build_result_2026-07-08.md
docs/v9a2_dynamic_horizon_controller_result_2026-07-08.md
docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-09.md
docs/v9a2_dynamic_horizon_video_stability_audit_result_2026-07-09.md
```

### 主要脚本

```text
scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py
scripts/v9a2_action_space_oracle_audit.py
scripts/v9a2_build_joint_w8_w16_dataset.py
scripts/v9a2_dynamic_horizon_controller.py
scripts/v9a2_dynamic_horizon_robust_audit.py
scripts/v9a2_dynamic_horizon_video_stability_audit.py
```

### 主要输出

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/
  v9a2_joint_w8_common_plus_w16_extension_v3.npz
  v9a2_dynamic_horizon_controller_report.json
  v9a2_dynamic_horizon_robust_threshold_audit.json
  v9a2_dynamic_horizon_video_stability_audit.json
  v9a2_action_space_oracle_audit.json
  v9a2_label_overlap_audit.json
  v9a2_preocc_strict_anchor_audit.json
  v9a2_w8_w16_feature_consistency_audit.json
```

大型 NPZ / cache 一般不应随意重新提交 Git；优先提交脚本、文档和小型 JSON 报告。

---

## 10. 推荐的下一步：V9-A2.4

推荐顺序：

### Step 1：冻结 paper-ready 主变体

冻结：

```text
all_logreg + event_max + fixed threshold 0.05
```

不要再用 dense trajectory sweep 选择主结果。

### Step 2：制作 paper-ready ablation package

至少生成：

```text
Native
CVRRM W8
CVRRM W16
V9-A1 best
V9-A2 base-only
V9-A2 anchor-only
V9-A2 all features
V9-A2 frame-level
V9-A2 event_max
V9-A2 event_mean
V9-A2 fixed threshold 0.01 / 0.02 / 0.05 / 0.10
Oracle upper bound
```

指标：

```text
AJ
OA
delta_avg
delta_4px
AJ_RD
AJ_RD_256
accepted extension rows
candidate_good / bad / false-visible / worse accepted
Pos/Neg/Zero
per-video gain/loss
runtime / latency / activation rate
```

### Step 3：做 event-level target audit

当前 frame label 与 trajectory metric 不完全对齐。建议构建 event-level supervision：

```text
event_id = (video_id, query_idx, first_event_t)
action = keep W8 or extend W16
label = event-level W16 extension utility / trajectory contribution
```

协议必须 group/video-heldout，阈值只在 train folds 选，不能在全部 OOF trajectory 上反向选。

优先比较：

```text
frame candidate_good target
event utility target
event W8-vs-W16 target
```

### Step 4：再决定是否上 semantic identity

如果 event-level objective 后仍明显低于 oracle，再进入：

```text
V9-A3 semantic/internal identity
```

优先级：

```text
1. TrackOn2 internal descriptor / memory features
2. DINOv3 aligned anchor-candidate features
3. learned multi-hypothesis candidate head
```

不要直接继续堆 RGB patch threshold。

---

## 11. V9-A2.4 成败标准

强成功：

```text
严格 heldout / nested threshold protocol 下：
AJ_RD_256 > +0.0286
AJ/OA 不低于 W16
Pos/Neg/Zero 至少不差于 15/5/5
```

有价值成功：

```text
AJ_RD_256 接近或略高 W16，
同时 AJ/OA、负视频数、false-visible 风险更好。
```

失败：

```text
仅 classifier AP/AUC 提升；
apply-back 无提升；
收益依赖全量 trajectory threshold sweep；
收益只来自单个视频；
或 fixed/predeclared threshold 低于 W16。
```

---

## 12. 新会话可直接复制的任务说明

```text
请先完整阅读：
/gemini/code/FSPT/docs/SESSION_HANDOFF_V9A2_2026-07-10.md
以及：
/gemini/code/FSPT/CURRENT_MAINLINE.md

项目目录：/gemini/code/FSPT
分支：mainline-pivot-foundation-20260626
当前远端同步 commit：368fa7e

不要重新做 V9-A1 或 V9-A2 smoke。请从 V9-A2.4 开始：
1. 复核并冻结 all_logreg + event_max + fixed threshold 0.05；
2. 构建 paper-ready ablation package；
3. 设计并执行 event-level target / nested threshold audit；
4. 再判断是否进入 TrackOn2 internal descriptor 或 DINOv3 semantic identity。

每一步都要先分析、再执行、再复核；不要只看 AP/AUC，必须做完整 trajectory apply-back 和 per-video stability。
```

---

## 13. 不要做的事情

```text
- 不要回到 W8-only filter 继续调 threshold。
- 不要把 candidate_good 和 utility_positive 当成独立多任务标签。
- 不要直接用 W16 full common rows；使用 canonical joint dataset。
- 不要把 dense trajectory best threshold 当最终严谨结论。
- 不要宣称 RGB anchor 已经证明 identity modeling 很强。
- 不要宣称广泛视频提升；当前收益集中且 modest。
- 不要随意提交大型 NPZ/PT/PKL 到 Git。
```

---

## 14. 当前一句话结论

```text
V9-A2 已经证明：保留 CVRRM W8，并用事件级 learned dynamic horizon 选择性扩展 W16，可以在保守固定阈值下将 AJ_RD_256 Δ 从 W16 的 +0.0286 提升到 +0.0294，并将 Pos/Neg/Zero 从 15/5/5 改善到 17/3/5；但增益较小且集中，下一步应做 paper-ready ablation 与 metric-aligned event-level objective，再决定是否升级为 semantic/internal identity 模块。
```
