# Claude 执行计划：Re-Acquisition / Re-Detection 主线（2026-06-16）

这份文档是当前 FSPT 项目新的主线执行清单。

它建立在一个已经确认的判断上：

> `hard oracle mask / gate-only memory hygiene` 不是 free gain。它能提高 `long_occ_AJ`，但会显著伤害 `re-entry`，因此不再作为主方法继续投入。

新的主线不是继续改 memory，而是：

> **explicit re-acquisition / re-detection**

也就是把问题从“遮挡期间别写坏 memory”切到“重现时显式找回目标”。

---

## 0. 当前总判断

### 已停止作为主线的方向

以下方向不再继续作为当前主线：

1. predicted visibility gate
2. hard gate memory write
3. soft retention 作为近期主线
4. 纯 dual-memory 作为近期主线
5. 围绕 `oracle_mask` 再做更多同类 ablation

这些结果保留为：

- 诊断证据
- 负结果
- 项目转向的依据

### 新主线

新的第一优先级是：

1. **Track-On2 / Track-On-R comparison baseline 保持不动**
2. **方法开发回到现有已打通的 CoTracker3 online recovery integration path**
3. **主问题切换为 re-acquisition / re-detection**
4. **主指标切换为 re-entry 优先，而不是 overall AJ 优先**

---

## 1. 你必须先读的文件

先完整阅读下面这些文件，不要跳过：

### 新主线与现有证据

1. `/gemini/code/FSPT/docs/online_recovery_module_design_2026-06-09.md`
2. `/gemini/code/FSPT/docs/online_recovery_training_plan_2026-06-10.md`
3. `/gemini/code/FSPT/docs/online_recovery_status_report_2026-06-10.md`
4. `/gemini/code/FSPT/docs/online_recovery_head_m0_status_2026-06-10.md`

### 刚结束的 memory hygiene 结论

5. `/gemini/code/FSPT/outputs/memory_hygiene_oracle_2026-06-16_full/decision.md`
6. `/gemini/code/FSPT/docs/memory_hygiene_experiment_review_2026-06-16.md`

### 关键代码入口

7. `/gemini/code/FSPT/models/cotracker_refiner.py`
8. `/gemini/code/FSPT/models/online_recovery_head.py`
9. `/gemini/code/FSPT/scripts/debug_online_recovery_real_eval.py`
10. `/gemini/code/FSPT/scripts/eval_recovery_position_error.py`
11. `/gemini/code/FSPT/scripts/build_online_recovery_anchor_dataset_v3.py`
12. `/gemini/code/FSPT/scripts/train_online_recovery_head.py`

---

## 2. 当前最重要的事实边界

### 2.1 这次不要再回到 memory hygiene 方向

不要再做：

- 新的 oracle mask 变体
- predicted gate
- hard/soft write 新 ablation
- persistent anchor-only 新实验

原因不是这些永远没价值，而是：

- 当前 full DAVIS 证据已经足够说明这条线不是近期主线
- 继续做只会消耗时间，不会改变路线判断

### 2.2 当前最值得复用的开发基线是 CoTracker3 online path

原因：

1. 现有 `online recovery` 链路已打通
2. 现有 `trigger -> search -> confidence -> retracking splice` 在真实数据上能跑
3. 现有 DINO recovery-only 特征源已经接入
4. 现有 offline recovery head 资产已经存在

所以本轮主线不是“重新选 base model”，而是：

> **沿用 CoTracker3 online recovery integration，升级 recovery geometry / re-detection。**

### 2.3 当前最准确的问题定义

不要再写成：

- “memory 被污染，所以 long-occ 失败”

当前更准确的定义是：

> 长遮挡之后的核心瓶颈在 `re-entry / re-acquisition`，而不是单纯的 memory hygiene。

---

## 3. 本轮目标

本轮不是直接追 overall AJ，也不是直接做大训练。

本轮目标分三步：

1. **确认现有 online recovery 集成资产是否还能工作**
2. **确认 offline geometric head 是否值得重新接回 online path**
3. **做一个最小可行的 re-acquisition baseline，对 re-entry 指标负责**

一句话：

> 先用最小代价验证“显式 re-detection / recovery head”是否能在 re-entry 指标上提供真实收益。

---

## 4. Phase A：现有 online recovery 资产体检

### A1. 跑真实数据链路审计

使用现有脚本：

- `/gemini/code/FSPT/scripts/debug_online_recovery_real_eval.py`

配置优先用：

- `configs/fspt_online_recovery_real_eval256.yaml`
- `configs/fspt_online_recovery_dino_real_eval256.yaml`

checkpoint 优先用：

- `checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth`

必须确认以下事情仍然成立：

1. online recovery 入口可触发
2. `relocal_mask` 非零
3. `relocal_conf` 正常
4. `retracking_retrack_mask_bn` 字段存在
5. DINO 特征源仍然比 CoTracker 特征源更容易触发 recovery 行为

产物：

- `outputs/reacquisition_mainline_2026-06-16/online_recovery_real_eval_audit.json`
- `outputs/reacquisition_mainline_2026-06-16/online_recovery_dino_real_eval_audit.json`
- `outputs/reacquisition_mainline_2026-06-16/phaseA_status.md`

### A2. 不要做的事

此阶段不要：

- 改网络结构
- 加新 loss
- 改 trigger 逻辑
- 重训

目标只是体检，不是改方法。

---

## 5. Phase B：重新核对 offline anchor / geometric head 资产

### B1. 核对 dataset v3 是否完整

优先检查：

- `outputs/recovery_anchor_dataset_v3_full/coverage_audit.json`
- `outputs/recovery_anchor_baselines_v3_full_val.json`

要写清楚：

1. train / val sample 数
2. val video 列表
3. `gt_inside` 覆盖率
4. `base >16px` / `base >32px` 子集规模

### B2. 核对 M0 head 现状

优先核对：

- `checkpoints/online_recovery_head_m0/best.pth`
- `outputs/train_online_recovery_head_m0/train_log.json`
- `outputs/train_online_recovery_head_m0_seed43/val_metrics.json`
- `outputs/train_online_recovery_head_m0_seed44/val_metrics.json`
- `docs/online_recovery_head_m0_status_2026-06-10.md`

现有共识是：

- M0 head 在 base-bad 子集上有改善信号
- 但 val bad subset 太小，统计支撑不足

因此这一步的任务不是重新下结论，而是确认：

> offline geometric head 仍然是当前最值得接回 online path 的候选资产。

### B3. 产物

- `outputs/reacquisition_mainline_2026-06-16/offline_head_asset_review.md`

必须明确写出：

1. 哪个 checkpoint 作为首选 online integration 候选
2. 现有证据支持什么，不支持什么
3. 下一步是否需要先扩大 val 再接 online

---

## 6. Phase C：主指标切换到 re-entry

### C1. 使用现有脚本重新确认 recovery-specific 指标口径

优先读并必要时复用：

- `/gemini/code/FSPT/scripts/eval_recovery_position_error.py`

本阶段必须统一以下指标口径：

1. `reentry_first_error`
2. `reentry_mean_error_px`
3. `reentry_median_error_px`
4. `reentry_<4px`
5. `reentry_<8px`
6. `n_queries_with_reentry`
7. `n_retracked`

### C2. 当前项目里最重要的 reporting 原则

以后所有 recovery 主线实验，结论优先级必须是：

1. `re-entry`
2. `long-occ`
3. `tail after retracking`
4. 最后才是 overall AJ

如果某方法：

- overall AJ 微动
- 但 re-entry 显著变好

这仍然是正信号。

反过来，如果：

- overall AJ 不坏
- 但 re-entry 更差

则不算成功。

### C3. 产物

- `outputs/reacquisition_mainline_2026-06-16/reentry_metric_protocol.md`

---

## 7. Phase D：做最小可行 re-acquisition baseline

这是本轮真正的核心执行。

### D1. 目标

不要一开始做复杂训练。

先做一个最小可行的 re-acquisition baseline，回答这个问题：

> 在真实 tracker-conditioned re-entry 场景里，显式 recovery head / redetection 分支能否比当前 raw relocal anchor 更准？

### D2. 首选路线

优先把 **现有 offline geometric head** 接回 online path。

具体优先级：

1. `DINO search feature map`
2. `support descriptor`
3. `OnlineRecoveryHead`
4. 替换当前 raw cosine / raw anchor readout

不要先做：

- 新 transformer
- 新 verifier
- 新 teacher
- 新 memory bank 机制

### D3. 最小接入方式

第一版只做：

1. 维持现有 trigger
2. 维持现有 support bank
3. 维持现有 retracking splice
4. 只替换 `anchor readout`

也就是：

- old path: `raw cosine -> anchor`
- new path: `DINO search fmap + support descriptor -> OnlineRecoveryHead -> anchor`

### D4. 首轮只做小规模 integration smoke

先用：

- `max_batches = 10`
- DAVIS real eval

目标不是最终数值，而是确认：

1. head 输出坐标没有坐标系错位
2. retracking 确实吃到了 head 预测 anchor
3. `reentry_first_error` 可计算
4. 不会出现“head 预测几何改了，但 online splice 根本没用”的假接入

### D5. 产物

必须新增：

- `outputs/reacquisition_mainline_2026-06-16/integration_smoke_audit.json`
- `outputs/reacquisition_mainline_2026-06-16/integration_smoke_reentry_metrics.json`
- `outputs/reacquisition_mainline_2026-06-16/integration_smoke_summary.md`

---

## 8. Phase E：是否重训

只有在 D 阶段 smoke 满足以下任一条件，才允许继续训练：

1. head 接入后，`reentry_first_error` 有明确下降
2. `retracked` 子集的误差相对 raw anchor 有改善
3. 至少在 `base-bad` 子集上改善明显

如果 smoke 没有任何正信号，则：

- 不要继续重训
- 直接写 stop note

### E1. 如果需要重训，优先做什么

优先做：

1. 扩大 bad-subset val
2. 重新训练 `OnlineRecoveryHead`
3. 保持网络轻量

不要优先做：

1. gate head
2. accept/reject policy
3. verifier as decision layer
4. 更复杂多任务训练

### E2. 如果重训，首选命令方向

优先沿用：

- `scripts/train_online_recovery_head.py`
- dataset: `outputs/recovery_anchor_dataset_v3_full`

但在开始重训前，必须先写清楚：

1. 当前 val bad subset 为什么不足
2. 扩大策略是什么
3. 这次 best selection criterion 是什么

---

## 9. 停止条件

以下任一成立，就停止这条新主线的当前实现版本：

1. online integration smoke 无法把 head 预测真实写回 retracking
2. re-entry 指标没有任何改善信号
3. 改善只来自极少数异常视频，方向不稳定
4. head 只能在 offline bad subset 上成立，接回 online 后完全消失
5. 需要大规模复杂结构改动才能看到任何正信号

出现这些情况时，不要再在当前 CoTracker3 online recovery 路径上消耗更多时间。

---

## 10. 本轮你最终必须交付的内容

### 文档

1. `outputs/reacquisition_mainline_2026-06-16/phaseA_status.md`
2. `outputs/reacquisition_mainline_2026-06-16/offline_head_asset_review.md`
3. `outputs/reacquisition_mainline_2026-06-16/reentry_metric_protocol.md`
4. `outputs/reacquisition_mainline_2026-06-16/integration_smoke_summary.md`
5. `outputs/reacquisition_mainline_2026-06-16/final_decision.md`

### JSON / 审计工件

1. `online_recovery_real_eval_audit.json`
2. `online_recovery_dino_real_eval_audit.json`
3. `integration_smoke_audit.json`
4. `integration_smoke_reentry_metrics.json`

### 代码

如果做了 online integration 改动：

1. 列出改动文件
2. 明确说明是否只替换 anchor readout
3. 明确说明是否动了 trigger / support bank / retracking splice

---

## 11. 给 Claude 的一句话任务定义

停止继续做 memory hygiene 主线，把开发重心切到现有 CoTracker3 online recovery integration path 上的 explicit re-acquisition / re-detection：先体检现有 online recovery 资产，再复核 offline geometric head 和 recovery-anchor dataset v3，随后只做一个最小可行的“用现有 OnlineRecoveryHead 替换 raw anchor readout”的 integration smoke，并用 re-entry 指标而不是 overall AJ 判断是否值得继续。
