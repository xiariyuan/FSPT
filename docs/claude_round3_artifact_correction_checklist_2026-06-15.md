# Claude Round 3 Artifact Correction Checklist (2026-06-15)

## 0. 目的

这份文档不是新一轮 baseline 复现清单。

它只做一件事：

**修正第二轮执行后留下的工件一致性问题，让当前结论变成“可引用、可交接、不会误导下一轮执行”的状态。**

当前不需要再重复：

1. Track-On 环境修复
2. DINOv3 本地加载
3. Track-On2 DINOv3 backbone smoke
4. Track-On2 DAVIS/Kinetics repo-native eval
5. CoTracker3 本地 repo-native eval

这些步骤已经做过。

第三轮目标是：

1. 补齐缺失工件
2. 修正命名和指标映射错误
3. 补全 status JSON
4. 把 `final_decision.md` 改成准确表述

---

## 1. 当前必须修正的 4 个问题

### 1.1 `fallback_decision.md` 缺失

第二轮总结声称已生成：

- `outputs/fallback_decision.md`

但当前工作区中没有这个文件。

第三轮必须补上。

注意：

这里的内容不是“触发 fallback”。

正确内容应是：

1. DINOv3 路线通过
2. 不需要回退到 DINOv2
3. Track-On-R 阻塞与 DINOv3 backbone 无关，而是 checkpoint 缺失

### 1.2 CoTracker3 的命名 / 变体映射混乱

当前存在三处不一致：

1. `outputs/cotracker3_offline_repo_native_metrics.json`
   - 指标是 `AJ 62.66 / δ_avg 77.22 / OA 88.15`
   - checkpoint 是 `scaled_offline.pth`
2. `outputs/cotracker3_online_repo_native_metrics.json`
   - 指标是 `AJ 64.89 / δ_avg 77.36 / OA 91.80`
   - checkpoint 是 `scaled_online.pth`
3. `outputs/attempt0_2026-06-15_recovery/final_decision.md`
   - 却写成：
     - `cotracker3_offline = 64.89`
     - `cotracker3_baseline = 62.66`

同时：

- `status/cotracker3_offline.json` 的 `checkpoint_path` 仍写的是 `scaled_offline.pth`
- 但它的数值却是 `64.89 / 77.36 / 91.80`

这说明：

**artifact label、status name、实际 predictor variant、checkpoint 路径没有对齐。**

### 1.3 Status JSON 仍未按 Attempt 0 要求填完整

尤其是下面这些文件：

1. `outputs/attempt0_2026-06-15_recovery/status/cotracker3_baseline.json`
2. `outputs/attempt0_2026-06-15_recovery/status/cotracker3_offline.json`
3. `outputs/attempt0_2026-06-15_recovery/status/trackon2.json`

当前仍缺或存在问题的字段包括：

1. `checkpoint_sha256`
2. `official_reference_numbers`
3. `delta_vs_official`
4. `repo_native_metric_names`
5. `rescoring_status`
6. `notes`
7. `next_action`

### 1.4 `final_decision.md` 的表述过度

当前文件标题与内容给人的感觉是：

- Attempt 0 已做完并可直接定主线

但实际并非如此。

当前真实状态是：

1. 已完成 `Track-On2` repo-native reproduction
2. 已完成 `CoTracker3` 本地 repo-native reproduction
3. `Track-On-R / TAPNext++ / AllTracker` 仍 blocked
4. unified cache / `attempt0_validate_cache.py` / `attempt0_rescore_cache.py` 仍未形成可信闭环
5. 已发现 `track_on` evaluator 与 `datasets/metrics.py` 的坐标 / shape / 归一化语义不兼容

所以当前文件应改成：

**repo-native partial decision**  
而不是  
**Attempt 0 fully completed decision**

---

## 2. 第三轮的总目标

第三轮执行完成后，应达到下面状态：

1. 所有“已生成但实际缺失”的工件都补齐
2. CoTracker3 的 artifact name / status name / variant meaning 对齐
3. `final_decision.md` 不再误导下一轮
4. 下一位执行者看到当前目录时，能一眼区分：
   - repo-native 已完成部分
   - unified rescoring 未完成部分
   - 真正 blocked 的 baseline

---

## 3. Phase R3-A: 补齐缺失工件

### Step A1. 生成 `outputs/fallback_decision.md`

必须新增：

- `outputs/fallback_decision.md`

内容至少写清：

1. 本轮没有触发 fallback
2. DINOv3 本地 backbone 路线成功
3. 不需要切换到 `Track-On2 DINOv2`
4. 当前真正阻塞 `Track-On-R` 的不是 backbone，而是：
   - `track_on_r.pt`
   - `verifier.pt`
5. 该文件只是对 fallback 判定的书面归档

### Step A2. 检查 `final_decision.md` 中列出的工件是否真实存在

至少核对：

1. `track_on_env_manifest.md`
2. `dinov3_load_smoke.json`
3. `trackon_local_backbone_smoke.txt`
4. `track_on_checkpoint_manifest.md`
5. `track_on2_repo_native_metrics.json`
6. `track_on2_kinetics_metrics.json`
7. `trackon_adapter_sanity_report.json`
8. `trackon_metric_parity_report.json`
9. `track_on_unified_rescoring.json`
10. `week1_partial_decision.md`
11. `fallback_decision.md`

如果某个文件不存在，就不要继续在文档里宣称它已产出。

---

## 4. Phase R3-B: 修正 CoTracker3 的命名与映射

### Step B1. 明确定义 3 个概念

第三轮必须先在书面上固定下面三个概念：

1. `cotracker3_baseline`
   - 在 Attempt 0 的冻结候选集中，它到底代表什么
   - 推荐定义为：当前仓库已有 comparison baseline 行，对应较弱 / 默认行
2. `cotracker3_offline`
   - 明确代表 offline 强锚点
3. `cotracker3_online`
   - 如果这次实际跑的是 online variant，就要明确它与 `baseline` 的关系

如果 `cotracker3_baseline` 在你的实现中其实等于 `online variant`，那就必须在文档里写明：

- `baseline row == CoTracker3 online variant`

不要让文件名、status name、checkpoint 路径三者互相矛盾。

### Step B2. 统一 artifact 命名策略

当前已有：

1. `outputs/cotracker3_offline_repo_native_metrics.json`
2. `outputs/cotracker3_online_repo_native_metrics.json`

第三轮要决定并执行下面二选一：

#### 方案 1：保留 `online/offline` 真实命名

则：

1. `status/cotracker3_offline.json` 必须对应 offline checkpoint + offline 指标
2. `status/cotracker3_baseline.json` 必须明确写：
   - baseline row maps to online variant
3. `final_decision.md` 必须显式说明：
   - `cotracker3_baseline` 使用的是 online variant row

#### 方案 2：重命名 artifact 以贴合 Attempt 0 行名

例如补生成：

1. `outputs/cotracker3_baseline_repo_native_metrics.json`
2. `outputs/cotracker3_offline_repo_native_metrics.json`

并让二者分别与 Attempt 0 的两行严格对应。

不管选哪种，都必须做到：

1. 文件名
2. status JSON
3. `final_decision.md`
4. `cotracker3_checkpoint_manifest.md`

四者完全一致。

### Step B3. 修正 `cotracker3_checkpoint_manifest.md`

当前 manifest 里写的是：

| Metric | CoTracker3 Offline | CoTracker3 Online |
|--------|-------------------|-------------------|
| AJ | 62.66 | 64.89 |

但这与 `final_decision.md` 里的行名对应关系冲突。

第三轮必须修正 manifest，使它能够回答两个问题：

1. 真实运行了哪些 predictor 变体
2. 这些变体分别映射到 Attempt 0 的哪一行

建议在 manifest 里新增一节：

- `Attempt 0 row mapping`

例如：

- `cotracker3_baseline -> cotracker3_online`
- `cotracker3_offline -> cotracker3_offline`

或其他真实映射。

---

## 5. Phase R3-C: 补全 status JSON

### Step C1. `trackon2.json`

当前问题：

1. `rescoring_status` 被写成 `complete_via_evaluator_native`
2. 这不是 Attempt 0 原始 schema 里的标准闭环含义

第三轮必须把它写清楚，不要混淆：

可选做法：

1. 保留 `rescoring_status = pending`
   - 并新增 `notes` / `rescoring_note`
   - 说明 repo-native 指标已完成，但 unified rescoring 因协议不兼容未完成
2. 或保留当前状态，但必须在 `final_decision.md` 明确说明：
   - 这不是 `attempt0_rescore_cache.py` 标准完成态

无论哪种，都必须保证下一位执行者不会误解成：

- Track-On2 已完成统一重算

### Step C2. `cotracker3_baseline.json`

至少补齐：

1. `checkpoint_sha256`
2. `repo_native_metric_names`
3. `official_reference_numbers`
4. `delta_vs_official`
5. `notes`
6. `next_action`

并确保它与真实映射一致。

### Step C3. `cotracker3_offline.json`

同样至少补齐：

1. `checkpoint_sha256`
2. `repo_native_metric_names`
3. `official_reference_numbers`
4. `delta_vs_official`
5. `notes`
6. `next_action`

并确保：

1. `checkpoint_path`
2. `repo_native_numbers`
3. `AJ / OA / <avg`

全部与对应 variant 一致。

### Step C4. Blocked baseline status 也要再收干净

当前 `trackonr / tapnextpp / alltracker` 已经从 skeleton 前进了一步，但还可以再明确一点。

至少补：

1. 官方 checkpoint URL
2. 缺失文件名
3. 若获得文件后第一步该运行什么命令

让下一轮执行者拿到 checkpoint 后能直接接着跑。

---

## 6. Phase R3-D: 重写 `final_decision.md`

### Step D1. 改标题

当前标题：

- `# Final Decision — Attempt 0 (2026-06-15 Recovery)`

建议改成更准确的标题，例如：

- `# Partial Repo-Native Decision — Attempt 0 Recovery (2026-06-15)`

或

- `# Attempt 0 Partial Decision (Repo-Native Only)`

关键是不要再暗示：

- Attempt 0 已闭环完成

### Step D2. 明确分三层结论

新文件必须清晰拆成三层：

1. **Confirmed Working**
   - Track-On2 DINOv3
   - CoTracker3 local variants
2. **Blocked by Missing Assets**
   - Track-On-R
   - TAPNext++
   - AllTracker
3. **Not Yet Resolved**
   - unified rescoring incompatibility
   - adapter/export semantics

### Step D3. 把当前结论改成“partial decision”

允许的结论形状应接近：

1. 当前可比较的 repo-native baselines 中，`trackon2_dinov3` 最强
2. 它可作为 **当前 recovery 主 teacher candidate**
3. 但这仍是 **repo-native partial decision**
4. 不应冒充为完整 Attempt 0 final ranking

### Step D4. 明确写出 unified rescoring 仍未完成

必须写进 `final_decision.md`：

1. 现有 `scripts/attempt0_rescore_cache.py` 与 `track_on` evaluator 在坐标语义 / shape / normalization 上不一致
2. 当前 authoritative 数字来自 evaluator-native metrics
3. 因此当前结论不等价于完整 Attempt 0 unified rescoring 结论

### Step D5. 删除或改写误导性句子

例如当前这类表述要改：

- `Proceed with trackon2_dinov3 as primary teacher candidate for FSPT recovery.`

应改成更精确的版本，例如：

- `Proceed with trackon2_dinov3 as the current repo-native best teacher candidate, pending unified rescoring resolution and missing-baseline checkpoint recovery.`

---

## 7. 第三轮最低交付物

第三轮结束时，至少必须新增 / 修正下面这些工件：

1. `outputs/fallback_decision.md`
2. 修正后的 `outputs/cotracker3_checkpoint_manifest.md`
3. 修正后的 `outputs/attempt0_2026-06-15_recovery/status/trackon2.json`
4. 修正后的 `outputs/attempt0_2026-06-15_recovery/status/cotracker3_baseline.json`
5. 修正后的 `outputs/attempt0_2026-06-15_recovery/status/cotracker3_offline.json`
6. 修正后的 `outputs/attempt0_2026-06-15_recovery/final_decision.md`

可选但推荐：

7. 一个简短的 `outputs/attempt0_2026-06-15_recovery/correction_notes_round3.md`

用来列出：

1. 哪些命名被修正
2. 哪些结论被降级为 partial
3. 哪些工件是第三轮补出来的

---

## 8. 第三轮禁止事项

1. 不要再重跑 Track-On2 DINOv3 backbone smoke
2. 不要再重跑环境修复
3. 不要扩展到新的 baseline 训练
4. 不要尝试解决 unified rescoring incompatibility 的完整工程问题
5. 不要把 partial decision 再包装成 final Attempt 0 completion

第三轮是 **artifact correction pass**，不是新实验轮次。

---

## 9. 一句话交接给 Claude

直接把下面这段发给 Claude：

先执行 `/gemini/code/FSPT/docs/claude_round3_artifact_correction_checklist_2026-06-15.md`。

这不是新实验轮次，而是工件修正轮次。重点只做 4 件事：

1. 补上实际缺失的 `outputs/fallback_decision.md`
2. 修正 `CoTracker3 baseline / offline / online` 的命名、checkpoint 路径、status JSON 和 `final_decision.md` 之间的映射错误
3. 把 `trackon2/cotracker3_*` 的 status JSON 补完整，不要让 `rescoring_status` 和字段含义误导下一轮
4. 把 `outputs/attempt0_2026-06-15_recovery/final_decision.md` 改成准确的 **repo-native partial decision**，明确说明 unified rescoring 仍未完成

全过程不需要重跑 DINOv3 环境和 Track-On2 smoke，不允许把 partial decision 继续写成完整 Attempt 0 结论。
