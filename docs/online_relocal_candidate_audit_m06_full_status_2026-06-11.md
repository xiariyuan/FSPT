# M0.6 Full Status: Online Template-Bank Candidate Route Audited (2026-06-11)

## 结论

**M0.6 审计发现遮挡过滤 bug，修复后当前 online template-bank candidate generation 路线被证伪：oracle gap 为零。基于现有候选做 top-k selector 没有工程意义。不是"所有 online recovery 都不可行"——失败点在 search space / candidate generation，不是 gate 或 selector。如果未来重开 online 分支，方向应是重做 candidate generator（如 local crop constrained search）。**

## 关键发现：遮挡过滤 bug

之前的审计没有过滤遮挡帧（GT=(0,0)），导致 "oracle gap" 是虚假的。TAP-Vid DAVIS 中遮挡帧 GT 被设为 (0,0)，审计脚本把"预测位置到 (0,0) 的距离"也算作 base_error，产生 500-700px 的虚假 "base_bad" 样本。修复后，这些样本被正确跳过。

修复后（n=106，只保留可见帧）：

| 指标 | 修复前（含遮挡帧） | 修复后 (n=106) |
|------|---------------|---------------|
| base_median | ~7.5px | **4.5px** |
| base_bad>16px | ~54 (~37%) | **13 (12%)** |
| base_very_bad>32px | ~46 (~31%) | **5 (5%)** |
| CoTracker oracle_better_2px (all) | ~25.9% | **0.0%** |
| CoTracker oracle_better_2px (base_bad) | ~64.0% | **0.0%** |
| DINO oracle_better_2px (all) | — | **0.9%** |

> 注：修复前的数字来自 in-memory 中间结果，未落盘为正式 JSON 工件。`outputs/topk_oracle_audit_cotracker_v2.json` 是更早的 n=75 小样本审计。精确的 pre-fix 统计不可追溯，但"bug 存在且修复后 gap 消失"的结论不受影响。
| DINO oracle_better_2px (base_bad) | 32.0% | **7.7%** |

## Full-Triggered 审计结果（30 batches, occlusion-filtered）

### Template-Bank Top-K Oracle

| | CoTracker | DINO |
|---|---|---|
| n | 106 | 106 |
| topk_kind | template_bank | template_bank |
| valid_topk_frac | 1.0 | 1.0 |
| base_median (all) | 4.5px | 4.5px |
| oracle_median (all) | **160.8px** | **66.8px** |
| oracle_better_2px (all) | **0.0%** | **0.9%** |
| oracle_better_2px (base>16px) | **0.0%** | **7.7%** |
| oracle_better_2px (base>32px) | **0.0%** | **0.0%** |
| pairwise_dist_mean | 230px | 42px |
| unique_modes | 3.84 | 2.19 |

**解读**：template-bank 候选在可见帧上完全无效。oracle_median（160/67px）远高于 base_median（4.5px），说明候选位置比 base 还差得多。

### Raw Candidate Audit

| | CoTracker | DINO |
|---|---|---|
| failure_type | **Type B** | **Type C** |
| raw_delta median | **252.2px** | **86.8px** |
| raw_delta p90 | 485.5px | 201.0px |
| gated_delta median | **0.0px** | **2.0px** |
| gate_value median | 0.0000 | 0.018 |
| relocal_conf median | 0.002 | 0.018 |
| raw_leaves_base | 100% | 100% |
| gate_collapses | 100% | 20.8% |

**解读**：
1. CoTracker raw step 移动 252px（非常大），但 gate 完全关闭 → Type B
2. DINO raw step 移动 87px，gate 保留 2px → Type C
3. 两种 raw step 都是有害的（oracle 比 base 差），gate 的抑制是正确的

## 7 条通过标准检查

| # | 标准 | 结果 |
|---|------|------|
| 1 | topk_kind == "template_bank" | **PASS** |
| 2 | valid_topk_frac >= 0.95 | **PASS** (1.0) |
| 3 | base_bad oracle_better_2px >= 0.50 | **FAIL** (0.00) |
| 4 | base_very_bad oracle_better_2px >= 0.70 | **FAIL** (0.00) |
| 5 | CoTracker >= DINO + 0.15 on base_bad | **FAIL** (0.00 vs 0.077) |
| 6 | CoTracker raw never leaves | **FAIL** (100% leaves) |
| 7 | No single video >50% oracle-positives | **N/A** (no positives) |

**Stage 1 未通过**。标准 3/4/5 是核心——oracle gap 在可见帧上不存在。

## 根因分析

### 为什么之前的审计显示 oracle gap？

1. TAP-Vid DAVIS 中，遮挡帧的 GT 被设为 (0,0)
2. 审计脚本在计算 error 时没有过滤这些帧
3. 当 base 预测在 (0.75, 0.58) 而 GT 在 (0,0) 时，base_error ≈ 700px → "base_bad"
4. template-bank 候选在 (0.5, 0.31) → oracle_error ≈ 480px → "oracle_better"
5. 这只是两个错误位置之间的距离比较，不是真正的改善

### 为什么 template-bank 在可见帧上无效？

1. **Base tracker 已经很好**：93% 的 triggered+visible 样本 base_error < 16px
2. **Template 匹配不精确**：support template 在当前帧中的最佳匹配位置不等于 GT
3. **没有空间约束**：模板搜索是全局的，候选位置散布在整帧（pairwise_dist=230px）
4. **根本矛盾**：当 base 好时不需要 recovery；当 base 差时候选不够好

## 当前在线 recovery pipeline 状态

```
[Trigger] → [Search] → [Gate] → [Retrack]
   OK        失败       正确      N/A
```

- **Trigger**：正确触发（106/30*219 ≈ 1.6% 触发率）
- **Search（瓶颈）**：template-bank 全局搜索无空间约束，CoTracker/DINO 都产生有害的大位移（252/87px）
- **Gate**：正确抑制有害位移——gate 在做正确的事，不是问题所在
- **Retrack**：因为 gate 全部关闭，retrack 从未启动

## 下一步选择

**停止 template-bank selector 路线。** 失败点在 candidate generation（search space 太大），不是 gate 或 selector。

### 当前优先级：回到 PRT offline 论文主线

在线 recovery 的当前 candidate generator 不具备可选性。PRT offline 结果已 ready。

### 如果未来重开 online 分支

方向应是重做 candidate generator，而非重做 gate/selector：

1. **Local crop constrained search**：从 base 位置裁剪区域（如 ±32px），只在 crop 内搜索。这是 PRT offline 有效的核心原因。
2. **Learned spatial prior / localizer**：用网络直接预测 re-entry 位置的 heat map。
3. **更强的 support-to-search matching**：比全局 cosine similarity 更精确的匹配机制。

这些都需要更根本的架构改变，不在当前项目范围内。

## 工件清单

- `outputs/topk_oracle_audit_cotracker_v3_full.json` — CoTracker top-k (occlusion-filtered)
- `outputs/topk_oracle_audit_dino_nogate_v3_full.json` — DINO top-k (occlusion-filtered)
- `outputs/relocal_candidate_audit_cotracker_v2_full.json` — CoTracker raw (occlusion-filtered)
- `outputs/relocal_candidate_audit_dino_nogate_v2_full.json` — DINO raw (occlusion-filtered)
- `outputs/topk_oracle_audit_compare_v3_full.json` — 对比 JSON
