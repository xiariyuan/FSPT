# FSPT v2 — Re-detection / Re-association 四级实验阶梯
## 启动清单 (00_execution_manifest)

**日期**: 2026-06-17  
**session_id**: `3e7cc46c-bc14-43d0-b3bf-a4754fc13ab2` (延续)  
**前序**: M0 Re-Acquisition 主线 Phase A→E 完成，CoTracker3 head-to-head 完成  

---

## 1. 当前主线 vs 历史失败路线

### ✅ 当前主线
**FSPT v2: Re-detection / Re-association 四级实验阶梯**

### 🚫 历史失败路线（禁止继续当主线）
| 路线 | 失败结论 | 引用 |
|---|---|---|
| Memory hygiene / Soft retention / Dual-memory | Phase 1 oracle diagnostic → 无实质改善 | 旧 conversation 总结 |
| M0 learned recovery head | Scale mismatch + crop design，3/4 stop criteria 命中 | `reacquisition-mainline-m0-failure` |
| Full CoTracker3 hybrid | head-to-head: Track-On2 总体更强，CoTracker3 局部信号不足 | `attempt0_reentry_head2head_decision` |

---

## 2. 四级实验阶梯概览

```
Phase 0 (hard prerequisite)
  ↓ GO
Phase 1: Oracle re-detection upper bound
  ↓ GO → Phase 2: Anchor Top-K recall + distractor classification
  ↓ STOP
Phase 3: Verifier disambiguation (GT trigger)
  ↓ GO → Phase 4: Re-detection-first training
  ↓ STOP
```

---

## 3. 各 Phase Stop Criteria

### Phase 0: DAVIS strided+original 协议锁死
- **Stop Criteria**: 如果 full DAVIS `strided+original` 仍然跑不通 → **STOP AT PHASE 0**，停止后续全部新主线实验
- **GO 条件**: 至少 Track-On2 或 CoTracker3 之一能完整跑通 full DAVIS strided+original，并产出 AJ/OA/delta_avg 指标

### Phase 1: Oracle re-detection upper bound
- **Stop Criteria**: 如果 long-occ AJ barely improves 且 re-entry error barely decreases → **STOP AT PHASE 1**
- **GO 条件**: GT exact reset 或 GT neighborhood oracle 在 long-occ (occ>=20) 子集上有明显 AJ 提升 (>3pp) 且 re-entry error 明显下降 (>5px)

### Phase 2: Anchor Top-K recall
- **Stop Criteria**: 如果 Top-K recall 不够高（Top-5 < 70% 在 re-entry 帧）→ **STOP AT PHASE 2**
- **GO 条件**: DINOv3 first-frame anchor matching 在 re-entry 帧上 Top-5 recall >= 70%，distractor 可分类且有pattern

### Phase 3: Verifier disambiguation
- **Stop Criteria**: 如果 verifier 不能稳定优于 raw similarity baseline → **STOP AT PHASE 3**
- **GO 条件**: verifier selector 稳定优于 raw similarity + oracle selector 提供可信上界

### Phase 4: Re-detection-first training
- **仅在 Phase 1+2+3 全通过时启动**
- 核心变化: 训练分布变更，而非架构变更

---

## 4. 现有资产盘点

### 4.1 Prediction Cache 状态

| Baseline | Cache 目录 | 协议 | 状态 |
|---|---|---|---|
| trackon2 | `trackon2_dinov3_davis_cache/` | first+input (256-space) | ✅ 存在 |
| trackon2 | strided+original | — | ❌ **不存在** |
| cotracker3_online | `cotracker3_online_davis_cache/` | first+input | ✅ 存在 |
| cotracker3_online | strided+original | `cotracker3_baseline_strided_original_cache/` | ⚠️ 目录存在但 `davis/` 子目录为空 |
| cotracker3_offline | `cotracker3_offline_davis_cache/` | repo-native strided | ✅ 30 npz 存在，需 bridge 导出 |
| cotracker3_offline | strided+original (Attempt0 schema) | — | ❌ 不存在 |

### 4.2 关键脚本

| 脚本 | 用途 | 状态 |
|---|---|---|
| `scripts/attempt0_eval_strided_original.py` | Track-On2 + CoTracker3 strided+original 评估 | ✅ 可用 |
| `scripts/attempt0_export_strided_original_cache.py` | bridge 导出 repo-native npz → unified schema | ✅ 可用 |
| `scripts/eval_global_retrieval.py` | DINOv3 first-frame anchor 全帧匹配 | ⚠️ 需确认兼容性 |
| `scripts/build_local_patch_verifier_dataset.py` | Verifier 训练数据构建 | ⚠️ 需确认兼容性 |
| `scripts/audit_topk_oracle_gap.py` | Top-K oracle gap 分析 | ✅ 可用 |
| `scripts/eval_long_occlusion_oracle_gap.py` | Long-occ oracle gap 评估 | ✅ 可用 |

### 4.3 已知风险

**Track-On2 strided+original 关键限制**（来自 `attempt0_eval_strided_original.py` 注释）：
> "Track-On2 uses fixed 384x512 input (from config) and cannot handle original video resolution."

这意味着：
- Track-On2 端到端 strided+original 可能需要先 resize 到 384×512 再处理
- 如果 `eval` 脚本强行传原始分辨率，Track-On2 行为未定义
- **Phase 0 需要首先验证：Track-On2 strided+original 是否能实际运行，如果不能则只能测 CoTracker3**

---

## 5. Phase 0 执行计划

### 5.1 工作项

1. **验证 Track-On2 strided+original 可行性**（smoke: 2-3 videos）
   - 命令: `python scripts/attempt0_eval_strided_original.py --model_name trackon2 ...`
   - 成功标准: 无 crash，产出 AJ/OA/delta_avg
   - 失败处理: 如果 Track-On2 跑不通，Phase 0 改为只测 CoTracker3

2. **导出 CoTracker3 offline repo-native npz → unified schema**
   - 命令: `python scripts/attempt0_export_strided_original_cache.py ...`
   - 使用 `cotracker3_offline_davis_cache/davis/cotracker3_offline/` 的 30 个 npz
   - 输出: `outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt`

3. **运行 full DAVIS strided+original 评估**
   - Track-On2: 如果 smoke 通过 → full 30 videos
   - CoTracker3 offline: 导出后直接评测

4. **回归测试: 指标 wrapper 正确性**
   - 验证 normalized coords、size-1 scaling、re-entry metrics 的统一口径

### 5.2 Phase 0 产出（7 个强制 artifacts）

| # | 文件名 | 内容 |
|---|---|---|
| 1 | `phase0_protocol_lock.md` | 协议锁定说明 |
| 2 | `phase0_protocol_lock.json` | 协议参数 (query_stride, metric_resolution_mode, normalization 等) |
| 3 | `phase0_metric_sanity.md` | 指标 sanity check |
| 4 | `phase0_metric_sanity.json` | 数值 sanity 结果 |
| 5 | `phase0_full_davis_metrics_summary.json` | 全部模型的 AJ/OA/delta_avg 对比 |
| 6 | `phase0_full_davis_per_video_metrics.json` | 每视频详细指标 |
| 7 | `phase0_decision.md` | GO/STOP 决策，STOP_AT_PHASE_0 / ADVANCE_TO_PHASE_1 |

---

## 6. Phase 1 预览（仅记录，待 Phase 0 通过后执行）

- **目标**: Oracle re-detection upper bound（GT exact reset + GT neighborhood oracle）
- **核心指标**: long-occ AJ delta、re-entry error delta
- **关键脚本**: `scripts/eval_long_occlusion_oracle_gap.py`
- **Stop Criteria**: 如果 long-occ AJ barely improves 且 re-entry error barely decreases → STOP AT PHASE 1

---

## 7. 禁止事项（重申）

1. ❌ 不要继续扩展 memory hygiene、soft retention、dual-memory 当主线
2. ❌ 不要继续扩展 M0 learned recovery head
3. ❌ 不要做 full CoTracker3 replacement/hybrid 当默认主线
4. ❌ 不要跳过 full DAVIS strided+original 去讲新故事
5. ❌ 不要 terminal log only — 每个 phase 必须有 md/json artifacts

---

## 8. 立即开始

下一动作: 执行 Phase 0 → 先做 Track-On2 strided+original smoke test，验证协议可通性。
