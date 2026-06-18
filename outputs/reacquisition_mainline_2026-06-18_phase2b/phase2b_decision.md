# Phase 2b — Candidate Generation Redesign: Complete Decision

**日期**: 2026-06-18  
**协议**: `strided+original`

---

## 1. 执行清单总览

| # | 任务 | 状态 | 结论 |
|---|---:|---:|---|
| 1 | 冻结当前 Phase 2 | ✅ | 已在 `phase2_decision.md` 标记 `STOP_AT_PHASE_2_CURRENT_VARIANT` |
| 2 | 建立 Phase 2b 主线 | ✅ | `outputs/reacquisition_mainline_2026-06-18_phase2b/` |
| A | CoTracker3 offline candidate audit | ✅ | median 3.2px re-entry error; 83.3% < 8px; 强 candidate generator |
| B | Last-visible multi-support retrieval | ✅ | 所有变体 top5@16px < 10% → **STOP** |
| C | Local-global hybrid candidate generation | ✅ | 所有变体 top5@16px < 10% → **STOP** |
| D | Verifier gate | ✅ | 已写入 gate criteria — verifier 不得在 candidate recall 达标前启动 |
| E | 文档修正 | ✅ | 见 `phase2b_manifest_patch.md` |
| F | 统一 stop/go 门槛 | ✅ | 见 `phase2b_gate_criteria.json` |

---

## 2. 实验结果汇总

### Task A: CoTracker3 offline 作为 candidate source

| 指标 | 值 |
|---|---|
| n_reentry queries | 2736 |
| Median re-entry error | **3.2 px** |
| < 8px | **83.3%** |
| < 16px | 91.2% |
| < 32px | 95.0% |
| Better than Track-On2 | 85.7% |
| When Track-On2 > 64px, CT-offline < 8px | 82.0% |

**结论**: CoTracker3 offline 是**强大的 coarse prior**。它自身不是最终输出（仅 61% < 4px），但提供极好的搜索起点。

### Task B: 多帧 support anchor → whole-frame retrieval

| 变体 | top5@16px | top5@32px | 判定 |
|---|---:|---:|---|
| query_frame_anchor (baseline) | 1.6% | 3.1% | ❌ STOP |
| single_last_visible | 0.0% | 3.1% | ❌ STOP |
| multi_support_mean | 2.3% | 8.6% | ❌ STOP (< 10%) |
| multi_support_max | 1.6% | 8.6% | ❌ STOP (< 10%) |

**结论**: 纯 DINOv2 整帧检索是死路。anchor frame 与 re-entry frame 之间的外观变化超出了 DINOv2 template matching 的胜任范围。

### Task C: CoTracker3 offline → 局部搜索

| 变体 | top5@16px | top5@32px | 判定 |
|---|---:|---:|---|
| global_whole_frame (baseline) | 0.0% | 7.0% | ❌ STOP |
| global_topk_then_local_refine | 7.0% | 15.6% | ❌ STOP (16px < 10%) |
| center_ct_offline | 7.0% | **28.1%** | ❌ STOP (16px < 10%) |
| center_dual | 7.0% | 18.0% | ❌ STOP (16px < 10%) |

**结论**: CoTracker3 offline prior 确实缩小了搜索空间（center_ct_offline top5@32px=28.1% 是唯一正的信号），但仍然无法在 16px 精度下达到 10% recall。DINOv2 跨帧 patch matching 的表征力不够。

---

## 3. 核心洞察

### 3.1 两条矛盾的线条终于清晰

```
线条 A) CoTracker3 offline 自身是优秀 candidate generator (83.3% < 8px)
线条 B) DINOv2 template matching 无论如何改 anchor 都无法利用这个 prior

结论: 问题不在 prior/coarse seed，而在 visual feature matching 本身。
```

### 3.2 DINOv2 template matching 为什么失败

在 `strided+original` 协议下，从 query-frame 到 re-entry frame 的时间跨度大，patch 的外观变化包括：
- 形变（非刚性物体）
- 光照变化
- 遮挡后重新出现时的姿态变化

DINOv2 ViT-S/14 的 patch-level features 在这样的外观变化下不具备足够的判别力来精确对应。

### 3.3 这问的不是"怎么匹配"，而是"学什么特征"

手写 DINOv2 跨帧匹配到极限了。如果这条路要继续，需要一个**在遮挡/重出现场景下专门训练的特征提取器**。

---

## 4. 递交给用户的关键问题

1. **当前 DINOv2 template matching 已到极限。** 如果继续 Phase 2 方向，需要的不是调参数，而是换方法——例如 learned verifier with local features。

2. **CoTracker3 offline 本身已经可以作为线。** 它 83% 的情况 < 8px，高于任何 template matching 变体。Phase 3 的 verifier 可以直接以 CoTracker3 offline 的 prediction 为输入，而不是从全图检索开始。

3. **or 停止 Phase 2 转而评估其他方法。** 如果验证了当前 family 没有希望，可以把能量放到更有可能的方向上。

---

---

## 5. 当前冻结状态

```
DINOv2 template matching (全图/多锚点/局部混合): ❌ STOP — 全部变体 top5@16px < 10%
CoTracker3 offline as coarse prior: ✅ VALIDATED — median 3.2px, 83.3% < 8px
Oracle local refinement audit: ✅ PASS — 3/3 radii 双阈值通过
下一动作: learned verifier / local refiner (CT-offline-centered, local crop only)
```

**明确关闭的分支（不再投入）:**
1. DINOv2 全图检索 → STOP
2. DINOv2 多锚点检索 → STOP
3. DINOv2 局部混合检索 → STOP
4. 任何以 whole-frame retrieval 为基础的 verifier → STOP

**当前活跃分支:**
1. ~~CT-offline-centered oracle local refinement audit~~ ✅ 已完成
2. Learned verifier / local refiner (下一轮)

---

## 6. Oracle Local Refinement 结果

Smoke: 5 videos / 128 queries

### Baseline: CoTracker3 Offline Raw at Re-entry
- Median: 4.0px
- <4px: 49.2%
- <8px: 84.4%

### Oracle Best-of-K Local Grid

| Radius | Oracle median | Oracle <4px | <4px improv | Long-occ <4px improv | 判定 |
|---|---:|---:|---:|---:|---:|
| 8px | 0.8px | 89.1% | **+39.8pp** | **+85.0pp** | ✅ PASS |
| 16px | 0.8px | 91.4% | **+42.2pp** | **+85.0pp** | ✅ PASS |
| 32px | 0.8px | 97.7% | **+48.4pp** | **+90.0pp** | ✅ PASS |

**结论: Oracle headroom 极大。以 CT-offline prediction 为中心、半径 8-32px 的局部搜索格点，通过 oracle best-of-K 选择可将 <4px 从 49% 提升到 89-98%。下一轮进入 learned verifier / local refiner。**

---

## 7. 下一步：Learned Stage 设计原则

```
输入:
  - CT-offline-centered local crop (radius 8-32px)
  - 不再做 whole-frame retrieval / DINOv2 top-k matching

方法:
  - learned local refiner (轻量网络，以 crop 为输入，微调预测位置)
  - 或 learned verifier (从候选格点中选择最佳)

门槛:
  - 先 smoke (5 videos / 128 queries)，再决定是否 full DAVIS
  - 目标: <4px 达到 oracle 的 80-90% (即从 ~49% baseline 提升到 ~75-85%)
```
