# Phase 2 — Anchor Top-K Recall Decision

**日期**: 2026-06-18  
**协议**: `strided+original`  
**状态**: `STOP_AT_PHASE_2_CURRENT_VARIANT`

---

## 1. 本轮回答的问题

Phase 2 预注册问题是：

> DINOv3 / DINOv2 first-frame anchor matching 在 re-entry 帧上，是否能提供足够高的 Top-K candidate recall，支撑后续 verifier / selector 主线？

本轮不再沿用旧的 `first+input` 口径，而是全部切换到 **`strided+original`**。

---

## 2. 先修正一个历史前提

旧文档中的 `CoTracker3 Hybrid` 判断基于 `first+input` head-to-head，不能继续作为主线依据。

本轮用 `strided+original` 统一缓存重跑后，结论变为：

### 2.1 Re-entry head-to-head（真实口径）

| Model | n | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|---:|
| Track-On2 | 1385 | 3.91 | 18.47 | 51.0% | 73.6% | 109.77 |
| CoTracker3 online | 1385 | 3.86 | 14.73 | 51.3% | 77.0% | 74.95 |
| CoTracker3 offline | 1385 | **3.71** | **12.21** | **53.8%** | **80.2%** | **44.97** |

### 2.2 Long-occ re-entry（occ >= 20）

| Model | n | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|---:|
| Track-On2 | 146 | 5.70 | 32.65 | **37.7%** | 56.2% | 198.04 |
| CoTracker3 online | 146 | 6.39 | 31.15 | 31.5% | 58.2% | 226.24 |
| CoTracker3 offline | 146 | **5.58** | **22.10** | 31.5% | **69.9%** | **164.99** |

### 2.3 修正后的含义

1. `CoTracker3 offline` 在 **真实协议** 下总体优于 `Track-On2`。  
2. 旧的 “Track-On2 总体更强，因此 full CoTracker3 hybrid 不成立” 这句话 **不再成立**。  
3. 但这只说明 **更强 base model 值得保留**，不等于 Phase 2 anchor retrieval 已经成功。

---

## 3. Phase 2 当前变体：Anchor Top-K Recall Smoke

本轮新做的 smoke：

- 脚本: `scripts/eval_davis_anchor_topk_recall.py`
- 数据: `tapvid_davis`
- 协议: `strided+original`
- Anchor: `query-frame GT patch`
- Target: `first re-entry frame`
- 特征: `DINOv2 ViT-S/14`
- 检索方式: whole-frame dense matching
- Sample cap: `5 videos / 128 queries`

### 3.1 Smoke 结果

| 指标 | Overall | Long-occ (occ>=20) |
|---|---:|---:|
| n | 128 | 20 |
| top1 median px | 136.02 | 108.10 |
| top5 best median px | 85.05 | 85.02 |
| top1@8px | 0.0% | 0.0% |
| top5@8px | 1.6% | 0.0% |
| top1@16px | 0.0% | 0.0% |
| top5@16px | 1.6% | 0.0% |
| top1 miss but top5 hit @16px | 1.6% | 0.0% |

### 3.2 与预注册 Stop Criteria 对照

Phase 2 预注册 GO 条件：

> Top-5 recall >= 70% at re-entry

实测：

- `top5@16px = 1.6%`
- `top5@8px = 1.6%`
- long-occ 子集 `top5@16px = 0.0%`

**结论**：当前 anchor retrieval 变体远低于通过线，不具备继续扩大样本或进入 verifier 阶段的价值。

---

## 4. 失败归因

### Layer 1: 纯 first-frame / query-frame anchor 的表征不足

从 query frame 裁出的 patch 到 re-entry frame 的 whole-frame dense match，外观变化和背景干扰过大，GT 点通常根本不在 Top-5 候选中。

### Layer 2: 这不是“排序错了”，而是“候选没进来”

如果 `top1` 很差但 `top5` 很高，说明可以交给 verifier / selector。  
但当前结果是：

- `top1 miss but top5 hit @16px = 1.6%`

这说明 **Phase 3 verifier 现在也没有足够候选可选**。  
问题发生在 **candidate generation**，不是 candidate ranking。

### Layer 3: 当前 smoke 其实已经足够强

这不是一个边界性失败，而是数量级失败：

- 目标阈值 70%
- 实测 1.6%

差了两个数量级。继续把同一变体从 128 queries 扩到 full DAVIS，只会更稳地证明失败，不会改变决策。

---

## 5. 决策

### 5.1 明确停止的方向

**停止当前 Phase 2 变体：**

`DINO first/query-frame anchor -> whole-frame dense retrieval -> Top-K candidates`

理由：

1. Top-K recall 极低；
2. 候选集本身几乎不含 GT；
3. verifier / selector 没有足够可操作空间；
4. 不值得扩大同一变体样本规模。

### 5.2 保留的结论

1. `strided+original` 是唯一真实协议；
2. `CoTracker3 offline` 在 re-entry 上强于 `Track-On2`，更强 base model 仍然值得保留；
3. 当前问题的核心不是 “candidate ranking”，而是 “candidate generation”。

---

## 6. 下一步建议

建议切到 **Phase 2b / candidate generation redesign**，而不是继续当前 anchor retrieval：

| 优先级 | 方向 | 原因 |
|---|---|---|
| 1 | CoTracker3-offline-first candidate source | 真实协议下它本身比 Track-On2 更强 |
| 2 | last-visible / visible-bank multi-support anchor | first-frame 单锚明显不够 |
| 3 | local-global hybrid candidate generation | 全局纯检索失败，需引入结构先验 |
| 4 | 只有在 candidate recall 上来后，才做 verifier | 当前 Phase 3 没有输入价值 |

---

## 7. 最终结论

```text
Phase 2 current variant: STOP

What failed:
  DINO anchor Top-K whole-frame retrieval on strided+original DAVIS re-entry queries

What remains valid:
  CoTracker3 offline is stronger than Track-On2 on real re-entry protocol

What to do next:
  Redesign candidate generation, then revisit verifier only after Top-K recall becomes non-trivial
```

