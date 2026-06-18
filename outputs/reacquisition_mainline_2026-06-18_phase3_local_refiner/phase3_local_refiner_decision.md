# Phase 3 — CT-Offline Local Refiner: Final Decision

**日期**: 2026-06-18  
**状态**: ❌ **STOP — learned local refiner fails to beat CT-offline baseline**

---

## 1. 完整路径回顾

```
Phase 0/1: Strided+original protocol locked, oracle measured
  ↓ PASS
Phase 2: DINOv2 whole-frame anchor Top-K recall (top5@16px=1.6%)
  ↓ STOP
Phase 2b-A: CoTracker3 offline candidate audit (validated as coarse prior)
  ↓ PASS
Phase 2b-B: Last-visible multi-support retrieval (top5@16px<10%)
  ↓ STOP
Phase 2b-C: Local-global hybrid DINO retrieval (top5@16px<10%)
  ↓ STOP
Phase 2b-Oracle: CT-offline-centered oracle local refinement (oracle <4px=94%)
  ↓ PASS (oracle headroom confirmed)
Phase 3: Learned local refiner (OnlineRecoveryHead on DINOv2 features)
  ↓ ❌ FAIL (head median 6.5px vs baseline 3.7px, better_frac=23.8%)
```

## 2. Phase 3 失败细节

### 训练结果

| 指标 | Learned Head | CT-offline Raw |
|---|---:|---:|
| Median | 6.51 px | **3.70 px** |
| <4px | 23.8% | **54.6%** |
| Better than baseline | 23.8% | — |

### Smoke 通过线（全部未达标）

| 门槛 | 要求 | 实测 |
|---|---:|---:|
| overall <4px | ≥ 70% | **23.8%** |
| overall median | ≤ 2.0px | **6.51px** |
| <8px 不退化 | 不退化 | **严重退化** |

## 3. 结论

```
Phase 2 主线: ❌ STOP
理由: 所有 DINOv2-based 方法均无法实现有效的 re-entry 候选定位
- 全图检索: top5@16px = 1.6%
- 局部精修 (learned): <4px = 23.8% (baseline 54.6%)

CoTracker3 offline raw prediction (median 3.7px, <4px=54.6%)
是当前 re-entry 定位的最佳可用信号。
```

## 4. 分支冻结状态

```
Phase 3 local refiner (learned offset regression): ❌ FAIL
DINOv2 retrieval family (whole-frame + multi-anchor + local-global): ❌ CLOSED
Active branch: CT-offline-centered local grid verifier (8px, stride 2px) — ⏳ 执行中
```

## 5. 下一步

Fallback: CT-offline-centered local grid verifier（离散选择，非连续回归）。
单次允许，范围严格受限。如果不过，整个方向永久关闭。

## 6. Verifier Smoke 结果

| 指标 | Verifier | CT-offline Raw | Oracle |
|---|---:|---:|---:|
| Median | 3.0px (val) | 3.0px | 0.8px |
| <4px | 71.2% | 71.2% | 98.5% |
| Better than baseline | — | 59.1% | — |

**结果**: Verifier 学不会区分候选。始终选择中心候选（= CT-offline），没有任何实际改善。

## 7. 最终决策

```
Phase 3 grid verifier (learned ranking): ❌ FAIL — 无法区分局部格点候选
Phase 2/3 re-entry 局部修正方向: ❌ 正式收尾

保留的唯一结论:
  raw CoTracker3 offline is the best available re-entry signal
  local learned correction failed under current DINOv2 feature family
```

