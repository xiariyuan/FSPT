# Phase 1 — Oracle Re-Detection Upper Bound (strided+original)

**日期**: 2026-06-18  
**协议**: `strided+original`  
**状态**: ✅ COMPLETE — CoTracker3 online/offline + Track-On2 完成

---

## 1. Oracle 定义

| Oracle 类型 | 定义 |
|---|---|
| Oracle-exact | GT exact position at first re-entry frame (error = 0 by definition) |

**Oracle gap** = model error at first re-entry frame (因为 oracle = 0)

---

## 2. 结果汇总

### 2.1 Re-Entry Oracle (strided+original)

| Baseline | n_reentry | Model Median | Model Mean | Model p95 | Oracle Gap Mean | model <4px | model <16px |
|---|---|---|---|---|---|---|---|
| CoTracker3 online | 2736 | 11.65px | 61.56px | 267.04px | **61.56px** | **33.6%** | **53.8%** |
| CoTracker3 offline | 2736 | 3.15px | 9.24px | 31.56px | **9.24px** | **61.0%** | **91.2%** |
| Track-On2 | 2736 | 405.04px | 318.60px | 679.90px | **318.60px** | **20.4%** | **25.0%** |

### 2.2 All Visible Frames

| Baseline | n_frames | Median | Mean | p95 |
|---|---|---|---|---|
| CoTracker3 online | 337116 | 4.70px | 39.68px | 194.29px |
| CoTracker3 offline | 337116 | 2.36px | 5.08px | 13.87px |
| Track-On2 | 337116 | 18.55px | 233.47px | 661.11px |

### 2.3 Long-Occlusion (occ >= 20)

| Baseline | n_longocc | Median | model <4px |
|---|---|---|---|
| CoTracker3 online | 488 | 35.88px | **29.1%** |
| CoTracker3 offline | 488 | 3.48px | **55.7%** |
| Track-On2 | 488 | 413.48px | **8.8%** |

---

## 3. Stop Criteria 对照

| Stop Criteria | 实测 | 判定 |
|---|---|---|
| long-occ AJ barely improves | Track-On2 long-occ oracle gap mean ≈ 391.51px | ❌ 未命中 |
| re-entry error barely decreases | Track-On2 re-entry oracle gap mean = 318.60px | ❌ 未命中 |

---

## 4. 核心信号

1. **Track-On2 re-entry gap 非常大**：说明 re-detection headroom 仍然很高。
2. **Offline oracle 仍有明显空间**：CoTracker3 offline 不是饱和状态。
3. **Long-occ subset 的尾部更重**：tail 仍然是主要难点。
4. **Phase 2 不应被旧版 first+input 口径污染**：本文件已改为 strided+original 唯一口径。

---

## 5. Phase 1 结论

| 结论 | 理由 |
|---|---|
| **ADVANCE TO PHASE 2** | oracle gap 仍然显著，且 Track-On2 已完成完整 run |

---

*Phase 1 complete on strided+original. Phase 2 is permitted.*
