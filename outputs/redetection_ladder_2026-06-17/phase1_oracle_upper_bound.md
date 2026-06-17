# Phase 1 — Oracle Re-Detection Upper Bound (strided+original)

**日期**: 2026-06-17  
**协议**: `strided+original`  
**状态**: ✅ Phase 1 Partial — CoTracker3 online/offline 完成，Track-On2 未完成

---

## 1. Oracle 定义

| Oracle 类型 | 定义 |
|---|---|
| Oracle-exact | GT exact position at re-entry frame (error = 0 by definition) |

**Oracle gap** = model error at re-entry frame (因为 oracle = 0)

---

## 2. 结果汇总

### 2.1 Re-Entry Oracle (strided+original)

| Baseline | n_reentry | Model Median | Model Mean | Model p95 | Oracle Gap Mean | oracle_better_4px | oracle_better_16px |
|---|---|---|---|---|---|---|---|
| CoTracker3 online | 1385 | 3.86px | 14.73px | 74.95px | **14.73px** | **48.7%** | **14.3%** |
| CoTracker3 offline | 1385 | 3.71px | 12.21px | 44.97px | **12.21px** | **46.2%** | **11.5%** |

### 2.2 All Visible Frames

| Baseline | n_frames | Median | Mean | p95 |
|---|---|---|---|---|
| CoTracker3 online | 173164 | 2.46px | 5.64px | ~30px |
| CoTracker3 offline | 173164 | 2.43px | 5.30px | ~28px |

### 2.3 Long-Occlusion (occ >= 20)

| Baseline | n_longocc | Median | oracle_better_4px |
|---|---|---|---|
| CoTracker3 online | 146 | 6.39px | **68.5%** |
| CoTracker3 offline | 146 | 5.58px | **68.5%** |

---

## 3. Stop Criteria 对照

| Stop Criteria | 实测 | 判定 |
|---|---|---|
| long-occ AJ barely improves | CoTracker3 offline oracle gap mean ≈ 12px (long-occ median ≈ 6px) | ⚠️ **NOT barely** |
| re-entry error barely decreases | CoTracker3 offline oracle gap mean = 12.21px | ⚠️ **NOT barely** |

---

## 4. 核心信号

1. **~47-49% of re-entry queries 有 >4px oracle gap**：re-detection 有显著 headroom
2. **Long-occ subset oracle gap 更显著**：68.5% long-occ queries 有 >4px gap
3. **CoTracker3 offline 优于 online**：re-entry mean 12.21px vs 14.73px（与 AJ 差异一致）
4. **Oracle gap 集中在 tail**：median ~4px 但 p95 ~45-75px

---

## 5. 局限性

| 局限 | 详情 |
|---|---|
| Track-On2 未完成 | 只测了 CoTracker3，无法得出 Track-On2 的 oracle gap |
| 样本有限 | long-occ n=146 (occ>=20)，统计方差大 |
| Partial completion | 仅 CoTracker3 baseline 完成，未覆盖 Track-On2 |

---

## 6. Phase 1 结论

| 结论 | 理由 |
|---|---|
| **ADVANCE TO PHASE 2** | oracle gap ≈ 12-15px mean，47-49% re-entry queries 有 >4px gap |
| **标注 Partial** | 仅 CoTracker3，未含 Track-On2 |

---

*Phase 1 partial complete on CoTracker3 strided+original. Advancing to Phase 2 with caveats.*
