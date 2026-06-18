# Phase 2b — Task A: CoTracker3 Offline Candidate Source Audit

**日期**: 2026-06-18  
**协议**: strided+original  
**数据**: 30 DAVIS videos, 2736 re-entry queries

---

## 1. Core Question

> Can CoTracker3 offline serve as a candidate generator for re-entry re-detection?

## 2. Answer

**YES — as a coarse prior for local search. NOT as a standalone final output.**

---

## 3. Error Distribution

| Metric | Overall (n=2736) | Occ < 20 (n=2248) | Occ >= 20 (n=488) |
|---|---:|---:|---:|
| Median | 3.2 px | 3.1 px | 3.5 px |
| Mean | 9.2 px | 6.2 px | 22.8 px |
| P95 | 24.7 px | 20.6 px | 83.0 px |
| < 4px | 61.0% | 63.1% | 51.4% |
| < 8px | 83.3% | 85.2% | 74.6% |
| < 16px | 91.2% | 92.7% | 84.2% |
| < 32px | 95.0% | 95.9% | 90.8% |

## 4. Comparison with Track-On2

| Metric | CoTracker3 Offline | Track-On2 |
|---|---|---|
| Median | **3.2 px** | 6.5 px |
| < 8px | **83.3%** | 56.1% |
| Better fraction | **85.7%** | 14.3% |

### 4.1 When Track-On2 fails badly (>64px)

- n = 1780 queries (65% of all re-entry queries)
- CoTracker3 offline median = **3.0 px**
- CoTracker3 offline < 8px = **82.0%**

CoTracker3 offline is not just "better on average" — it specifically rescues the cases where Track-On2 catastrophically fails.

## 5. Error Histogram

| Bin | Count | % |
|---|---:|---:|
| 0-1 px | 345 | 12.6% |
| 1-2 px | 586 | 21.4% |
| 2-4 px | 739 | 27.0% |
| 4-8 px | 609 | 22.3% |
| 8-16 px | 216 | 7.9% |
| 16-32 px | 103 | 3.8% |
| 32-64 px | 67 | 2.4% |
| 64-128 px | 35 | 1.3% |
| 128-256 px | 20 | 0.7% |
| 256-512 px | 10 | 0.4% |
| 512-1024 px | 5 | 0.2% |
| 1024+ px | 1 | 0.04% |

## 6. Recommendation

### ✅ Use as:
- **Coarse prior for local search**: place a search window (e.g., 32-64px radius) around CoTracker3 offline prediction
- **Hybrid candidate generation**: combine with visual matching within the coarse region
- **Fallback anchor**: when Track-On2 fails, use CoTracker3 offline as the alternative prior

### ❌ Do NOT use as:
- **Standalone final output**: only 61% < 4px, insufficient for precise re-localization
- **Direct replacer**: Track-On2 still wins on some queries (14.3%)

## 7. Implication for Phase 2b

CoTracker3 offline provides a strong structural prior. The remaining 9-17% of queries where it's >16px off need a different approach (likely visual matching within the coarse region).

The natural next step is **local-global hybrid**: use CoTracker3 offline for the coarse region, then apply visual feature matching within that region to refine.
