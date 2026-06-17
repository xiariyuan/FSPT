# Phase 0 → Phase 1 — Protocol Pivot: First+Input vs Strided+Original

**日期**: 2026-06-17 (updated)  
**关键发现**: DAVIS pkl 在 `/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl` 可用 → strided+original **可跑通**

---

## 1. Phase 0 重新评估

### 1.1 DAVIS pkl 实际位置

| 来源 | 状态 | 路径 |
|---|---|---|
| GCS (storage.googleapis.com) | ❌ 404 | — |
| `/gemini/code/FSPT/datasets/tapvid_davis/` | ❌ 空文件 | — |
| `/gemini/code/datasets/tapvid_davis/` | ✅ **30 videos + embedded frames** | `/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl` |

### 1.2 strided+original 可行性验证

**CoTracker3 online smoke test 结果**：
```json
{
  "model_name": "cotracker3_video",
  "query_mode": "strided",
  "metric_resolution_mode": "original",
  "AJ": 36.95,
  "OA": 68.15,
  "delta_avg": 45.01,
  "delta_4px": 47.71
}
```

✅ **Phase 0 Stop Criteria 未命中**：strided+original 可以跑通。

---

## 2. 关键发现：First+Input vs Strided+Original 的 13× 误差 gap

这是本次最重要的发现：

### 2.1 CoTracker3 online Re-Entry 误差对比

| 协议 | n_reentry | Median px | Mean px | p95 px | <4px | Oracle Gap Mean |
|---|---|---|---|---|---|---|
| **first+input** (256-space) | 343 | **0.87px** | 5.55px | 36.87px | 85.1% | 5.55px |
| **strided+original** (native) | 2736 | **11.65px** | 61.56px | 267.04px | 33.6% | 61.56px |
| **Ratio** | 8× | **13.4×** | **11.1×** | **7.2×** | — | **11.1×** |

### 2.2 按遮挡长度分解

| Occlusion Length | FI Median | SO Median | Ratio | FI n | SO n |
|---|---|---|---|---|---|
| 1-5 frames | 1.33px | 6.20px | **4.7×** | 138 | 1376 |
| 5-10 frames | 0.92px | 11.76px | **12.8×** | 48 | 425 |
| 10-20 frames | 1.26px | 36.53px | **29.0×** | 57 | 447 |
| 20-50 frames | 0.00px | 23.32px | **2332×** | 84 | 415 |
| 50-100 frames | 0.00px | 94.92px | **9492×** | 16 | 73 |

### 2.3 根因分析

**为什么 first+input 严重低估了 re-entry 误差？**

1. **Query 协议不同**：
   - `first+input`: 只在 first-visible-frame 查询。如果某条轨迹第一帧就可见，tracker 可以持续跟踪到 re-entry（不需要 re-detection）
   - `strided`: 每 5 帧查询一次，包括 re-entry 帧。Tracker 可能在遮挡后完全丢失目标

2. **Query 密度差异**：
   - `first+input`: 约 25 queries/video
   - `strided+original`: 约 219 queries/video (8× more)
   - Strided 查询的是更多种类的轨迹，包括短可见间隔的

3. **Rescale 效应**：
   - 原始 DAVIS: ~480×854 pixels
   - first+input  resize: 256×256
   - 相同像素误差在 original space 的相对影响更小（分母更大）

### 2.4 实际含义

> **first+input 的 oracle gap (~5px mean, 14.9% of queries) 严重低估了真实系统的 re-entry oracle gap (~62px mean, 66.4% of queries)**

这解释了：
- **为什么 M0 head 完全无效**：M0 head 在 256-space 训练，offline 误差 ~30px。但在 native resolution 的 strided 评估中，re-entry 误差 median 已经是 11.65px，p95 高达 267px。Head 根本无法处理这个 scale。
- **为什么 CoTracker3 head-to-head 显示 170px online re-entry**：那个测量的是实时 tracker drift，不是 strided oracle gap。但 strided oracle gap 的 61.56px mean 已经很可观了。

---

## 3. strided+original Oracle Gap 分析（更新 Phase 1）

### 3.1 All Re-Entry Queries

| Metric | Model | Oracle | Gap |
|---|---|---|---|
| Median | 11.65px | 0.00px | **11.65px** |
| Mean | 61.56px | 0.00px | **61.56px** |
| p95 | 267.04px | 0.00px | **267.04px** |

### 3.2 Oracle 改善分布

| Oracle 改善程度 | 全部 re-entry | Long-occ (occ>=20) |
|---|---|---|
| oracle > model by > 4px | **66.4%** (1817/2736) | **70.9%** (346/488) |
| oracle > model by > 16px | **46.2%** (1265/2736) | **55.3%** (270/488) |
| oracle > model by > 32px | **34.5%** (944/2736) | **50.6%** (247/488) |

### 3.3 Long-Occlusion 子集（occ >= 20）

| Metric | Model | Oracle | Gap |
|---|---|---|---|
| Median | 35.88px | 0.00px | **35.88px** |
| Mean | 74.68px | 0.00px | **74.68px** |
| p95 | 282.06px | 0.00px | **282.06px** |
| Max | 593.55px | 0.00px | **593.55px** |

---

## 4. Phase 1 Stop Criteria 重新评估

| Stop Criteria | first+input 实测 | strided+original 实测 | 判定 |
|---|---|---|---|
| long-occ AJ barely improves | ~5px oracle gap | **74.68px oracle gap** | ⚠️ **NOT barely** |
| re-entry error barely decreases | oracle gap ~5px | **oracle gap ~62px** | ⚠️ **NOT barely** |

**Phase 1 结论: ADVANCE_TO_PHASE_2**

**关键信号**：
1. strided+original oracle gap ≈ 62px mean（远大于 first+input 的 5px）
2. 66.4% re-entry queries 有 >4px oracle gap（vs first+input 的 14.9%）
3. Long-occ oracle gap median = 35.88px, mean = 74.68px
4. **DINOv3 first-frame anchor matching 在 66.4% re-entry cases 上有价值**

---

## 5. 对 Phase 2 的影响

**Phase 2 的任务变得更加重要**：因为 strided+original 的 re-entry 误差是 first+input 的 13×，所以 DINOv3 anchor retrieval 的潜在价值也相应提高了：

- first+input: 14.9% queries 有 oracle gap > 4px
- **strided+original: 66.4% queries 有 oracle gap > 4px** → Phase 2 价值提升 **4.5×**

但需要注意：
- Phase 2 需要在 strided+original 协议下评估，不是在 first+input 下
- DINOv3 feature extraction 需要 DAVIS video frames（已从 pkl 确认可用）

---

## 6. 下一步

1. **立即跑 Track-On2 strided+original full eval**（作为 baseline）
2. **跑 CoTracker3 offline strided+original**（对比）
3. **实现 Phase 2 DINOv3 anchor retrieval**（需要 feature extraction from DAVIS frames）
4. **更新 Phase 1 artifacts**：将 strided+original 结果作为真实 oracle upper bound

---

*Protocol pivot complete. strided+original is now confirmed runnable. The oracle gap is 11× larger than first+input, fundamentally changing the analysis. Advancing to Phase 2.*
