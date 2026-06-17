# Phase 0 — Protocol Lock (strided+original)

**日期**: 2026-06-17  
**协议**: `strided+original` (confirmed runnable)  
**状态**: ✅ LOCKED

---

## 1. DAVIS pkl 位置

| 路径 | 状态 |
|---|---|
| `/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl` | ✅ 30 videos + embedded frames |

---

## 2. 协议参数

| 参数 | 值 |
|---|---|
| query_mode | `strided` (query_stride=5) |
| metric_resolution_mode | `original` (actual video dimensions, varies per video) |
| normalization | `yx-normalized`, denominator = `(H-1, W-1)` per video |
| model_input_size | = original_size for all baselines (verified via schema audit) |

---

## 3. Baseline 状态

| Baseline | Status | Reason |
|---|---|---|
| CoTracker3 online (cotracker3_video) | ✅ Complete | model_input_size = original_size |
| CoTracker3 offline (cotracker3_window) | ✅ Complete | model_input_size = original_size |
| Track-On2 DINOv3 | ✅ Complete | dtype mismatch bug fixed; full 30-video strided+original run completed 2026-06-18 |

---

## 4. Cache 文件

| Baseline | npz cache | unified .pt cache |
|---|---|---|
| cotracker3_online | `caches/davis/cotracker3_video/00000X.npz` (30 files) | `caches/cotracker3_online_strided_original.pt` ✅ |
| cotracker3_offline | `caches/davis/cotracker3_window/00000X.npz` (30 files) | `caches/cotracker3_offline_strided_original.pt` ✅ |
| trackon2 | `caches/trackon2_strided_original.pt` ✅ | Full 30-video run completed, unified cache exported |

---

## 5. Phase 0 Stop Criteria

| Criteria | 判定 |
|---|---|
| full DAVIS strided+original 跑不通 | ❌ **NOT HIT** — CoTracker3 online/offline 均已完整运行 |

---

## 6. 决策

**✅ GO — ADVANCE TO PHASE 1**

CoTracker3 online + offline + Track-On2 全部完成 strided+original 30-video eval。统一 cache 已导出，schema 验证通过。

---
