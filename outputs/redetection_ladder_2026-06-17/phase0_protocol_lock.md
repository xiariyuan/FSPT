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
| model_input_size | = original_size for CoTracker3; needs rerun for Track-On2 after dtype bug fix |

---

## 3. Baseline 状态

| Baseline | Status | Reason |
|---|---|---|
| CoTracker3 online (cotracker3_video) | ✅ Complete | model_input_size = original_size |
| CoTracker3 offline (cotracker3_window) | ✅ Complete | model_input_size = original_size |
| Track-On2 DINOv3 | ⚠️ Needs rerun | dtype mismatch bug (已由用户修复)，修复后可运行，但完整 30-video run 尚未完成 |

---

## 4. Cache 文件

| Baseline | npz cache | unified .pt cache |
|---|---|---|
| cotracker3_online | `caches/davis/cotracker3_video/00000X.npz` (30 files) | `caches/cotracker3_online_strided_original.pt` ✅ |
| cotracker3_offline | `caches/davis/cotracker3_window/00000X.npz` (30 files) | `caches/cotracker3_offline_strided_original.pt` ✅ |
| trackon2 | ⚠️ probe (14 npz) | 修复后可运行，完整 run 未完成 |

---

## 5. Phase 0 Stop Criteria

| Criteria | 判定 |
|---|---|
| full DAVIS strided+original 跑不通 | ❌ **NOT HIT** — CoTracker3 online/offline 均已完整运行 |

---

## 6. 决策

**✅ GO — ADVANCE TO PHASE 1**

CoTracker3 online + offline 完整跑通 strided+original。Track-On2 有 dtype bug 但修复后可运行，需 rerun。

---
