# Phase 0 — Metric Sanity Check (strided+original)

**日期**: 2026-06-17  
**协议**: `strided+original`

---

## 1. DAVIS pkl 可用性

| 检查项 | 结果 |
|---|---|
| pkl 路径 | `/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl` |
| 视频数量 | 30 |
| 视频 ID 列表 | goat, car-roundabout, motocross-jump, breakdance, drift-chicane, ... |
| embedded frames | ✅ 存在 (ndarray, uint8, range [0, 255]) |
| original sizes | varies: e.g. 480×854, 480×910, 480×1152 |

---

## 2. Smoke Results

| Baseline | AJ | OA | delta_avg | delta_4px | Cache |
|---|---|---|---|---|---|
| CoTracker3 online | 36.95 | 68.15 | 45.01 | 47.71 | ✅ 30 npz |
| CoTracker3 offline | 51.54 | 92.15 | 63.59 | 68.01 | ✅ 30 npz |
| Track-On2 | ❌ | ❌ | ❌ | ❌ | ❌ 0 npz (SystemExit) |

---

## 3. Re-Entry Oracle Metrics (strided+original)

| Baseline | n_reentry | Median px | Mean px | p95 px | Oracle Gap Mean |
|---|---|---|---|---|---|
| CoTracker3 online | 1385 | 3.86px | 14.73px | 74.95px | 14.73px |
| CoTracker3 offline | 1385 | 3.71px | 12.21px | 44.97px | 12.21px |

### Oracle 改善分布

| Oracle 改善程度 | CoTracker3 online | CoTracker3 offline |
|---|---|---|
| oracle > model by > 4px | 48.7% | 46.2% |
| oracle > model by > 16px | 14.3% | 11.5% |
| oracle > model by > 32px | 8.0% | 6.7% |

### Long-Occlusion (occ >= 20)

| Baseline | n_longocc20 | Median px | oracle_better_4px |
|---|---|---|---|
| CoTracker3 online | 146 | 6.39px | 68.5% |
| CoTracker3 offline | 146 | 5.58px | 68.5% |

---

## 4. Metric Wrapper 正确性

| 检查项 | 结果 |
|---|---|
| Normalization | yx-normalized, denominator = (H-1, W-1) per video ✅ |
| Re-entry 识别 | first visible frame after occlusion ✅ |
| Coordinate space | original resolution (not 256-space) ✅ |
| model_input_size | = original_size for CoTracker3 ✅ |

---

## 5. 结论

Metric wrapper 正确。`strided+original` 协议下 CoTracker3 re-entry oracle gap ~13-15px mean，~47-49% of re-entry queries 有 >4px oracle gap。

---
