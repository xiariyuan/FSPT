# Phase 0 — Metric Sanity Check (strided+original)

**日期**: 2026-06-18  
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
| Track-On2 | 28.38 | 58.33 | 33.23 | 35.90 | ✅ 30 npz |

---

## 3. Re-Entry Oracle Metrics (strided+original)

| Baseline | n_reentry | Median px | Mean px | p95 px | Oracle Gap Mean |
|---|---|---|---|---|---|
| CoTracker3 online | 2736 | 11.65px | 61.56px | 267.04px | 61.56px |
| CoTracker3 offline | 2736 | 3.15px | 9.24px | 31.56px | 9.24px |
| Track-On2 | 2736 | 405.04px | 318.60px | 679.90px | 318.60px |

### Re-Entry Threshold Rates

| 指标 | CoTracker3 online | CoTracker3 offline | Track-On2 |
|---|---|---|---|
| model <4px | 33.6% | 61.0% | 20.4% |
| model <16px | 53.8% | 91.2% | 25.0% |
| model <32px | 60.1% | 95.0% | 33.4% |

### Long-Occlusion (occ >= 20)

| Baseline | n_longocc20 | Median px | model <4px |
|---|---|---|---|
| CoTracker3 online | 488 | 35.88px | 29.1% |
| CoTracker3 offline | 488 | 3.48px | 55.7% |
| Track-On2 | 488 | 413.48px | 8.8% |

---

## 4. Metric Wrapper 正确性

| 检查项 | 结果 |
|---|---|
| Normalization | yx-normalized, denominator = (H-1, W-1) per video ✅ |
| Re-entry 识别 | first visible frame after occlusion ✅ |
| Coordinate space | original resolution (not 256-space) ✅ |
| model_input_size | = original_size for all baselines ✅ |

---

## 5. 结论

Metric wrapper 正确。`strided+original` 协议下 re-entry oracle gap 仍然显著，且 Track-On2 已完整完成 30-video run，不存在 `needs_rerun`。
