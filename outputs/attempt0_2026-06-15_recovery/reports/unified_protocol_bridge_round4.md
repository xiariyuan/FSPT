# Unified Protocol Bridge — Round 4 (2026-06-15)

## Purpose

This document records the exact coordinate/format conversions required to bridge repo-native prediction caches (`.npz`) to the Attempt 0 unified schema (`.pt` cache).

---

## Coordinate Semantics Table

### Repo-Native Cache (`.npz`)

| Field | Shape | Format | Space | Notes |
|---|---|---|---|---|
| `tracks` | `(B, T, N, 2)` | `[x, y]` | pixel (0–255), 256×256 input | From `evaluate_tapvid` caching |
| `visibility` | `(B, T, N)` | bool | — | True=visible |
| Video resolution | — | — | 256×256 (fixed resize) | `resize_to_256=True` |

### Repo-Native TAP-Vid Dataset (`.pkl`)

| Field | Shape | Format | Space | Notes |
|---|---|---|---|---|
| `query_points` | `(B, N, 3)` | `[t, y, x]` | pixel (0–255) | `queried_first=True` |
| `target_points` | `(B, N, T, 2)` | `[x, y]` | pixel (0–255) | `resize_to_256=True` |
| `occluded` | `(B, N, T)` | bool | — | True=occluded |

### Repo-Native Evaluator (`eval_utils.compute_tapvid_metrics`)

| Property | Value |
|---|---|
| Tracks format | `[x, y]` raster pixel space |
| Query format | `[t, y, x]` pixel space |
| Evaluation space | 256×256 (fixed) |
| Occlusion convention | `occluded=True` means occluded |

### Attempt 0 Unified Schema (`utils/attempt0_schema.py`)

| Field | Shape | Format | Space | Notes |
|---|---|---|---|---|
| `pred_tracks` | `(N, T, 2)` | `[y, x]` | normalized (0–1) by model input size |
| `gt_tracks` | `(N, T, 2)` | `[y, x]` | normalized (0–1) by model input size |
| `pred_visibility` | `(N, T)` | bool | — | True=visible |
| `gt_visibility` | `(N, T)` | bool | — | True=visible |
| `query_points` | `(N, 3)` | `[t, y, x]` | normalized (0–1) by model input size |
| `original_size` | `(H, W)` | int | actual video dims | For metric rescaling |
| `model_input_size` | `(H, W)` | int | — | Model input resolution |

### `datasets/metrics.compute_tapvid_metrics` (Unified Rescorer)

| Property | Value |
|---|---|
| Tracks format | `[y, x]` normalized (0–1) |
| Query format | `[t, y, x]` normalized |
| Resolution mode `input` | Use `model_input_size` for scaling |
| Resolution mode `original` | Use `original_size` for scaling |
| Occlusion convention | `visibility=True` means visible |

---

## Required Transformations for `first + input` Bridge

### Predicted Tracks

1. **Remove batch dimension**: `(1, T, N, 2)` → `(T, N, 2)`
2. **Transpose to `(N, T, 2)`**: `tracks.transpose(1, 2)` or `np.transpose(..., (1, 2, 0))`
3. **Swap xy → yx**: `tracks[..., [1, 0]]` (each point becomes `[y, x]`)
4. **Normalize by input size**: `tracks_yx / 255.0` (since `resize_to_256=True`)
5. **Result**: `(N, T, 2)`, normalized `[y, x]`, range `[0, 1]`

### Visibility

1. **Remove batch dimension**: `(1, T, N)` → `(T, N)`
2. **Transpose to `(N, T)`**: `visibility.transpose(0, 1)`
3. **Invert**: `~visibility` (npz True=visible → schema True=visible — no change needed since both conventions are the same)
4. **Result**: `(N, T)`, bool, True=visible

### Query Points

The query points stored in the GT data from the evaluator are in `[t, y, x]` pixel space (0–255). We need to:

1. **Keep `[t, y, x]` order** (already correct)
2. **Normalize by input size (256)**: `query_points / 255.0`
3. **Result**: `(N, 3)`, `[t, y, x]`, normalized

### Ground Truth Tracks

The GT tracks from the evaluator are in `(B, N, T, 2)` `[x, y]` pixel space:

1. **Remove batch dimension**: `(B, N, T, 2)` → `(N, T, 2)`
2. **Swap xy → yx**: `gt_tracks[..., [1, 0]]`
3. **Normalize by 256**: `gt_yx / 255.0`
4. **Result**: `(N, T, 2)`, normalized `[y, x]`

### Ground Truth Visibility

1. **Remove batch dimension**: `(B, N, T)` → `(N, T)`
2. **GT occlusion is already boolean** (True=occluded)
3. **Invert to visibility**: `~gt_occluded` → True=visible
4. **Result**: `(N, T)`, bool, True=visible

### Metadata

- `original_size`: store actual original video dimensions (H, W) from the video loader
- `model_input_size`: store (256, 256) since `resize_to_256=True`
- `adapter_version`: record source query protocol, space, and track format

---

## Normalization Denominators

| Source field | Raw value range | Normalization denominator | Result range |
|---|---|---|---|
| `tracks` (pixel x,y) | 0–255 | 255 | 0–1 |
| `gt_tracks` (pixel x,y) | 0–255 | 255 | 0–1 |
| `query_points` (pixel t,y,x) | 0–255 | 255 | 0–1 |

**Critical note**: The `first + input` bridge stores unified-cache coordinates normalized by the 256-input raster, so pixel-space values from repo-native evaluation must be normalized by `255` before entering the unified schema. Earlier debugging also surfaced large anchor errors in a separate broken cache path, but the final verified CoTracker3 bridge mismatch discussed in Round 5 was not caused by the exporter; it was caused by the normalized-coordinate rescaling heuristic in `datasets/metrics.py`.

---

## Conversion Summary

```
Repo-native .npz (from evaluate_tapvid caching):
  tracks:       (1, T, N, 2)  [x, y]  pixel 0-255
  visibility:   (1, T, N)     bool    True=visible

Repo-native GT (from evaluator):
  gt_tracks:    (B, N, T, 2) [x, y]  pixel 0-255
  gt_occluded:  (B, N, T)    bool    True=occluded
  query_points: (B, N, 3)    [t, y, x] pixel 0-255

Attempt 0 unified cache (target):
  pred_tracks:  (N, T, 2)    [y, x]  normalized 0-1
  pred_visibility: (N, T)  bool    True=visible
  gt_tracks:    (N, T, 2)    [y, x]  normalized 0-1
  gt_visibility: (N, T)      bool    True=visible
  query_points: (N, 3)       [t, y, x] normalized 0-1
  original_size: (H, W)      int     actual video dims
  model_input_size: (256, 256) int
```

## Bridge Protocol Labels

| Source | Query Protocol | Space | Bridge Label |
|---|---|---|---|
| `track_on` / `cotracker3` | `first` (queried_first) | `input` (256) | `first_input` |
| Future: `track_on` / `cotracker3` | `strided` | `original` | `strided_original` |

The label `first_input` in cache names indicates this bridge's provenance.
