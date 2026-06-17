# CoTracker3 Checkpoint Manifest

**Date:** 2026-06-15  
**Run Type:** Fresh本轮运行 (not historical snapshot)

## Repository

- **Path:** `/gemini/code/FSPT/baselines/cotracker/`

## Checkpoints

| Model | Path | Size | Load Method |
|-------|------|------|-------------|
| CoTracker3 Offline | `checkpoints/scaled_offline.pth` | 512KB | `CoTrackerPredictor(checkpoint=path, offline=True, v2=False, window_len=60)` |
| CoTracker3 Online | `checkpoints/scaled_online.pth` | 512KB | `CoTrackerPredictor(checkpoint=path, offline=False, v2=False, window_len=16)` |

## Eval Config

| Parameter | Value |
|-----------|-------|
| Dataset | TAP-Vid DAVIS |
| Dataset Path | `/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl` |
| Evaluator | `track_on/evaluation/evaluator.evaluate_tapvid` (TAPVid metrics) |
| Grid Size | 20 (via support_grid_size parameter) |
| Batch Size | 1 |
| Workers | 4 |

## Results (this round)

| Metric | CoTracker3 Offline | CoTracker3 Online |
|--------|-------------------|-------------------|
| AJ | 62.66 | 64.89 |
| δ_avg | 77.22 | 77.36 |
| OA | 88.15 | 91.80 |

## Attempt 0 Row Mapping

This section disambiguates which predictor variant maps to which Attempt 0 status row:

| Attempt 0 status row | Predictor variant | Checkpoint | DAVIS AJ | DAVIS δ_avg |
|---|---|---|---|---|
| `cotracker3_baseline` | `CoTrackerPredictor(offline=False, window_len=16)` | `scaled_online.pth` | 64.89 | 77.36 |
| `cotracker3_offline` | `CoTrackerPredictor(offline=True, window_len=60)` | `scaled_offline.pth` | 62.66 | 77.22 |

**Important**: `cotracker3_baseline` in Attempt 0 uses the **online** variant (not the offline strong anchor). The offline variant is tracked as a separate row (`cotracker3_offline`).

## Historical Reference

- **Date:** 2026-03-02
- **File:** `outputs/cleanup_logs/baseline_refs/cotracker3_davis_result_eval_20260302.json`
- **average_jaccard:** 70.43
- **average_pts_within_thresh:** 82.24
- **Note:** Historical snapshot uses CoTracker3 evaluator native metrics, NOT the same as TAPVid δ_avg. Cross-comparison requires unified rescoring.

## Key Note

CoTracker3 `ensemble/cotracker.py` wrapper (used by track_on) depends on `torch.hub.load` which requires GitHub network. We bypassed this by using `baselines/cotracker/cotracker/predictor.CoTrackerPredictor` directly with local checkpoints. This is the correct approach.
