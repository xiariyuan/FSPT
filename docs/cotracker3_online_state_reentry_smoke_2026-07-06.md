# CoTracker3 Online State-Level ReEntry Smoke — 2026-07-06

## Purpose

Test whether ReEntry-style visibility correction can be written back into CoTracker3 online state, so that corrected visibility/confidence affects later windows.

This smoke is intentionally 1-video only.

## Key code finding

CoTracker3 online maintains state:

```text
online_track_feat
online_track_support
online_coords_predicted
online_vis_predicted
online_conf_predicted
online_ind
```

The next online window copies previous-window predictions into initialization:

```text
coords_prev -> coords_init
vis_prev    -> vis_init
conf_prev   -> conf_init
```

Therefore, visibility/confidence correction can theoretically affect later online windows.

## Important protocol finding

The existing cache:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt
```

is the CoTracker3 online-architecture baseline produced through `CoTrackerPredictor(offline=False, window_len=16)` full-sequence evaluation. It is not identical to the true streaming `CoTrackerOnlinePredictor` loop used for state writeback.

The 1-video streaming native rerun does not parity-match the existing cache:

```text
video: bike-packing
existing cache AJ/OA: 58.7728 / 90.3654
streaming native AJ/OA: 57.5292 / 86.7774
visibility diff vs existing cache: 326 frames, 18.90%
```

Decision:

```text
Do not compare state-writeback results directly against the old cotracker3_baseline cache.
Use streaming native rerun as the baseline for state-writeback experiments.
```

## Smoke variants

Two parameter sets were tested on one video (`bike-packing`).

### Aggressive

```text
tau_low=0.40, pre_window=4, max_jump_px=24, state_prob=0.97
```

| Variant | AJ | OA | delta_avg | AJ_RD | AJ_RD_256 | ΔAJ vs streaming native | ΔOA | ΔAJ_RD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| streaming native | 57.5292 | 86.7774 | 76.0077 | 0.2869 | 0.4995 | 0 | 0 | 0 |
| output-only | 57.2643 | 86.3123 | 76.0077 | 0.2879 | 0.5038 | -0.2649 | -0.4651 | +0.0010 |
| state-writeback | 56.6728 | 85.4485 | 76.0077 | 0.2710 | 0.4722 | -0.8563 | -1.3289 | -0.0159 |

### Conservative

```text
tau_low=0.55, pre_window=2, max_jump_px=16, state_prob=0.97
```

| Variant | AJ | OA | delta_avg | AJ_RD | AJ_RD_256 | ΔAJ vs streaming native | ΔOA | ΔAJ_RD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| streaming native | 57.5292 | 86.7774 | 76.0077 | 0.2869 | 0.4995 | 0 | 0 | 0 |
| output-only | 57.4569 | 86.6445 | 76.0077 | 0.2867 | 0.4991 | -0.0723 | -0.1329 | -0.0002 |
| state-writeback | 57.1400 | 86.2458 | 75.9885 | 0.2886 | 0.4981 | -0.3891 | -0.5316 | +0.0017 |

## Interpretation

The architectural hypothesis is valid:

```text
CoTracker3 online visibility/confidence state can be modified and carried into later windows.
```

But the initial writeback rule is not yet useful:

```text
State writeback can contaminate later windows; with current simple gate it hurts AJ/OA and does not produce meaningful AJ_RD gain.
```

This is not a full negative result, because only one video and two hand-set parameter configs were tested. However it is enough to block full 30-video state-writeback evaluation for now.

## Next action

Do not expand state-writeback to 30 videos yet.

Recommended next steps:

```text
1. Establish a clean true-streaming CoTracker3 native baseline cache for all DAVIS videos.
2. Build a candidate audit that measures which frames are opened by output-only/state-writeback rules.
3. Add stronger gates before state writeback: stricter score, multi-frame consistency, confidence rise, and coordinate stability.
4. Only rerun state-writeback once the 1-video smoke shows AJ/OA-safe behavior.
```

Current paper-use status:

```text
No CoTracker3 online state-level improvement claim yet.
```
