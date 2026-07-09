# CoTracker3 True-Streaming Online V3 Result — 2026-07-06

## Goal

Evaluate whether ReEntry can be safely connected to the true streaming CoTracker3 online state.

This is different from the old `cotracker3_baseline` cache, which is an online-architecture full-sequence evaluation cache. V3 uses the true `CoTrackerOnlinePredictor` streaming loop.

## Method

V3 changes over earlier attempts:

```text
1. Decouple internal state writeback from final output.
2. Only write to the overlap region that later windows can inherit.
3. Use stable re-entry confirmation: confirm_visible_len = 4.
4. Use tight coordinate gate: max_jump_px = 4.
5. Use soft state writeback: state_prob = 0.81 for each vis/conf component.
```

Parameters:

```text
tau_low = 0.55
pre_window = 2
max_jump_px = 4
min_len = 1
confirm_visible_len = 4
trend_min = -0.02
state_prob = 0.81
```

Output path:

```text
outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/
```

## Full 30-video result

| Variant | AJ | OA | delta_avg | delta_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| true-streaming native | 65.2366 | 90.8186 | 77.9458 | 85.8670 | 0.3534 | 0.5333 |
| output-only | 65.2159 | 90.7955 | 77.9458 | 85.8670 | 0.3535 | 0.5335 |
| state-writeback | 65.2395 | 90.8228 | 77.9463 | 85.8636 | 0.3534 | 0.5333 |
| state+output | 65.2192 | 90.8015 | 77.9463 | 85.8636 | 0.3535 | 0.5335 |

Deltas vs true-streaming native:

| Variant | dAJ | dOA | dDelta | d4px | dAJ_RD | dAJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| output-only | -0.0207 | -0.0231 | +0.0000 | +0.0000 | +0.0001 | +0.0002 |
| state-writeback | +0.0029 | +0.0043 | +0.0005 | -0.0034 | +0.0000 | +0.0000 |
| state+output | -0.0174 | -0.0171 | +0.0005 | -0.0034 | +0.0001 | +0.0002 |

## Opened-frame audit

`full30_opened_frame_audit.json`:

```text
output-only:
  opened frames: 31
  GT visible: 11
  GT occluded: 20
  precision: 35.48%

state-writeback:
  opened frames: 5
  GT visible: 3
  GT occluded: 2
  precision: 60.00%

state+output:
  opened frames: 35
  GT visible: 14
  GT occluded: 21
  precision: 40.00%
```

## Interpretation

V3 state writeback is structurally correct and much safer than earlier state-writeback attempts.

It achieves:

```text
AJ/OA/delta_avg: tiny positive deltas
AJ_RD: no meaningful improvement
opened-frame precision: improved to 60% for pure state-writeback
```

But the effect size is too small for a paper main result.

Safe claim:

```text
A conservative overlap-only state writeback can be made non-destructive in true-streaming CoTracker3, but it does not yet produce meaningful re-entry improvement.
```

Unsafe claim:

```text
CoTracker3 online state-writeback improves re-entry performance.
```

## Decision

Do not use as a main result.

Potential use:

```text
Appendix / online-extension diagnostic / limitation analysis.
```

Next technical direction:

```text
The remaining bottleneck is candidate verification. CoTracker3 score + coordinate smoothness is insufficient. A real online state-level improvement likely needs an appearance verifier or memory-based candidate validation.
```
