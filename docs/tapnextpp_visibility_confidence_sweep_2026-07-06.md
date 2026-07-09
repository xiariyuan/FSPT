# TAPNext++ Visibility Confidence Threshold Sweep — 2026-07-06

## Decision

Global visibility-threshold tuning is **not sufficient** as a deployable ReEntry method for TAPNext++.

The sweep shows:

```text
Lowering the threshold below 0.5 can slightly improve AJ_RD, but it substantially hurts AJ/OA because false-visible frames increase.
Raising the threshold above 0.5 hurts AJ/OA and AJ_RD by creating false-invisible frames.
The native threshold tau=0.5 is near the best operating point for standard AJ/OA.
```

Therefore, the next step should not be global threshold tuning. The next step should be a **re-entry-local confidence recovery rule** that only opens visibility inside candidate re-entry windows and only when safety checks pass.

## Artifacts

Input cache:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt
```

Script:

```text
scripts/eval_tapnextpp_visibility_threshold_sweep.py
```

Output:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_visibility_threshold_sweep.json
```

The v5 cache was validated:

```text
pred_vis_logit exists
pred_vis_conf exists
pred_vis_conf in [0,1]
pred_visibility == (pred_vis_logit > 0)
pred_visibility == (pred_vis_conf > 0.5)
```

## Threshold sweep results

Reference is native tau=0.50:

```text
AJ     = 65.85
OA     = 92.32
delta_avg = 79.09
AJ_RD  = 0.5961
```

| tau | AJ | OA | AJ_RD | ΔAJ vs 0.5 | ΔOA vs 0.5 | ΔAJ_RD vs 0.5 | pred visible rate | FV rate over GT-occluded | FI rate over GT-visible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 60.18 | 84.23 | 0.6015 | -5.67 | -8.09 | +0.0053 | 0.9146 | 0.7077 | 0.0008 |
| 0.10 | 61.23 | 86.16 | 0.6079 | -4.62 | -6.16 | +0.0117 | 0.8910 | 0.6470 | 0.0016 |
| 0.20 | 62.80 | 88.86 | 0.6118 | -3.05 | -3.46 | +0.0157 | 0.8327 | 0.5010 | 0.0069 |
| 0.30 | 64.07 | 90.64 | 0.6066 | -1.78 | -1.68 | +0.0104 | 0.8014 | 0.4056 | 0.0164 |
| 0.40 | 65.14 | 91.75 | 0.6019 | -0.71 | -0.57 | +0.0057 | 0.7649 | 0.3081 | 0.0309 |
| 0.50 | 65.85 | 92.32 | 0.5961 | +0.00 | +0.00 | +0.0000 | 0.7295 | 0.2182 | 0.0487 |
| 0.60 | 65.81 | 91.64 | 0.5762 | -0.04 | -0.68 | -0.0199 | 0.6943 | 0.1448 | 0.0777 |
| 0.70 | 64.14 | 88.80 | 0.5406 | -1.71 | -3.52 | -0.0555 | 0.6438 | 0.0853 | 0.1325 |
| 0.80 | 58.82 | 81.85 | 0.4744 | -7.03 | -10.47 | -0.1218 | 0.5604 | 0.0341 | 0.2348 |
| 0.90 | 39.49 | 61.72 | 0.3214 | -26.36 | -30.60 | -0.2748 | 0.3616 | 0.0057 | 0.5019 |
| 0.95 | 18.01 | 41.21 | 0.1590 | -47.84 | -51.11 | -0.4371 | 0.1676 | 0.0011 | 0.7667 |

## Interpretation

### 1. Global low threshold improves AJ_RD slightly but is not safe

Best AJ_RD in the sweep is around tau=0.20:

```text
tau=0.20:
  AJ_RD +0.0157
  AJ    -3.05
  OA    -3.46
```

This is not an acceptable method operating point because the AJ/OA cost is too high.

### 2. Native tau=0.5 is near optimal for standard metrics

```text
tau=0.50:
  AJ 65.85
  OA 92.32
```

No lower tau improves standard metrics. Higher thresholds create false-invisible errors and quickly degrade all metrics.

### 3. Confidence contains some re-entry signal, but it is not enough globally

Lower thresholds indicate recoverable visibility mass exists, because AJ_RD rises from 0.5961 to 0.6118. But global recovery also opens too many false-visible frames. This confirms the need for a local/selective rule.

## Claim boundary

Safe claim:

```text
TAPNext++ visibility confidence contains recoverable re-entry signal, but global threshold tuning trades re-entry recovery against standard AJ/OA.
```

Unsafe claim:

```text
A simple global threshold improves TAPNext++.
```

## Next step

Implement a **re-entry-local confidence recovery** experiment:

```text
Base visibility: conf >= 0.5
Candidate recovery frames: frames in an invisible run immediately preceding or near a native re-entry event.
Recovery rule: open frames with conf >= tau_low only inside those local windows.
Safety gate: reject recovery if coordinate continuity jump is too large or segment is too short/isolated.
```

Suggested sweep:

```text
tau_low in {0.10, 0.20, 0.30, 0.40}
pre_window in {1, 2, 4, 8}
max_jump_px in {16, 32, 64, inf}
min_recovered_segment_len in {1, 2}
```

Success criterion:

```text
AJ >= native - 0.10
OA >= native - 0.10
AJ_RD >= native + 0.01
```

If this fails, stop TAPNext++ deployable ReEntry for now and keep the controlled-lag audit as diagnostic evidence only.
