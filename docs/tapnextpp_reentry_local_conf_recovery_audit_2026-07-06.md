# TAPNext++ Re-Entry Local Confidence Recovery Audit — 2026-07-06

## Decision

The re-entry-local confidence recovery sweep **does not pass** the deployable-method success gate.

Result:

```text
No configuration satisfies:
  AJ >= native - 0.10
  OA >= native - 0.10
  AJ_RD >= native + 0.01
```

Therefore, do **not** promote TAPNext++ confidence-based ReEntry as a deployable method result.

Keep the TAPNext++ controlled visibility-lag audit as diagnostic/headroom evidence only.

## Artifacts

Input cache:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt
```

Scripts:

```text
scripts/eval_tapnextpp_davis_full_v5_conf.py
scripts/eval_tapnextpp_visibility_threshold_sweep.py
scripts/eval_tapnextpp_reentry_local_conf_recovery.py
```

Outputs:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_full_eval_v5_conf.json
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_visibility_threshold_sweep.json
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_reentry_local_conf_recovery_sweep.json
```

Related docs:

```text
docs/tapnextpp_online_controlled_visibility_lag_audit_2026-07-06.md
docs/tapnextpp_visibility_confidence_sweep_2026-07-06.md
docs/next_step_tapnextpp_confidence_export_2026-07-06.md
```

## v5 confidence cache validation

The v5 cache was exported successfully and does not overwrite v4:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt
```

Validation passed:

```text
payload schema = tapnextpp_v5_conf
records = 30
pred_vis_logit exists
pred_vis_conf exists
pred_vis_conf in [0,1]
pred_visibility == (pred_vis_logit > 0)
pred_visibility == (pred_vis_conf > 0.5)
```

First-record example:

```text
video_id = bike-packing
pred_visibility mean = 0.7849
pred_vis_conf mean = 0.6929
pred_vis_conf std = 0.2825
logit range = [-6.1550, 10.5086]
```

## Native TAPNext++ reference

Native visibility rule:

```text
pred_visibility = pred_vis_conf >= 0.5
```

Native metrics:

| Metric | Value |
|---|---:|
| AJ | 65.8512 |
| OA | 92.3217 |
| delta_avg | 79.0900 |
| AJ_RD | 0.5961 |
| AJ_RD @D4 | 0.6995 |
| AJ_RD @D16 | 0.8046 |
| false-visible rate over GT-occluded | 0.2182 |
| false-invisible rate over GT-visible | 0.0487 |

## Global threshold sweep conclusion

Global threshold tuning failed as a method:

```text
tau=0.20 gives the best AJ_RD gain:
  AJ_RD +0.0157
but costs:
  AJ -3.05
  OA -3.46
```

Native tau=0.50 remains near best for standard AJ/OA.

## Local confidence recovery protocol

Base visibility:

```text
base_vis = pred_vis_conf >= 0.5
```

Recovery candidate:

```text
Frames immediately before native invisible -> visible re-entry segments.
```

Recovery rule:

```text
Open only candidate frames with pred_vis_conf >= tau_low.
Optionally reject if coordinate jump to native re-entry point is too large.
Optionally require recovered segment length >= min_len.
```

Sweep:

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

## Local recovery results

### Success count

```text
success_count = 0
```

### Best safe-ish configuration under AJ/OA >= native - 0.10

```text
tau_low = 0.40
pre_window = 1
max_jump_px = 32
min_recovered_segment_len = 1

AJ    = 65.7678  (Δ -0.0834)
OA    = 92.2903  (Δ -0.0314)
AJ_RD = 0.5978   (Δ +0.0017)
recovered_frames = 314
```

Interpretation:

```text
This is standard-metric safe but the AJ_RD gain is too small to matter.
```

### Best AJ_RD gain overall

```text
tau_low = 0.20
pre_window = 8
max_jump_px = inf
min_recovered_segment_len = 1

AJ    = 65.2426  (Δ -0.6086)
OA    = 91.8187  (Δ -0.5030)
AJ_RD = 0.6043   (Δ +0.0081)
recovered_frames = 1602
```

Interpretation:

```text
Even the best AJ_RD gain is below +0.01 and has too much AJ/OA cost for a clean method claim.
```

### Best under looser AJ/OA budgets

```text
AJ/OA >= native - 0.05:
  best is effectively no-op, AJ_RD +0.0000

AJ/OA >= native - 0.10:
  best AJ_RD gain +0.0017

AJ/OA >= native - 0.20:
  best AJ_RD gain +0.0043

AJ/OA >= native - 0.50:
  best AJ_RD gain +0.0077

AJ/OA >= native - 1.00:
  best AJ_RD gain +0.0081
```

## Interpretation

### What this proves

```text
1. TAPNext++ native visibility confidence contains some recoverable signal.
2. However, most recoverable frames are not safely separable by simple local confidence + jump gates.
3. TAPNext++ native tau=0.5 is already well-calibrated for standard metrics.
4. The deployable ReEntry opportunity on TAPNext++ first/input is much smaller than the controlled-lag oracle suggests.
```

### What this does not prove

```text
It does not prove that no learned selector could ever improve TAPNext++.
It does prove that the simple global/local confidence rules are not enough.
```

## Claim boundary

Safe claim:

```text
A controlled TAPNext++ online-coordinate audit shows large sensitivity to artificial re-entry visibility lag, but deployable confidence-threshold and local confidence-recovery rules do not yet convert this headroom into a reliable method gain.
```

Unsafe claim:

```text
ReEntry improves TAPNext++ online as a deployable method.
TAPNext++ confidence recovery is solved.
TAPNext++ SOTA route is ready for main paper table.
```

## Next step

Stop TAPNext++ deployable ReEntry development for the current paper cycle unless a stronger learned selector is explicitly scoped.

Recommended immediate paper action:

```text
Use TAPNext++ controlled visibility-lag and confidence-sweep results as diagnostic appendix evidence.
Keep main method claims centered on CoTracker3 / current ReEntry diagnostic story.
Do not train V26 yet.
```

If continuing research beyond the current paper cycle, the only justified next experiment is:

```text
Train a small metric-aware selector on held-out videos using:
  TAPNext++ confidence/logit
  re-entry-local temporal features
  coordinate continuity features
  false-visible/false-invisible utility labels

But this is a new research branch and must use a clean train/validation/test split.
```

Do not start this branch automatically.
