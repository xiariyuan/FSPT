# True-Base ReEntry Requirement and Current Status — 2026-07-06

## Decision

The paper must not present degraded/offline stress caches as if they were real tracker baselines.

Hard rule:

```text
Main improvement tables require real base tracker outputs plus the ReEntry module.
Stress/window-reset/offline-batch variants may only appear as diagnostic appendix rows.
```

## What counts as real base

A real base is the tracker's intended inference path for the reported protocol:

```text
CoTracker3 offline base: accepted as a real published/base mode if clearly named.
TAPNext++ normal online first/input: real base.
TrackOn2 normal forward_online first/input: real base.
```

Not real base for main improvement:

```text
TAPNext++ offline_w8 / window-reset stress.
TrackOn2 predictor.model.forward() offline_batch stress.
TrackOn2 non-parity strided/original weak cache.
```

## Current status by tracker

### CoTracker3

Status:

```text
PASS for main result.
```

Real base:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt
```

ReEntry result:

```text
AJ     62.66 -> 64.60  (+1.94)
OA     88.15 -> 91.80  (+3.65)
AJ_RD  0.5112 -> 0.5644 (+0.0532)
```

Interpretation:

```text
This is the main headline improvement result.
```

### TAPNext++

Real base:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt
```

Real base metrics:

```text
AJ     65.85
OA     92.32
AJ_RD  0.5961
```

True-base module status:

```text
TAPNext++ true-base confidence/logit module was tested.
Global threshold sweep: no safe method gain.
Local re-entry confidence recovery: success_count = 0.
```

Best safe-ish local result:

```text
AJ     -0.0834
OA     -0.0314
AJ_RD  +0.0017
```

Best AJ_RD result overall:

```text
AJ     -0.6086
OA     -0.5030
AJ_RD  +0.0081
```

Interpretation:

```text
TAPNext++ true-base module does not currently produce a meaningful deployable improvement.
Do not use TAPNext++ as a main improvement row.
Use controlled visibility-lag and confidence sweep as diagnostic/negative evidence only.
```

### TrackOn2

Real base:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
```

Current cache status:

```text
Normal first/input TrackOn2 cache exists.
It has pred_tracks and binary pred_visibility.
It does not store pred_vis_conf or pred_vis_logit.
```

Therefore:

```text
A true-base confidence-based ReEntry module cannot be fairly evaluated yet.
```

Required next step if TrackOn2 is to be used as a true-base module baseline:

```text
Export TrackOn2 normal forward_online first/input cache with visibility logits/confidence.
Then run the same threshold/local recovery audit as TAPNext++.
```

Do not use the following as main result:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/trackon2_davis_offline_batch_cache.pt
```

Reason:

```text
It is predictor.model.forward() offline_batch stress and has AJ 3.90, not a real normal TrackOn2 base.
```

## Main-table policy

Allowed main rows now:

```text
CoTracker3 offline + ReEntry on DAVIS first/input.
RGB-Stacking CoTracker3 + ReEntry as secondary/trade-off result.
DAVIS strided/original V25 safe as conservative protocol result.
```

Not allowed in main table:

```text
TAPNext++ offline_w8.
TrackOn2 offline_batch.
TAPNext++ controlled lag as method row.
TAPNext++ local recovery negative result as method row.
```

## Next action choices

### Option A — writing-first route

Recommended for current paper cycle:

```text
Freeze main results around CoTracker3 and use TAPNext++/TrackOn2 only as appendix diagnostic/negative evidence.
```

### Option B — continue true-base baseline route

Only useful if the user wants more baselines in main table:

```text
1. Export TrackOn2 normal forward_online with confidence/logits.
2. Run true-base TrackOn2 threshold/local recovery audit.
3. Only include if AJ/OA remain non-weak and AJ_RD improves meaningfully.
```

Success gate:

```text
AJ >= base - 0.10
OA >= base - 0.10
AJ_RD >= base + 0.01
```

If this gate fails, TrackOn2 must stay appendix-only.
