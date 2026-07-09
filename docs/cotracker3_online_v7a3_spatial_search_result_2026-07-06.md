# CoTracker3 Online V7-A3 Internal Spatial Search Result — 2026-07-06

## Purpose

V7-A3 tests whether internal CoTracker fnet support-feature spatial search can find re-entry candidates better than scoring only at the native predicted coordinate.

This was motivated by V7-A1/V7-A2: raw/normalized internal features at the native coordinate showed pooled signal but poor video-level LOOV generalization.

## Script

```text
scripts/export_cotracker3_online_v7a3_internal_spatial_search.py
```

Important implementation correction:

```text
fnet input now matches CoTracker forward:
video = 2 * (video / 255.0) - 1.0
```

This fixes a likely mismatch in earlier V7-A0/V7-A1 feature extraction.

## Runs

Both runs use first10 videos with a stratified early8 sample:

```text
early8 positives: 48
sampled negatives: 96
total events: 144
effective videos: 7
```

Runs:

```text
r64:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a3_spatial_search/v7a3_spatial_search_first10_early8_stratified_r64.npz

r32:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a3_spatial_search/v7a3_spatial_search_first10_early8_stratified_r32.npz
```

## r64 result

### early8 positive subset

```text
native_r16: 1.0000
search_top1_r16: 0.5000
search_top5_r16: 0.8125
native_r8: 0.8333
search_top1_r8: 0.1458
search_top5_r8: 0.3542
native median error: 2.41 px
search top1 median error: 15.84 px
search top5 median error: 8.77 px
```

### early8 negative subset

```text
native_r16: 0.1354
search_top1_r16: 0.0938
search_top5_r16: 0.1875
native_r8: 0.1042
search_top1_r8: 0.0521
search_top5_r8: 0.0833
```

LOOV search-stat classifier:

```text
early4 AP 0.1505, AUC 0.2285
early8 AP 0.2150, AUC 0.1853
useful_open_t AP 0.2886, AUC 0.3028
```

## r32 result

### early8 positive subset

```text
native_r16: 1.0000
search_top1_r16: 0.7500
search_top5_r16: 0.9583
native_r8: 0.8333
search_top1_r8: 0.2708
search_top5_r8: 0.6458
native median error: 2.41 px
search top1 median error: 12.89 px
search top5 median error: 6.25 px
```

### early8 negative subset

```text
native_r16: 0.1354
search_top1_r16: 0.1250
search_top5_r16: 0.1667
native_r8: 0.1042
search_top1_r8: 0.0521
search_top5_r8: 0.0833
```

LOOV search-stat classifier:

```text
early4 AP 0.1517, AUC 0.2228
early8 AP 0.2364, AUC 0.2741
useful_open_t AP 0.2934, AUC 0.3205
```

## Interpretation

r32 is clearly better than r64 as a candidate-search radius:

```text
early8 positive top5_r16 improves from 0.8125 to 0.9583.
early8 positive top5_r8 improves from 0.3542 to 0.6458.
```

However, native coordinate is already safe for the positive early8 events by construction:

```text
early8 positive native_r16 = 1.0
native_r8 = 0.8333
```

Therefore, spatial candidate search does not solve the actual bottleneck. The bottleneck is not coordinate recovery; it is deciding when to open visibility.

Search-stat LOOV classifiers also perform poorly, so the search peaks/margins/entropy are not enough to robustly classify early re-entry under video-level split.

## Reflection

V7-A3 corrects one earlier conceptual issue: if native coordinates were wrong, searching would be necessary. But the V6-A3 early-reentry positives are defined by native coordinate being already safe while native visibility is low. So candidate search is mostly orthogonal to the current oracle headroom.

This means:

```text
CoTracker3 online's AJ_RD headroom in this branch is visibility calibration at already-good coordinates, not spatial re-localization.
```

The spatial-search path may be useful for a different failure mode, but it is not the right next step for the current early-reentry oracle.

## Decision

Do not continue V7-A3 spatial search as the main branch.

Next step should return to visibility calibration, but with raw CoTracker components not preserved in the current cache:

```text
V7-A4: export raw vis/conf component temporal features.
```

Rationale:

```text
Current cache stores only pred_vis_score = sigmoid(raw_vis) * sigmoid(raw_conf).
It does not store raw_vis and raw_conf separately.
CoTracker scripts already expose online_vis_predicted and online_conf_predicted.
The product score may hide useful distinctions, e.g. high visibility / low confidence vs low visibility / high confidence.
```

V7-A4 should export:

```text
raw_vis_logit, raw_conf_logit
sigmoid(raw_vis), sigmoid(raw_conf)
product score
vis/conf ratio and margins
short-latency t...t+4 slopes
component disagreement features
```

Evaluate on the same V6-A3 labels:

```text
early4 / early8 / useful_open_t
```

Success criterion:

```text
First10 LOOV early8 AP/AUC must improve over native product-score-only features.
If raw vis/conf components also fail, lightweight CoTracker3 online visibility calibration is likely exhausted.
```
