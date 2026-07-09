# CoTracker3 Online V7-A1 Internal Correlation Review — 2026-07-06

## Purpose

V7-A1 tests whether CoTracker internal/support-correlation features can identify AJ_RD-aligned early re-entry events better than native score/visibility/motion features.

This follows V6-A3, where early re-entry oracle had strong AJ_RD-aligned headroom but native temporal features failed to identify those events.

## Scripts

```text
scripts/export_cotracker3_online_v7a1_internal_corr_first10.py
```

Two important runs were made:

```text
1. first10 no-exact export:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_internal_corr_first10.npz
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_internal_corr_first10.posthoc_report.json

2. exact-corr stratified early8 smoke:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_exact_first10_early8_stratified.npz
```

## Implementation notes

The exact-corr implementation initially failed because `get_correlation_feat()` expects the query-coordinate batch dimension to match `B*T` after internal flattening. The fix was to use a single-frame feature map slice:

```text
fmap_one = fmap[:, t:t+1]
curr_coords shape = [1, 1, 2]
```

This corrected the batch-size mismatch and allowed exact level-0 correlation-volume summaries to be exported.

## V7-A1 first10 no-exact review

Dataset:

```text
n_events = 688
feature_dim = 76
effective videos = 9
exact_enabled = false
```

Effective videos:

```text
bike-packing, bmx-trees, breakdance, camel, car-roundabout,
car-shadow, cows, dance-twirl, dog
```

### early4

```text
positives: 33 / 688 = 4.8%
```

Best pooled single features:

```text
approx_peak_std AP 0.1128, AUC 0.5383
approx_entropy_std AP 0.1038, AUC 0.5684
native_score_dt1 AP 0.0788, AUC 0.5675
```

LOOV logistic:

```text
native-only AP 0.0453, AUC 0.3594
approx-internal-only AP 0.0360, AUC 0.3727
all AP 0.0347, AUC 0.3367
```

Conclusion:

```text
No-exact approximate internal features do not show stable first10 LOOV early4 signal.
```

### early8

```text
positives: 48 / 688 = 7.0%
```

Best pooled single features:

```text
native_score_dt0 AP 0.1151, AUC 0.5817
approx_peak_std AP 0.1084, AUC 0.5003
approx_cos_last_std AP 0.1024, AUC 0.6111
```

LOOV logistic:

```text
native-only AP 0.0640, AUC 0.3455
approx-internal-only AP 0.0606, AUC 0.4399
all AP 0.0588, AUC 0.3741
```

Conclusion:

```text
Approx internal features are not yet enough for cross-video early8 prediction.
```

## V7-A1 exact-corr stratified early8 smoke

To keep exact-corr manageable, this run selected all first10 early8 positives and 2x negatives:

```text
positive early8 events = 48
negative sampled events = 96
total = 144
feature_dim = 153
exact_enabled = true
```

Effective videos:

```text
bike-packing, bmx-trees, breakdance, car-roundabout,
car-shadow, dance-twirl, dog
```

This is a stratified smoke, not a natural-distribution estimate.

### early4 on stratified sample

```text
positives: 33 / 144 = 22.9%
```

Pooled single features:

```text
native_score_dt2 AP 0.3509, AUC 0.6055
approx_entropy_std AP 0.3442, AUC 0.5777
exact_l0_margin_dt1 AP 0.3597, AUC 0.5416
exact_l0_std_std AP 0.3513, AUC 0.6249
```

LOOV logistic:

```text
native-only AP 0.2012, AUC 0.3568
approx-internal-only AP 0.1840, AUC 0.3732
exact-corr-only AP 0.1797, AUC 0.3754
all AP 0.1656, AUC 0.3131
```

### early8 on stratified sample

```text
positives: 48 / 144 = 33.3%
```

Pooled single features:

```text
native_score_dt1 AP 0.4701, AUC 0.5898
native_score_dt2 AP 0.4691, AUC 0.6155
exact_l0_entropy_max AP 0.4636, AUC 0.6043
exact_l0_entropy_mean AP 0.4633, AUC 0.5775
approx_peak_dt4 AP 0.4307, AUC 0.5931
```

LOOV logistic:

```text
native-only AP 0.2913, AUC 0.2999
approx-internal-only AP 0.2554, AUC 0.3507
exact-corr-only AP 0.2509, AUC 0.3388
all AP 0.3037, AUC 0.3553
```

### useful_open_t on stratified sample

```text
positives: 55 / 144 = 38.2%
```

Pooled single features:

```text
native_score_dt2 AP 0.5165, AUC 0.6339
exact_l0_entropy_mean AP 0.4844, AUC 0.5598
approx_cos_support_dt3 AP 0.4786, AUC 0.5765
```

LOOV logistic:

```text
native-only AP 0.3347, AUC 0.3338
approx-internal-only AP 0.3118, AUC 0.3947
exact-corr-only AP 0.2753, AUC 0.2795
all AP 0.3037, AUC 0.3553
```

## Interpretation

V7-A1 changes the conclusion from V7-A0:

```text
Internal correlation features have pooled single-feature signal, but do not yet generalize under video-level LOOV.
```

The most likely issue is video/query distribution shift in raw correlation magnitudes:

```text
- pooled AP is high;
- LOOV AP/AUC collapses;
- exact/internal features do not consistently beat native-only across videos.
```

Therefore, the next issue is not simply “need more exact-corr features.” It is:

```text
Need per-video/per-query normalization or relative ranking features before training a verifier.
```

## Reflection

Earlier optimism from V7-A0 was too strong because:

```text
1. first3 had only two effective videos;
2. pooled single-feature AP can be inflated by video-specific distribution differences;
3. exact corr-volume was not yet included;
4. no video-level generalization was proven.
```

V7-A1 fixed part of this by expanding to first10 and adding exact-corr smoke. The result is more conservative:

```text
V7 should continue only after feature normalization, not directly to V7-B verifier.
```

## Decision

Do not proceed to V7-B method simulation yet.

Next step:

```text
V7-A2 normalized internal correlation features.
```

V7-A2 should transform raw internal features into:

```text
1. per-video z-score features;
2. per-query / support-relative features;
3. within-event temporal rank features;
4. internal-minus-native-score residual features;
5. top-k / percentile features rather than raw magnitudes.
```

Success gate:

```text
early4 first10 LOOV AP >= 0.10 and AUC >= 0.60,
early8 first10 LOOV AP >= 0.14 and AUC >= 0.60,
and internal/normalized features must beat native-only.
```

If normalized internal features still fail, frozen CoTracker3 internal features are likely insufficient for lightweight early re-entry verification.
