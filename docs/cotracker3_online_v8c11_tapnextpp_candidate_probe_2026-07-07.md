# CoTracker3 Online V8-C1.1 TAPNext++ Candidate Probe

Date: 2026-07-07

Purpose: check whether CVRRM can work with a stronger non-TrackOn2 candidate source.

## Setup

Several TAPNext++ DAVIS caches have the same record shapes as the native CoTracker3 cache, but strict alignment reports about 1/255 normalized query/GT differences. For diagnostic only, a native-metadata bridge was used:

```text
Keep TAPNext++ candidate pred_tracks / pred_visibility.
Use native query_points / gt_tracks / gt_visibility / metadata for evaluation.
```

This is not yet final protocol, but it tests candidate potential under aligned metadata.

## Online-style TAPNext++ caches

The following caches behave identically in this probe:

```text
tapnextpp_davis_baseline_cache.pt
tapnextpp_davis_baseline_w8_cache.pt
tapnextpp_davis_first_input_cache_v4.pt
tapnextpp_davis_first_input_cache_v5_conf.pt
```

Standalone after metadata bridge:

```text
AJ -1.4554, OA +1.5031, AJ_RD -0.0052, AJ_RD_256 +0.0035
```

CVRRM default:

```text
dist_nc_le64_w8_candidate_visible
AJ -0.6909, OA +0.5097, AJ_RD +0.0063, AJ_RD_256 +0.0204
positive/negative/zero = 16/6/3
```

CVRRM high-gain:

```text
dist_nc_le64_w16_candidate_visible
AJ -0.9359, OA +0.4455, AJ_RD +0.0045, AJ_RD_256 +0.0209
positive/negative/zero = 16/6/3
```

Interpretation:

```text
CVRRM can reach the +0.020 AJ_RD_256 threshold with online-style TAPNext++ candidate caches, but overall AJ is negative and false-visible/damage rates are higher than TrackOn2 bridge.
```

## Offline TAPNext++ caches

Offline caches are not suitable as candidate sources here:

```text
tapnextpp_davis_offline_cache.pt:    CVRRM w8 AJ_RD_256 -0.0695
tapnextpp_davis_offline_w8_cache.pt: CVRRM w8 AJ_RD_256 -0.0616
```

They produce high damage and false-visible rates.

## Updated conclusion

V8-C1.1 changes the previous conclusion:

```text
CVRRM is not only TrackOn2-specific.
It can also extract re-entry gain from online-style TAPNext++ candidates.
However, TrackOn2 bridge remains the best candidate provider among tested sources.
```

Current candidate ranking for CVRRM default w8:

```text
TrackOn2 bridge:          AJ_RD_256 +0.0274, AJ +0.0964
TAPNext++ online-style:   AJ_RD_256 +0.0204, AJ -0.6909
old CoTracker online:     AJ_RD_256 +0.0065, AJ -0.3241
old CoTracker offline:    AJ_RD_256 +0.0039, AJ -0.0773
TAPNext++ offline:        negative
```

## Next step

Do not start state-level repair yet. First make V8-C1.1 official by turning the metadata-bridge diagnostic into a clean script/report, then decide whether to include TAPNext++ as a cross-candidate ablation.

Recommended next task:

```text
V8-C1.2: formalize TAPNext++ metadata-bridge candidate evaluation.
```

This should produce a machine JSON report and include the TAPNext++ rows in the final candidate-source table.
