# Three-Baseline ReEntry Full-Metric Audit — 2026-07-06

## Decision

The reported three-baseline table is numerically reproducible, with one important clarification:

```text
TAPNext++ numbers correspond to the W=8 sliding-window offline cache:
  outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_offline_w8_cache.pt
not to the fully per-frame independent offline cache:
  outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_offline_cache.pt
```

The correct interpretation is:

```text
ReEntry / visibility transplant improves visibility-derived metrics across three baselines.
However, only CoTracker3 is currently a meaningful method-level result.
For TAPNext++ and TrackOn2, the tested offline variants have weak coordinates, so the result is evidence for visibility failure/recovery, not a competitive SOTA method result.
```

## What was recomputed

Recomputed by using:

```text
offline tracks + offline visibility         -> offline result
offline tracks + online/baseline visibility -> +ReEntry result
```

This exactly tests coordinate-preserving visibility repair:

```text
tracks are unchanged
visibility is changed
therefore all δ metrics remain unchanged
```

Recomputed summary path:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/three_baseline_reentry_recomputed_official_full.json
```

## Caches used

### CoTracker3

```text
offline tracks/visibility:
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt

visibility source:
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt
```

### TAPNext++

```text
offline tracks/visibility:
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_offline_w8_cache.pt

visibility source:
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_baseline_w8_cache.pt
```

Important:

```text
The fully independent offline cache gives much worse numbers:
AJ 7.40, OA 71.17, delta_avg 14.34, AJ_RD 0.0190.
The table uses W=8 sliding-window offline, which gives AJ 12.49, OA 74.47, delta_avg 25.05, AJ_RD 0.1351.
```

### TrackOn2

```text
offline tracks/visibility:
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/trackon2_davis_offline_batch_cache.pt

visibility source:
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
```

Important:

```text
TrackOn2 offline here is predictor.model.forward() batch mode, not normal forward_online().
It is extremely weak: AJ 3.90, delta_avg 7.36.
This must not be described as TrackOn2's normal performance.
```

## Recomputed table

| Metric | CoTracker3 offline | CoTracker3 +ReEntry | Δ | TAPNext++ offline_w8 | TAPNext++ +ReEntry | Δ | TrackOn2 offline_batch | TrackOn2 +ReEntry | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AJ | 62.66 | 64.60 | +1.94 | 12.49 | 14.94 | +2.45 | 3.90 | 4.52 | +0.62 |
| OA | 88.15 | 91.80 | +3.65 | 74.47 | 92.32 | +17.85 | 59.68 | 92.09 | +32.41 |
| δ@1px | 43.82 | 43.82 | 0.00 | 14.54 | 14.54 | 0.00 | 1.30 | 1.30 | 0.00 |
| δ@4px | 84.58 | 84.58 | 0.00 | 22.09 | 22.09 | 0.00 | 5.43 | 5.43 | 0.00 |
| δ@16px | 96.94 | 96.94 | 0.00 | 42.14 | 42.14 | 0.00 | 17.38 | 17.38 | 0.00 |
| δ_avg | 77.22 | 77.22 | 0.00 | 25.05 | 25.05 | 0.00 | 7.36 | 7.36 | 0.00 |
| J@4px | 69.64 | 71.34 | +1.70 | 10.50 | 12.54 | +2.04 | 2.60 | 3.03 | +0.43 |
| J@8px | 78.63 | 82.30 | +3.67 | 13.96 | 16.75 | +2.79 | 5.07 | 5.81 | +0.75 |
| J@16px | 81.92 | 86.89 | +4.97 | 22.85 | 27.39 | +4.54 | 9.89 | 11.45 | +1.56 |
| AJ_RD | 0.5112 | 0.5644 | +0.0532 | 0.1351 | 0.1429 | +0.0078 | 0.0528 | 0.0537 | +0.0009 |
| AJ_RD @D4 | 0.5951 | 0.6556 | +0.0605 | 0.0856 | 0.0936 | +0.0081 | 0.0339 | 0.0384 | +0.0045 |
| AJ_RD @D16 | 0.6663 | 0.7604 | +0.0941 | 0.2105 | 0.2234 | +0.0128 | 0.0917 | 0.0990 | +0.0074 |

## Key verification points

### 1. The table is internally consistent

```text
All δ metrics are unchanged for all three trackers.
This confirms that the ReEntry variant is coordinate-preserving.
The gains come only from visibility correction.
```

### 2. CoTracker3 is the strongest real method evidence

```text
CoTracker3 has strong coordinate quality:
  delta_avg = 77.22
and ReEntry improves:
  AJ +1.94
  OA +3.65
  AJ_RD +0.0532
```

This is meaningful because the coordinate base is already usable.

### 3. TAPNext++ validates visibility recovery but not SOTA method strength

```text
TAPNext++ offline_w8 has weak coordinates:
  delta_avg = 25.05
ReEntry strongly improves OA:
  +17.85 pp
but AJ_RD gain is small:
  +0.0078
```

The normal online TAPNext++ baseline is much stronger:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_full_eval_v4.json
AJ = 65.85
OA = 92.32
AJ_RD = 0.5961
```

Therefore, the offline_w8 + ReEntry result should not be compared as if it were improving normal TAPNext++ online performance. It is a controlled visibility-failure / recovery stress test.

### 4. TrackOn2 validates visibility failure but is not a useful SOTA-base result

```text
TrackOn2 offline_batch has extremely weak coordinates:
  AJ = 3.90
  delta_avg = 7.36
```

ReEntry raises OA from 59.68 to 92.09, but AJ_RD barely changes:

```text
AJ_RD +0.0009
```

This means TrackOn2 offline_batch's visibility is bad, but its coordinate base is too poor for ReEntry to matter as a method result.

## Correct claim boundary

Safe claim:

```text
Across CoTracker3, TAPNext++ offline_w8, and TrackOn2 offline_batch stress baselines, coordinate-preserving visibility correction consistently improves visibility-derived metrics, especially OA, while leaving coordinate metrics unchanged.
```

Stronger but still safe:

```text
The gains confirm that ReEntry isolates a real visibility failure mode that appears across tracker families when the coordinate stream is held fixed.
```

Unsafe claim:

```text
ReEntry improves SOTA TAPNext++.
ReEntry improves TrackOn2 as a normal online tracker.
ReEntry is SOTA-compatible in the main protocol.
Three strong baselines all improve competitively.
```

Why unsafe:

```text
TAPNext++ and TrackOn2 results are based on degraded/offline stress variants, not their normal online operation.
Their coordinate quality is too weak in the tested offline modes.
```

## Paper impact

This result can improve the paper as a diagnostic / cross-family failure-mode section, not as a top-tier SOTA method claim.

Recommended placement:

```text
Appendix or diagnostic subsection:
  Cross-family visibility-stress audit
```

Possible wording:

```text
We further construct coordinate-preserving visibility-repair stress tests across CoTracker3, TAPNext++, and TrackOn2-derived offline variants. Replacing the degraded visibility stream with the corresponding online visibility stream consistently improves OA and Jaccard-based visibility metrics while leaving all coordinate thresholds unchanged. This confirms that re-entry visibility failure is not unique to one codebase. However, for TAPNext++ and TrackOn2 offline stress variants, coordinate quality is weak, so these results should be interpreted as failure-mode evidence rather than competitive tracker improvements.
```

## Next step

Do not train V26 yet.

Recommended next action:

```text
1. Add this as a diagnostic cross-family visibility-stress result.
2. Separately run normal-online TAPNext++ + true ReEntry candidate experiment only if we can define a real degraded visibility source that preserves strong TAPNext++ coordinates.
3. Do not use TrackOn2 offline_batch as a method baseline.
```

Best next technical route:

```text
TAPNext++ normal online coordinates + controlled visibility degradation / recovery:
  base tracks = normal TAPNext++ online tracks
  degraded visibility = artificially lagged / thresholded / window-reset visibility
  recovery visibility = normal TAPNext++ online visibility or ReEntry-predicted visibility

This would test visibility correction while preserving strong coordinates.
```
