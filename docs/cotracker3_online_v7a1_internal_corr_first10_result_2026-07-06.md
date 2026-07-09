# CoTracker3 Online V7-A1 Internal Correlation First10 Result — 2026-07-06

## Purpose

V7-A1 tested whether CoTracker internal/support-correlation features can identify AJ_RD-aligned early re-entry events better than external native score/visibility/motion features.

This followed V7-A0 first3 smoke, where pooled feature AP looked promising.

## Artifacts

Approx-only first10:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_internal_corr_first10.npz
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_internal_corr_first10.manual_report.json
```

Exact-corr first10:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_internal_corr_first10_exact.npz
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_internal_corr_first10_exact.report.json
```

Script:

```text
scripts/export_cotracker3_online_v7a1_internal_corr_first10.py
```

## Implementation notes

V7-A1 adds two internal feature families:

```text
approx_internal: fnet support-current / last-visible cosine + local patch correlation summary
exact_corr: level-0 CoTracker-style corr_volume summary using get_track_feat() and get_correlation_feat()
```

A bug in exact-corr shape handling was fixed:

```text
get_correlation_feat() expects queried_coords batch size B*T, not B.
```

The fixed exact-corr smoke completed successfully.

## Dataset

First10 requested videos:

```text
bike-packing, blackswwan/blackswan, bmx-trees, breakdance, camel,
car-roundabout, car-shadow, cows, dance-twirl, dog
```

Effective videos with events:

```text
bike-packing, bmx-trees, breakdance, camel, car-roundabout,
car-shadow, cows, dance-twirl, dog
```

Events:

```text
688
```

Exact-corr feature dimension:

```text
153
```

## First3 optimism revisited

V7-A0 first3 had only two effective videos and showed strong pooled AP:

```text
early4 best pooled AP: 0.2794
early8 best pooled AP: 0.3584
```

A two-video LOOV check was already weaker, but still suggested the exact-corr path was worth testing.

## First10 exact-corr result

### early4

Label:

```text
positive = useful opening within first 4 GT-visible re-entry frames
```

Stats:

```text
positive: 33 / 688 = 4.80%
```

LOOV AP/AUC:

| Feature family | AP | AUC |
|---|---:|---:|
| native only | 0.0453 | 0.3594 |
| approx_internal | 0.0360 | 0.3727 |
| exact_corr | 0.0348 | 0.3439 |
| all_internal | 0.0330 | 0.3149 |
| all | 0.0332 | 0.3028 |

Best pooled exact-corr feature:

```text
exact_l0_std_mean: AP 0.1032, AUC 0.5046
```

Interpretation:

```text
Pooled features show mild signal, but video-level generalization fails.
```

### early8

Stats:

```text
positive: 48 / 688 = 6.98%
```

LOOV AP/AUC:

| Feature family | AP | AUC |
|---|---:|---:|
| native only | 0.0640 | 0.3455 |
| approx_internal | 0.0606 | 0.4399 |
| exact_corr | 0.0495 | 0.3383 |
| all_internal | 0.0561 | 0.3850 |
| all | 0.0535 | 0.3560 |

Best pooled exact-corr feature:

```text
exact_l0_std_mean: AP 0.1169, AUC 0.5416
```

Interpretation:

```text
Even early8 does not generalize across first10 videos with these frozen handcrafted internal features.
```

### useful_open_t

Stats:

```text
positive: 89 / 688 = 12.94%
```

LOOV AP/AUC:

| Feature family | AP | AUC |
|---|---:|---:|
| native only | 0.1979 | 0.6030 |
| approx_internal | 0.1332 | 0.4778 |
| exact_corr | 0.1013 | 0.4017 |
| all_internal | 0.1137 | 0.4095 |
| all | 0.1532 | 0.4773 |

Interpretation:

```text
For general useful opening, native score remains the strongest signal. Internal features do not help in LOOV.
```

## Main conclusion

V7-A1 falsifies the first3 optimism:

```text
Internal fnet/corr-volume summary features do not generalize across first10 videos for early re-entry detection.
```

The exact-corr features are not enough in their current handcrafted summary form.

## What this means for the online branch

The frozen CoTracker3 online + handcrafted verifier route has now been tested through:

```text
V3/V4: state-writeback / appearance gate
V5: candidate pool / coordinate verifier / event opening
V6: temporal native-score verifier / utility target / early-reentry target
V7-A: internal support-correlation summary features
```

Current conclusion:

```text
AJ_RD-aligned early re-entry has oracle headroom, but it is not recoverable with the current frozen, handcrafted feature/verifier approach.
```

## Next step decision

Do not proceed to V7-B LOOV metric simulation using these features.

The next viable online direction must change one of the following assumptions:

```text
1. Use a learned sequence-level verifier trained end-to-end or supervised on event windows.
2. Use a stronger online tracker/memory architecture such as Track-On / TAPNext-style memory.
3. Modify CoTracker itself to expose/train a re-entry head, rather than posthoc handcrafted summaries.
```

If staying with CoTracker3 online, the next minimal branch is:

```text
V8-A: train a small temporal MLP/1D-CNN on event windows with utility labels, using more training data and strict held-out video split.
```

But this is no longer a zero-training inference-time plug-in. It is a learned online re-entry module.
