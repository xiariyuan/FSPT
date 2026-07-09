# Standard Benchmark Comparison: TAPVid-DAVIS Bridge — 2026-07-04

## 1. Scope and caution

This document evaluates the frozen ReEntry pipeline on the available TAPVid-DAVIS bridge caches.

This is more standard than the RGB-Stacking fresh20-49 stress/fresh split, but it is still important to describe the exact protocol:

```text
Dataset/protocol: TAPVid-DAVIS first-input bridge cache
Videos: 30
Queries: 650
Re-entry queries: 259
Base cache: CoTracker3 offline DAVIS first-input bridge
Override cache for ReEntry methods: CoTracker3 baseline DAVIS first-input bridge
```

This comparison is therefore a DAVIS bridge full 30-video comparison, not the official public leaderboard submission.

---

## 2. Caches and artifacts

Base/override/external caches:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
```

Generated ReEntry caches:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v1_offline_to_baseline_thr010/reentry_v1_davis_offline_to_baseline_thr010.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v22Q_offline_to_baseline/reentry_v22Q_davis_offline_to_baseline.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v24_dinoscore_offline_to_baseline/reentry_v24_dinoscore_davis_offline_to_baseline.pt
```

DINOScore support artifacts:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/davis_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/davis_video_cache.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v24_dinoscore_decisions_000000_003613.npz
```

Paired stats:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/paired_stats_davis_v24_vs_all.json
docs/standard_benchmark_davis_paired_stats_2026-07-04.md
```

---

## 3. DAVIS full 30-video metrics

Video-weighted paired-stat means:

| Method | Role | AJ_RD | AJ | OA |
|---|---|---:|---:|---:|
| offline | CoTracker3 offline base | 0.409808 | 62.166050 | 89.010664 |
| baseline | CoTracker3 baseline / override | 0.479548 | 62.818065 | 90.263068 |
| trackon2 | TrackOn2 DINOv3 external baseline | 0.497728 | 67.040639 | 93.061465 |
| v1 | ReEntry-VisCalibrator V1 | 0.480024 | 64.095014 | 92.347688 |
| v22Q | V22Q interval/gate extension | 0.476936 | 64.238349 | 92.338242 |
| v24 | V24-DINOScore 0.395 | 0.473188 | 64.354510 | 92.471189 |

Query-weighted manifest values for generated ReEntry methods:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V1 | 0.5305 | 64.0950 | 92.3477 |
| V22Q | 0.5252 | 64.2383 | 92.3382 |
| V24-DINOScore | 0.5236 | 64.3545 | 92.4712 |

Note:

```text
The paired-stat table reports video-weighted means.
The manifest table reports query-weighted / cache-level evaluation outputs from the evaluation scripts.
Both show the same trend: V24 improves AJ/OA over V22Q but slightly reduces AJ_RD.
```

---

## 4. V24-DINOScore vs baselines on DAVIS

Paired differences of V24 against each baseline:

| Comparison | ΔAJ_RD | ΔAJ | ΔOA | Interpretation |
|---|---:|---:|---:|---|
| V24 - offline | +0.063380 | +2.188460 | +3.460525 | Strongly positive vs offline base |
| V24 - baseline | -0.006360 | +1.536446 | +2.208121 | AJ/OA strongly positive; AJ_RD slightly lower |
| V24 - trackon2 | -0.024540 | -2.686129 | -0.590276 | TrackOn2 remains stronger overall on DAVIS |
| V24 - V1 | -0.006836 | +0.259496 | +0.123501 | V24 improves AJ/OA over V1 but lowers AJ_RD |
| V24 - V22Q | -0.003748 | +0.116161 | +0.132948 | Same pattern as fresh: AJ/OA positive, AJ_RD negative |

Bootstrap/sign-test details:

| Comparison | Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---|---:|---:|---:|---:|
| V24 - offline | AJ_RD | +0.063380 | [0.038576, 0.090013] | 19 / 1 / 5 | 0.000040 |
| V24 - offline | AJ | +2.188460 | [1.248721, 3.242748] | 24 / 5 / 1 | 0.000546 |
| V24 - offline | OA | +3.460525 | [1.937991, 5.361877] | 27 / 2 / 1 | 0.000002 |
| V24 - baseline | AJ_RD | -0.006360 | [-0.020940, 0.008208] | 8 / 14 / 3 | 0.286279 |
| V24 - baseline | AJ | +1.536446 | [0.447824, 2.572093] | 22 / 8 / 0 | 0.016125 |
| V24 - baseline | OA | +2.208121 | [1.325593, 3.152698] | 22 / 5 / 3 | 0.001514 |
| V24 - V22Q | AJ_RD | -0.003748 | [-0.009124, -0.000288] | 1 / 6 / 18 | 0.125000 |
| V24 - V22Q | AJ | +0.116161 | [0.021575, 0.237639] | 7 / 6 / 17 | 1.000000 |
| V24 - V22Q | OA | +0.132948 | [0.003486, 0.292907] | 6 / 6 / 18 | 1.000000 |

---

## 5. DINOScore DAVIS decision statistics

DINOScore fixed policy:

```text
target = V22Q
feature = last_candidate_cosine
threshold = 0.395
direction = low_is_bad
```

DAVIS decision stats:

```text
frame-keep rows = 3613
selected visible rows = 3322
fires = 153
flipped = 148
flipped / scanned rows = 0.04096
```

This is less conservative than on fresh20-49, where flipped/all rows was roughly 0.004--0.006.

Interpretation:

```text
On DAVIS, the frozen DINOScore threshold fires on a larger fraction of candidate rows than on RGB-Stacking fresh20-49.
This may explain why it gives more AJ/OA gain but also a slightly clearer AJ_RD cost.
```

---

## 6. What this means for the paper

The DAVIS result strengthens the evidence that the ReEntry pipeline is not only a fresh20-49 artifact:

```text
Against CoTracker3 offline on DAVIS, V24 improves:
  AJ_RD by +0.0634
  AJ by +2.1885
  OA by +3.4605
```

However, the DAVIS result also shows that external TrackOn2 remains stronger overall on this DAVIS bridge protocol:

```text
TrackOn2 vs V24:
  TrackOn2 has higher AJ_RD, AJ, and OA.
```

Therefore the paper claim should be careful:

Recommended:

```text
On the DAVIS bridge benchmark, the ReEntry pipeline substantially improves over the CoTracker3 offline base, especially in AJ/OA, and preserves the same trade-off pattern observed on fresh: V24 increases AJ/OA over V22Q while slightly lowering AJ_RD.
```

Avoid:

```text
V24 is the best DAVIS method overall.
V24 beats TrackOn2 on DAVIS.
V24 improves every metric over V1/V22Q.
```

---

## 7. Final updated positioning after DAVIS

The previous positioning remains valid:

```text
V1 Learned:
  AJ_RD-oriented main method.
  On DAVIS, V1 has higher AJ_RD than V22Q/V24 among ReEntry variants.

V22Q:
  Stability-oriented extension.
  On DAVIS, it improves AJ over V1 but slightly lowers AJ_RD.

V24-DINOScore:
  Optional AJ/OA-oriented appearance micro-filter.
  On DAVIS, it further improves AJ/OA over V22Q but again lowers AJ_RD.
```

The DAVIS evidence supports the broader statement:

```text
ReEntry methods can improve standard benchmark behavior relative to the CoTracker3 offline base, but the DINOScore appearance micro-filter is still a trade-off extension rather than a universal replacement.
```

---

## 8. Recommended next step

Update the paper summary and table plan with this DAVIS section.

Then create a final table document that includes both:

```text
1. DAVIS bridge standard full 30-video comparison.
2. RGB-Stacking fresh20-49 re-entry stress/fresh comparison.
```

Suggested next file:

```text
docs/final_paper_tables_2026-07-04.md
```
