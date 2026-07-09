# B2 False-Trigger Taxonomy — 2026-06-28

## Goal

Explain why B2 can keep standard AJ high even though its trigger precision is only moderate.

Current B2 trigger:

```text
base invisible run >= 1 and override visible
pre = 1
post = full
```

## Trigger counts

```text
total_tracks = 5882
gt_reentry_tracks = 1385
triggered_tracks = 2517
true_trigger_tracks = 1334
false_trigger_tracks = 1183
no_trigger_reentry_tracks = 51
nontrigger_nonreentry_tracks = 3314
```

Track-level trigger precision / recall:

```text
precision = 1334 / 2517 = 0.529996
recall = 1334 / 1385 = 0.963177
```

## False-trigger cost summary

Among the 1183 false-trigger tracks:

```text
gt_visible_at_trigger = 700
gt_invisible_at_trigger = 483
low_cost_full_delta_ge_-0.01 = 612
harmful_full_delta_lt_-0.05 = 438
severe_full_delta_lt_-0.10 = 266
```

Rates:

```text
low_cost_rate = 0.517329
harmful_rate = 0.370245
severe_rate = 0.224852
```

Full-query AJ delta, B2 minus fixed, on false triggers:

```text
mean = -0.024486
median = -0.003385
p10 = -0.255542
p90 = 0.167436
```

Interpretation:

> Most false triggers are not catastrophic. The median full-query AJ cost is only about -0.0034, and more than half of false triggers are low-cost. The remaining standard-AJ loss comes from a concentrated harmful subset.

## True-trigger cost / benefit summary

Among the 1334 true-trigger tracks:

```text
full-query delta mean = +0.000439
full-query delta median = -0.002606
post-trigger delta mean = +0.025194
post-trigger delta median = 0.0
```

True triggers are not uniformly positive on full-query AJ, but they preserve the re-entry behavior that drives AJ_RD. The benefit is clearer in post-trigger / re-entry-focused metrics than in full-track standard AJ.

## Harmful false triggers concentrate in specific videos

Top harmful false-trigger videos:

| video | false triggers | harmful false triggers | severe false triggers | false-trigger rate | mean false full delta |
|---|---:|---:|---:|---:|---:|
| drift-chicane | 105 | 83 | 77 | 0.766423 | -0.210977 |
| mbike-trick | 126 | 68 | 24 | 0.400000 | -0.056804 |
| soapbox | 76 | 51 | 34 | 0.226190 | -0.110243 |
| pigs | 69 | 44 | 39 | 0.423313 | -0.079045 |
| gold-fish | 110 | 32 | 12 | 0.330330 | -0.000241 |
| parkour | 84 | 29 | 21 | 0.387097 | -0.035943 |
| shooting | 43 | 23 | 15 | 0.443299 | -0.078764 |
| bike-packing | 101 | 21 | 14 | 0.461187 | +0.012359 |
| drift-straight | 21 | 16 | 2 | 0.225806 | -0.063296 |

This aligns with the per-video standard-AJ drops. The largest failure is `drift-chicane`, where false triggers are frequent and strongly harmful.

## Important nuance

False-trigger precision alone underestimates B2 quality. A false trigger can still be low-cost or even beneficial for standard AJ if:

```text
1. base was already uncertain / invisible,
2. override is close to base or GT,
3. the affected frames have low contribution to full-track AJ,
4. override improves post-trigger continuity even without a GT re-entry event.
```

This explains why precision is only about 53%, while B2 still recovers standard AJ from global B1's 47.40 to 68.97.

## Main remaining failure mode

B2's remaining standard-AJ loss versus fixed_offline is not caused by all false triggers. It is concentrated in a harmful subset, especially:

```text
drift-chicane
pigs
shooting
soapbox
drift-straight
```

These should be the first qualitative/failure cases to inspect.

## Paper-facing interpretation

B2 should be described as a high-recall re-entry trigger with low-cost false positives:

> B2 favors re-entry recall. False triggers are common but often low-cost because they occur when the base tracker is already uncertain. The remaining standard-AJ loss is concentrated in a small set of identifiable failure videos.

## Next step

Export qualitative cases for:

```text
1. Successful B2 recovery where fixed_offline misses re-entry.
2. B2 avoiding global B1's standard-AJ collapse.
3. Harmful false trigger examples, especially drift-chicane / soapbox / pigs.
```
