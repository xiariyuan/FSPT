# Paper Tables Plan — 2026-07-04

This document converts the current ReEntry-TAP evidence package into paper-ready tables.

Primary summary document:

```text
docs/paper_results_summary_2026-07-04.md
```

---

## Table 1. Main fresh20-49 results

Purpose:

```text
Show the main method and the safest stability-oriented extension.
```

Recommended rows:

```text
Base / CoTracker3 offline
V1 Learned ReEntry-VisCalibrator
V22Q Interval/Gate
```

Optional row:

```text
V24-DINOScore 0.395
```

Recommended caption:

```text
Fresh20-49 evaluation across natural RGB-Stacking and two re-entry stress variants. V1 is the main re-entry recovery model. V22Q is the stability-oriented post-processing extension. V24-DINOScore is included as an optional appearance-based micro-filter and should be interpreted as an AJ-oriented trade-off variant.
```

Suggested columns:

```text
Setting
Method
AJ_RD ↑
AJ ↑
OA ↑
ΔAJ_RD vs V22Q
ΔAJ vs V22Q
ΔOA vs V22Q
```

Known V22Q / V24 values:

| Setting | Method | AJ_RD | AJ | OA | ΔAJ_RD vs V22Q | ΔAJ vs V22Q | ΔOA vs V22Q |
|---|---|---:|---:|---:|---:|---:|---:|
| natural | V22Q | 0.4520 | 79.2775 | 92.9233 | 0 | 0 | 0 |
| natural | V24-DINOScore | 0.4514 | 79.2884 | 92.9244 | -0.0006 | +0.0109 | +0.0011 |
| translate_L16 | V22Q | 0.5350 | 74.9919 | 92.1943 | 0 | 0 | 0 |
| translate_L16 | V24-DINOScore | 0.5348 | 74.9987 | 92.1951 | -0.0002 | +0.0068 | +0.0008 |
| occluder_L16 | V22Q | 0.6669 | 77.2218 | 92.2326 | 0 | 0 | 0 |
| occluder_L16 | V24-DINOScore | 0.6667 | 77.2276 | 92.2293 | -0.0002 | +0.0058 | -0.0033 |

Action needed:

```text
Fill Base and V1 rows from existing manifests / paired stats before final paper export.
```

---

## Table 2. Paired-video statistics for V24-DINOScore vs V22Q

Purpose:

```text
Show that V24-DINOScore has a real AJ gain but also a small AJ_RD cost.
```

Suggested caption:

```text
Paired per-video differences of V24-DINOScore 0.395 against V22Q on fresh20-49. DINOScore consistently improves AJ but introduces a small systematic AJ_RD cost. OA is near-neutral.
```

| Setting | Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---|---:|---:|---:|---:|
| natural | AJ_RD | -0.000433 | [-0.000853, -0.000107] | 3 / 11 / 16 | 0.057373 |
| natural | AJ | +0.010910 | [0.003504, 0.021412] | 23 / 3 / 4 | 0.000088 |
| natural | OA | +0.001039 | [-0.006727, 0.009888] | 11 / 14 / 5 | 0.690038 |
| translate_L16 | AJ_RD | -0.000163 | [-0.000330, -0.000027] | 7 / 12 / 11 | 0.359283 |
| translate_L16 | AJ | +0.006792 | [0.002758, 0.011738] | 23 / 5 / 2 | 0.000912 |
| translate_L16 | OA | +0.000823 | [-0.005200, 0.007547] | 12 / 15 / 3 | 0.701108 |
| occluder_L16 | AJ_RD | -0.000240 | [-0.000393, -0.000120] | 1 / 16 / 13 | 0.000275 |
| occluder_L16 | AJ | +0.005800 | [0.003057, 0.008610] | 26 / 2 / 2 | 0.000003 |
| occluder_L16 | OA | -0.003240 | [-0.009600, 0.002270] | 11 / 17 / 2 | 0.344928 |

---

## Table 3. V24-DINOScore decision statistics

Purpose:

```text
Show the micro-filter is conservative: it flips only ~0.4--0.6% of candidate rows.
```

Suggested caption:

```text
DINOScore decision statistics for the fixed threshold 0.395. The filter only applies to V22Q-visible candidate rows and flips a small fraction back to invisible.
```

| Setting | Frame-keep rows | Selected visible rows | Fires | Flipped rows | Flipped / all rows |
|---|---:|---:|---:|---:|---:|
| natural | 363629 | 347352 | 2294 | 2195 | 0.00604 |
| translate_L16 | 348013 | 328325 | 1455 | 1384 | 0.00398 |
| occluder_L16 | 342766 | 328562 | 2087 | 2019 | 0.00589 |

---

## Table 4. DINOv3 feature diagnostic

Purpose:

```text
Show that DINOv3 appearance embeddings are a real signal and much stronger than raw RGB patch similarity.
```

Suggested caption:

```text
Feature-level AUC of DINOv3 patch embeddings on dev7-9 V2.3 frame-keep samples. DINOv3 is much stronger than raw RGB patch similarity and provides appearance information complementary to numeric features.
```

| Label | Top DINOv3 AUC | Note |
|---|---:|---|
| y_safe16 | 0.7188 | Best reference-candidate similarity |
| y_gt_visible | 0.7142 | Best reference-candidate similarity |
| y_safe8 | 0.7545 | Last-visible candidate similarity |
| y_utility / y_safe4 | 0.8266 | Strongest DINOv3 signal |
| raw RGB patch cross-video | ~0.54--0.55 | Too weak for V2.4 |

---

## Table 5. Negative and diagnostic results

Purpose:

```text
Prevent overclaiming and show careful method selection.
```

Suggested caption:

```text
Summary of explored alternatives and why they were not promoted as the final method.
```

| Attempt | Result | Final decision |
|---|---|---|
| V2.3 frame-keep MLP | Oracle high, learned model did not improve downstream metrics | Negative / appendix |
| RiskScore aggressive | AJ improves but AJ_RD/OA cost appears | Diagnostic trade-off |
| RiskScore conservative | AJ_RD mostly recovered, AJ gain almost disappears | Not promoted |
| Raw RGB patch similarity | Cross-video AUC only ~0.54--0.55 | Too weak |
| Track-On2 DINOv3 teacher | Runnable but weak under stress smoke | Not a strong teacher |
| DINOScore 0.453 | Higher AJ, but AJ_RD/OA drop | Not main policy |

---

## Recommended narrative order

1. Define re-entry visibility lag and AJ_RD.
2. Show Base failure and V1 improvement.
3. Introduce V22Q as stability-oriented post-processing.
4. Present V22Q fresh results as the safest extension.
5. Present V24-DINOScore as an appearance analysis / optional AJ-oriented micro-filter.
6. Include negative-result table to justify method choices.
7. Mention future untouched holdout fresh50-79 if stronger generalization evidence is needed.

---

## Final recommended claim wording

Use:

```text
DINOv3 appearance verification provides a small but statistically paired-stable AJ improvement on top of V22Q, while introducing a tiny AJ_RD cost. We therefore treat it as an optional appearance micro-filter rather than the default stability method.
```

Do not use:

```text
DINOScore is the new best method.
DINOScore improves all metrics.
DINOScore replaces V22Q.
```

---

## Table 6. TAPVid-DAVIS bridge standard benchmark

Purpose:

```text
Add a standard benchmark comparison beyond RGB-Stacking fresh20-49.
```

Artifact:

```text
docs/standard_benchmark_davis_comparison_2026-07-04.md
```

Recommended caption:

```text
TAPVid-DAVIS bridge full 30-video comparison. ReEntry methods are applied using CoTracker3 offline as the base and CoTracker3 baseline as the override. V24-DINOScore substantially improves over the offline base, while TrackOn2 remains the strongest external baseline overall.
```

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline | 0.409808 | 62.166050 | 89.010664 |
| baseline | 0.479548 | 62.818065 | 90.263068 |
| trackon2 | 0.497728 | 67.040639 | 93.061465 |
| V1 | 0.480024 | 64.095014 | 92.347688 |
| V22Q | 0.476936 | 64.238349 | 92.338242 |
| V24-DINOScore | 0.473188 | 64.354510 | 92.471189 |

Recommended interpretation:

```text
This table should be used to show general benchmark credibility, not to claim V24 is the best overall DAVIS method. It improves strongly over the CoTracker3 offline base, but TrackOn2 remains stronger on DAVIS.
```

---

## Table 7. TAPVid RGB-Stacking full50 standard/full benchmark

Purpose:

```text
Provide the strongest full standard benchmark result for the ReEntry pipeline.
```

Artifact:

```text
docs/standard_benchmark_rgb_full50_comparison_2026-07-04.md
```

Recommended caption:

```text
TAPVid RGB-Stacking full50 comparison over videos rgb_stacking_000000--000049. ReEntry methods use CoTracker3 offline as the base and CoTracker3 online as the visibility override. V1 gives the largest AJ_RD improvement, while V24-DINOScore slightly improves AJ over V22Q but retains the small AJ_RD/OA trade-off.
```

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline Base | 0.3617 | 79.9345 | 91.6371 |
| online | 0.4028 | 44.8432 | 55.8092 |
| V1 | 0.4414 | 79.4302 | 93.0758 |
| V22Q | 0.4400 | 79.5756 | 93.0481 |
| V24-DINOScore | 0.4394 | 79.5974 | 93.0389 |

Recommended interpretation:

```text
Use this as the main full-standard RGB-Stacking result: ReEntry gives a large AJ_RD/OA improvement over the offline base, with a small AJ trade-off.
```

---

## Table 8. DAVIS first/input official-style local evaluation

Purpose:

```text
Add query-first DAVIS protocol evidence, matching the common TAP-Vid reporting style more closely than bridge-only results.
```

Artifact:

```text
docs/official_protocol_davis_first_input_comparison_2026-07-04.md
```

Caption:

```text
TAPVid-DAVIS first-query/input-resolution official-style local evaluation. ReEntry improves over the CoTracker3 offline base in AJ, OA, and AJ_RD, while TrackOn2 remains the strongest external baseline overall.
```

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 baseline | 64.8949 | 91.7983 | 77.3560 | 0.3486 |
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 0.3142 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 0.3714 |
| V1 | 64.5758 | 91.7274 | 77.2244 | 0.3588 |
| V22Q | 64.7260 | 91.7406 | 77.2244 | 0.3556 |
| V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 0.3549 |
