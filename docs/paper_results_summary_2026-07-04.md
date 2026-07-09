# ReEntry-TAP Paper Results Summary — 2026-07-04

## 0. Executive conclusion

Current evidence supports the following final method positioning:

```text
V1 Learned ReEntry-VisCalibrator
  Main method.
  Best positioned as the AJ_RD-oriented re-entry recovery method.

V22Q Interval/Gate Post-processing
  Stability extension.
  Best positioned as the safest extension when balancing AJ_RD, AJ, and OA.

V24-DINOScore 0.395
  Optional DINOv3 appearance micro-filter.
  It gives a real paired-stable AJ gain, but introduces a systematic tiny AJ_RD cost.
  It should not replace V22Q as the default stability method.
```

The strongest paper-ready claim is not that every extension monotonically improves every metric. The strongest claim is:

```text
A learned ReEntry-VisCalibrator improves recovery after re-entry events.
A conservative interval/gate post-processor stabilizes the method.
DINOv3 appearance cues provide a real but small AJ-oriented micro-filter, revealing a useful direction for future appearance verification.
```

---

## 1. Method map

| Name | Role | Current status | Paper positioning |
|---|---|---|---|
| Base / offline | Original CoTracker3 offline cache | Baseline | Required baseline |
| V1 Learned | Frame-level ReEntry-VisCalibrator | Main positive method | Main method |
| V2.1 Event Gate | Event-level no-action gate | Stable but superseded | Intermediate ablation |
| V22Q Interval | Event gate + interval trimming | Best stability-oriented extension | Stability extension |
| V2.3 MLP Frame-Keep | Learned keep/drop decoder | Negative / underfit distribution shift | Negative result / appendix |
| V2.3 RiskScore | Hand-coded numeric risk filter | AJ improves but trade-off | Diagnostic / optional analysis |
| Raw RGB Patch Similarity | Simple RGB patch verification | Too weak across videos | Negative result |
| Track-On2 DINOv3 Teacher | External teacher route | Runnable but weak under current stress smoke | Engineering feasibility / not main |
| V24-DINOScore 0.395 | DINOv3 appearance micro-filter | AJ stable positive, tiny AJ_RD cost | Optional extension / analysis |

---

## 2. Primary fresh20-49 results: V22Q vs V24-DINOScore

V24-DINOScore fixed policy:

```text
target = V22Q
feature = last_candidate_cosine
threshold = 0.395
direction = low_is_bad
```

### Query-weighted full-cache metrics

| Setting | Method | AJ_RD | AJ | OA |
|---|---|---:|---:|---:|
| natural | V22Q | 0.4520 | 79.2775 | 92.9233 |
| natural | V24-DINOScore | 0.4514 | 79.2884 | 92.9244 |
| natural | Δ V24 - V22Q | -0.0006 | +0.0109 | +0.0011 |
| translate_L16 | V22Q | 0.5350 | 74.9919 | 92.1943 |
| translate_L16 | V24-DINOScore | 0.5348 | 74.9987 | 92.1951 |
| translate_L16 | Δ V24 - V22Q | -0.0002 | +0.0068 | +0.0008 |
| occluder_L16 | V22Q | 0.6669 | 77.2218 | 92.2326 |
| occluder_L16 | V24-DINOScore | 0.6667 | 77.2276 | 92.2293 |
| occluder_L16 | Δ V24 - V22Q | -0.0002 | +0.0058 | -0.0033 |

Interpretation:

```text
V24-DINOScore consistently improves AJ in all three fresh settings.
However, it also consistently introduces a tiny AJ_RD cost.
OA is near-neutral: slightly positive on natural/translate, slightly negative on occluder.
```

Recommended paper phrasing:

```text
DINOv3 appearance verification yields a small but consistent AJ improvement, but we retain V22Q as the safest stability extension because the DINOScore filter slightly trades off AJ_RD.
```

---

## 3. Paired-video stats: V24-DINOScore vs V22Q

Artifacts:

```text
docs/reentry_v24_dinoscore_paired_stats_fresh20_49_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/paired_stats_natural_v24_vs_v22Q.json
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/paired_stats_translate_v24_vs_v22Q.json
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/paired_stats_o_v24_vs_v22Q.json
```

### Natural

| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD | -0.000433 | [-0.000853, -0.000107] | 3 / 11 / 16 | 0.057373 |
| AJ | +0.010910 | [0.003504, 0.021412] | 23 / 3 / 4 | 0.000088 |
| OA | +0.001039 | [-0.006727, 0.009888] | 11 / 14 / 5 | 0.690038 |

### Translate_L16

| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD | -0.000163 | [-0.000330, -0.000027] | 7 / 12 / 11 | 0.359283 |
| AJ | +0.006792 | [0.002758, 0.011738] | 23 / 5 / 2 | 0.000912 |
| OA | +0.000823 | [-0.005200, 0.007547] | 12 / 15 / 3 | 0.701108 |

### Occluder_L16

| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD | -0.000240 | [-0.000393, -0.000120] | 1 / 16 / 13 | 0.000275 |
| AJ | +0.005800 | [0.003057, 0.008610] | 26 / 2 / 2 | 0.000003 |
| OA | -0.003240 | [-0.009600, 0.002270] | 11 / 17 / 2 | 0.344928 |

Summary:

```text
AJ gains are paired-stable across all fresh settings.
AJ_RD losses are very small but systematic.
OA is statistically mixed / near-neutral.
```

Decision:

```text
Do not promote V24-DINOScore over V22Q as the default method.
Use it as an optional appearance micro-filter or analysis result.
```

---

## 4. Important dev7-9 evidence for V24-DINOScore

Dev7-9 was used to select the conservative DINOScore threshold.

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.3212 | 79.9763 | 94.0033 |
| V24-DINOScore 0.395 | 0.3212 | 79.9950 | 94.0168 |
| Δ | +0.0000 | +0.0187 | +0.0135 |

This showed clean dev behavior, but fresh20-49 reveals the more realistic conclusion:

```text
The DINOScore micro-filter is AJ-positive but not completely free in AJ_RD.
```

---

## 5. DINOv3 evidence chain

### 5.1 Availability

DINOv3 exists locally:

```text
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

It can be loaded as:

```text
DINOv3ViTModel
hidden_size = 384
image_size = 224
patch_size = 16
```

The Hugging Face fast processor was incompatible with torch 2.2.2 because it called a missing `torch.compiler.is_compiling`, so manual ImageNet preprocessing was used.

### 5.2 Feature-level diagnostic

On dev7-9 uniform3000:

| Label | Top DINOv3 AUC |
|---|---:|
| y_safe16 | 0.7188 |
| y_gt_visible | 0.7142 |
| y_safe8 | 0.7545 |
| y_utility / y_safe4 | 0.8266 |

Compared with raw RGB patch similarity:

```text
raw RGB patch cross-video top AUC ≈ 0.54--0.55
```

Conclusion:

```text
DINOv3 patch embeddings are a real appearance signal, much stronger than raw RGB patch similarity.
```

### 5.3 Threshold diagnostic

On dev7-9 uniform3000:

```text
last_candidate_cosine < 0.395
n_drop ≈ 1%
utility-negative precision = 1.0
safe16-positive loss ≈ 0.0004
```

This justified trying a conservative downstream micro-filter.

---

## 6. Negative and diagnostic results to preserve

### 6.1 V2.3 frame-keep MLP

Oracle upper bound was strong, but learned MLP did not translate to downstream improvement.

Key lesson:

```text
The V2.3 idea has headroom, but the current learning setup / feature mix did not generalize well enough.
```

Recommended placement:

```text
Appendix / negative result / diagnostic.
```

### 6.2 Numeric RiskScore

Aggressive RiskScore improved AJ beyond V22Q on fresh settings, but paid small AJ_RD/OA costs.

Conservative RiskScore recovered AJ_RD but erased most AJ gain.

Key lesson:

```text
Numeric risk signals are real, but hand-tuned numeric filtering sits on a trade-off boundary.
```

### 6.3 Raw RGB patch similarity

Raw RGB patch similarity initially looked strong in a prefix sample, but that was a single-video artifact.

Cross-video uniform sample showed:

```text
raw RGB top AUC ≈ 0.54--0.55
```

Key lesson:

```text
Frame access works, but raw RGB similarity is too weak. Strong representation features are needed.
```

### 6.4 Track-On2 DINOv3 teacher

Track-On2 DINOv3 can run, but the smoke result was weak under the ReEntry stress protocol.

Small dev0 translate smoke:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| B2-W16-P2 | 0.5373 | 55.4240 | 82.0030 |
| Track-On2 DINOv3 | 0.2643 | 29.0617 | 79.6938 |

Key lesson:

```text
Do not use current Track-On2 DINOv3 stress result as a strong teacher.
```

---

## 7. Recommended final paper tables

### Main table

Use the following rows:

```text
Base / CoTracker3 offline
V1 Learned ReEntry-VisCalibrator
V22Q Interval/Gate
```

Do not make V24 the default main-table winner unless the table explicitly allows trade-off variants.

### Optional / analysis table

Use:

```text
V22Q
V24-DINOScore 0.395
```

and report the paired differences:

```text
AJ gain: stable positive
AJ_RD: tiny systematic negative
OA: mixed / near-neutral
```

### Negative-result table / appendix

Include:

```text
V2.3 MLP frame-keep
RiskScore aggressive / conservative
Raw RGB patch similarity
Track-On2 DINOv3 teacher smoke
```

---

## 8. Final claims to use

Recommended claims:

```text
1. Re-entry visibility lag is a measurable failure mode in point tracking.
2. A learned ReEntry-VisCalibrator can improve recovery after re-entry.
3. Conservative event/interval post-processing gives the best stability-oriented extension.
4. DINOv3 appearance similarity provides a real signal and can yield a small paired-stable AJ improvement.
5. The current DINOScore filter is not free: it slightly trades off AJ_RD, so it remains an optional appearance micro-filter.
```

Avoid claiming:

```text
- V24-DINOScore is the new best default method.
- DINOScore improves every metric.
- Track-On2 DINOv3 is a strong teacher under the current protocol.
- Raw RGB patch similarity is sufficient.
```

---

## 9. Recommended next actions

### Immediate next action

Stop tuning on fresh20-49.

### Paper preparation

Create final paper tables from existing manifests and paired stats.

Suggested next document:

```text
docs/paper_tables_plan_2026-07-04.md
```

### Optional future validation

If stronger generalization evidence is needed, create a new untouched holdout:

```text
fresh50-79
```

Frozen policies only:

```text
Base
V1
V22Q
V24-DINOScore 0.395
```

No further tuning on the new holdout.

---

## 10. Standard benchmark: TAPVid-DAVIS bridge full 30-video comparison

Artifact:

```text
docs/standard_benchmark_davis_comparison_2026-07-04.md
```

Scope:

```text
Dataset/protocol: TAPVid-DAVIS first-input bridge cache
Videos: 30
Queries: 650
Re-entry queries: 259
Base cache: CoTracker3 offline DAVIS first-input bridge
Override cache for ReEntry methods: CoTracker3 baseline DAVIS first-input bridge
```

Video-weighted paired-stat means:

| Method | Role | AJ_RD | AJ | OA |
|---|---|---:|---:|---:|
| offline | CoTracker3 offline base | 0.409808 | 62.166050 | 89.010664 |
| baseline | CoTracker3 baseline / override | 0.479548 | 62.818065 | 90.263068 |
| trackon2 | TrackOn2 DINOv3 external baseline | 0.497728 | 67.040639 | 93.061465 |
| v1 | ReEntry-VisCalibrator V1 | 0.480024 | 64.095014 | 92.347688 |
| v22Q | V22Q interval/gate extension | 0.476936 | 64.238349 | 92.338242 |
| v24 | V24-DINOScore 0.395 | 0.473188 | 64.354510 | 92.471189 |

Key paired differences:

```text
V24 - offline:
  ΔAJ_RD = +0.063380
  ΔAJ    = +2.188460
  ΔOA    = +3.460525

V24 - baseline:
  ΔAJ_RD = -0.006360
  ΔAJ    = +1.536446
  ΔOA    = +2.208121

V24 - V22Q:
  ΔAJ_RD = -0.003748
  ΔAJ    = +0.116161
  ΔOA    = +0.132948
```

Interpretation:

```text
The DAVIS bridge result strengthens the evidence that the ReEntry pipeline is not only a fresh20-49 artifact: V24 substantially improves over the CoTracker3 offline base on AJ_RD, AJ, and OA.
However, TrackOn2 remains stronger overall on this DAVIS bridge protocol, and V24 still follows the same trade-off pattern: higher AJ/OA than V22Q but lower AJ_RD.
```

Updated final positioning remains:

```text
V1 Learned: AJ_RD-oriented main method.
V22Q: stability-oriented extension.
V24-DINOScore: optional AJ/OA-oriented appearance micro-filter.
```

---

## 11. Standard/full benchmark: TAPVid RGB-Stacking full50

Artifact:

```text
docs/standard_benchmark_rgb_full50_comparison_2026-07-04.md
```

Scope:

```text
Dataset: TAPVid RGB-Stacking
Split: rgb_stacking_000000 -- rgb_stacking_000049
Videos: 50
Queries: 60,829
Re-entry queries: 10,585
Base cache: CoTracker3 offline RGB-Stacking full50
Override cache: CoTracker3 online RGB-Stacking full50
```

Query-weighted results:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline Base | 0.3617 | 79.9345 | 91.6371 |
| online | 0.4028 | 44.8432 | 55.8092 |
| V1 | 0.4414 | 79.4302 | 93.0758 |
| V22Q | 0.4400 | 79.5756 | 93.0481 |
| V24-DINOScore | 0.4394 | 79.5974 | 93.0389 |

Key deltas vs offline Base:

```text
V1 - Base:
  ΔAJ_RD = +0.0797
  ΔAJ    = -0.5043
  ΔOA    = +1.4387

V22Q - Base:
  ΔAJ_RD = +0.0783
  ΔAJ    = -0.3589
  ΔOA    = +1.4110

V24 - Base:
  ΔAJ_RD = +0.0777
  ΔAJ    = -0.3371
  ΔOA    = +1.4018
```

Video-weighted paired-stat means show the same trend:

```text
V24 - offline:
  ΔAJ_RD = +0.074850, CI [0.060516, 0.090088], wins/losses/ties = 48/0/2
  ΔAJ    = -0.337109, CI [-0.624740, -0.048121], wins/losses/ties = 11/39/0
  ΔOA    = +1.401793, CI [0.940280, 1.873843], wins/losses/ties = 40/10/0
```

Interpretation:

```text
This is the strongest full standard evidence for the current ReEntry pipeline. On TAPVid RGB-Stacking full50, ReEntry substantially improves AJ_RD and OA over the CoTracker3 offline base, at the cost of a small standard-AJ reduction. V1 remains the best AJ_RD-oriented method; V24 is still an optional AJ-oriented micro-filter, not the default main method.
```

---

## 12. Official-style DAVIS first/input protocol

Artifacts:

```text
docs/official_protocol_davis_first_input_comparison_2026-07-04.md
docs/evaluation_protocol_section_2026-07-04.md
```

Scope:

```text
Dataset: TAPVid-DAVIS
Videos: 30
Queries: 650
Query mode: first
Metric resolution: input / 256
Metrics: AJ / OA / δ_avg / AJ_RD
Evaluation type: official-style local evaluation, not official leaderboard submission
```

Main results:

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 baseline | 64.8949 | 91.7983 | 77.3560 | 0.3486 |
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 0.3142 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 0.3714 |
| V1 | 64.5758 | 91.7274 | 77.2244 | 0.3588 |
| V22Q | 64.7260 | 91.7406 | 77.2244 | 0.3556 |
| V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 0.3549 |

Key deltas vs CoTracker3 offline:

```text
V1:
  ΔAJ    = +1.9193
  ΔOA    = +3.5787
  ΔAJ_RD = +0.0446

V22Q:
  ΔAJ    = +2.0695
  ΔOA    = +3.5919
  ΔAJ_RD = +0.0414

V24:
  ΔAJ    = +2.1874
  ΔOA    = +3.7363
  ΔAJ_RD = +0.0407
```

Interpretation:

```text
Under DAVIS first/input official-style local evaluation, ReEntry improves AJ/OA/AJ_RD over CoTracker3 offline. V24 is the strongest ReEntry variant on AJ/OA, while TrackOn2 remains the strongest external baseline overall.
```
