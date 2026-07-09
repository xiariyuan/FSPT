# ReEntry V2.4-DINOScore Fresh20-49 Paired Video Statistics — 2026-07-03

## 1. Purpose

This document summarizes paired per-video statistics for the fixed V24-DINOScore policy against V22Q on fresh20-49.

Fixed policy:

```text
target = V22Q
feature = last_candidate_cosine
threshold = 0.395
direction = low_is_bad
```

Compared methods:

```text
V22Q
V24-DINOScore 0.395
```

Artifacts:

```text
docs/reentry_v24_dinoscore_paired_stats_natural_2026-07-03.md
docs/reentry_v24_dinoscore_paired_stats_translate_2026-07-03.md
docs/reentry_v24_dinoscore_paired_stats_o_2026-07-03.md

outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/paired_stats_natural_v24_vs_v22Q.json
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/paired_stats_translate_v24_vs_v22Q.json
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/paired_stats_o_v24_vs_v22Q.json
```

---

## 2. Natural fresh20-49

Video-weighted means:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.403180 | 79.277535 | 92.923321 |
| V24-DINOScore | 0.402747 | 79.288445 | 92.924360 |
| Δ | -0.000433 | +0.010910 | +0.001039 |

Paired stats:

| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD | -0.000433 | [-0.000853, -0.000107] | 3 / 11 / 16 | 0.057373 |
| AJ | +0.010910 | [0.003504, 0.021412] | 23 / 3 / 4 | 0.000088 |
| OA | +0.001039 | [-0.006727, 0.009888] | 11 / 14 / 5 | 0.690038 |

Interpretation:

```text
Natural shows a paired-stable AJ gain, neutral OA, and a small negative AJ_RD shift.
```

---

## 3. Translate_L16 fresh20-49

Video-weighted means:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.556553 | 74.991948 | 92.194259 |
| V24-DINOScore | 0.556390 | 74.998739 | 92.195082 |
| Δ | -0.000163 | +0.006792 | +0.000823 |

Paired stats:

| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD | -0.000163 | [-0.000330, -0.000027] | 7 / 12 / 11 | 0.359283 |
| AJ | +0.006792 | [0.002758, 0.011738] | 23 / 5 / 2 | 0.000912 |
| OA | +0.000823 | [-0.005200, 0.007547] | 12 / 15 / 3 | 0.701108 |

Interpretation:

```text
Translate_L16 shows a paired-stable AJ gain, neutral OA, and a very small negative AJ_RD shift.
```

---

## 4. Occluder_L16 fresh20-49

Video-weighted means:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.676490 | 77.221772 | 92.232550 |
| V24-DINOScore | 0.676250 | 77.227572 | 92.229310 |
| Δ | -0.000240 | +0.005800 | -0.003240 |

Paired stats:

| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD | -0.000240 | [-0.000393, -0.000120] | 1 / 16 / 13 | 0.000275 |
| AJ | +0.005800 | [0.003057, 0.008610] | 26 / 2 / 2 | 0.000003 |
| OA | -0.003240 | [-0.009600, 0.002270] | 11 / 17 / 2 | 0.344928 |

Interpretation:

```text
Occluder_L16 has the most stable AJ gain, but also the clearest systematic small AJ_RD cost.
OA is slightly negative on average but the CI crosses zero.
```

---

## 5. Overall paired conclusion

Across all three fresh20-49 settings:

```text
AJ:
  consistently positive and paired-stable.
  natural:   +0.010910, wins/losses/ties 23/3/4
  translate: +0.006792, wins/losses/ties 23/5/2
  occluder:  +0.005800, wins/losses/ties 26/2/2

AJ_RD:
  consistently slightly negative.
  natural:   -0.000433
  translate: -0.000163
  occluder:  -0.000240

OA:
  near-neutral / mixed.
  natural:   +0.001039
  translate: +0.000823
  occluder:  -0.003240
```

Therefore:

```text
V24-DINOScore 0.395 is a real AJ-improving appearance micro-filter.
However, it introduces a systematic tiny AJ_RD cost.
It should not replace V22Q as the safest stability-oriented extension.
```

Recommended positioning:

```text
V1 Learned: AJ_RD-oriented main method.
V22Q: best stability-oriented extension.
V24-DINOScore: optional AJ-oriented appearance micro-filter / analysis result.
```

If used in paper tables, it should be clearly marked as a trade-off extension rather than the new default method.

---

## 6. Recommended next step

Do not tune DINOScore further on fresh20-49.

Next recommended action:

```text
Freeze method development and consolidate paper-ready results.
```

If additional evidence is required later, create an untouched holdout such as fresh50-79 and evaluate the already-frozen policies there:

```text
V1
V22Q
V24-DINOScore 0.395
```
