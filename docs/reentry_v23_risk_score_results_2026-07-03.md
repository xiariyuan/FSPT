# ReEntry-VisCalibrator V2.3 RiskScore Results — 2026-07-03

## 1. Goal

V2.3 learned frame-keep MLP did not approach the strong oracle upper bound. Feature analysis showed that a few interpretable numeric signals are useful:

```text
event_gate_prob
frame_base_override_dist_norm
v1_prob
segment probability statistics
```

V2.3-RiskScore tests whether a low-capacity, transparent filter based on these signals can beat V22Q without training a new model.

---

## 2. New code

```text
scripts/eval_reentry_viscalibrator_v23_risk_score.py
```

The script:

```text
1. uses V1 to propose recovery frames;
2. uses V2.1 event gate to compute event_gate_prob;
3. reads per-frame V1/segment/distance features;
4. drops proposed recovery frames according to a named policy;
5. keeps base/offline coordinates unchanged;
6. evaluates AJ_RD / AJ / OA.
```

No GT is used at inference.

---

## 3. Dev7-9 policy sweep

Reference metrics:

```text
V1 dev7-9:    AJ_RD 0.3213, AJ 79.7517, OA 94.0018
V2.1 dev7-9:  AJ_RD 0.3216, AJ 79.9493, OA 94.0191
V22Q dev7-9:  AJ_RD 0.3212, AJ 79.9763, OA 94.0033
```

Policy sweep:

| Policy | AJ_RD | AJ | OA | Keep rate | Drop rate |
|---|---:|---:|---:|---:|---:|
| none | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 0.0000 |
| event_lt_0005 | 0.3216 | 79.9035 | 94.0431 | 0.9485 | 0.0515 |
| event_lt_001 | 0.3216 | 79.9493 | 94.0191 | 0.9293 | 0.0707 |
| dist_ge_025 | 0.3196 | 79.8982 | 93.9941 | 0.9492 | 0.0508 |
| dist_ge_050 | 0.3216 | 79.8484 | 93.9696 | 0.9701 | 0.0299 |
| event0005_or_dist025 | 0.3213 | 79.9597 | 94.0495 | 0.9293 | 0.0707 |
| event001_or_dist025 | 0.3215 | 79.9917 | 94.0135 | 0.9149 | 0.0851 |
| dist025_and_v1lt050 | 0.3196 | 79.8928 | 93.9907 | 0.9517 | 0.0483 |
| dist025_and_v1lt030 | 0.3198 | 79.8352 | 94.0212 | 0.9697 | 0.0303 |
| event001_and_v1lt030 | 0.3204 | 79.8572 | 94.0058 | 0.9581 | 0.0419 |
| event002_and_dist025_and_v1lt050 | 0.3197 | 79.8689 | 93.9998 | 0.9594 | 0.0406 |
| score_ge_1.0 | 0.3201 | 80.1238 | 93.7578 | 0.8366 | 0.1634 |
| score_ge_1.5 | 0.3205 | 79.9293 | 94.0142 | 0.9391 | 0.0609 |
| score_ge_2.0 | 0.3201 | 79.8955 | 93.9823 | 0.9501 | 0.0499 |
| softscore_ge_1.0 | 0.3200 | 80.0557 | 93.8305 | 0.8772 | 0.1228 |
| softscore_ge_1.5 | 0.3200 | 79.9559 | 93.9211 | 0.9236 | 0.0764 |
| softscore_ge_2.0 | 0.3201 | 79.9045 | 93.9531 | 0.9457 | 0.0543 |

Selected policy for fresh sanity:

```text
event001_or_dist025:
  drop if event_gate_prob < 0.01 OR base_override_dist_norm >= 0.25
```

Reason:

```text
It beats V22Q on dev7-9 AJ while preserving AJ_RD and OA:
  AJ_RD 0.3215 >= V22Q 0.3212
  AJ 79.9917 > V22Q 79.9763
  OA 94.0135 > V22Q 94.0033
```

---

## 4. Fresh20-49 query-weighted results

### V23-RiskScore policy

```text
event001_or_dist025:
  drop if event_gate_prob < 0.01 OR base_override_dist_norm >= 0.25
```

| Setting | V1 AJ_RD | V22Q AJ_RD | V23-Risk AJ_RD | V1 AJ | V22Q AJ | V23-Risk AJ | V1 OA | V22Q OA | V23-Risk OA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 0.4533 | 0.4520 | 0.4514 | 79.1867 | 79.2775 | 79.2997 | 92.9234 | 92.9233 | 92.9076 |
| translate_L16 | 0.5354 | 0.5350 | 0.5345 | 74.8798 | 74.9919 | 75.0304 | 92.2390 | 92.1943 | 92.1662 |
| occluder_L16 | 0.6671 | 0.6669 | 0.6664 | 77.1320 | 77.2218 | 77.2536 | 92.2273 | 92.2326 | 92.2322 |

### V23-Risk minus V1

| Setting | ΔAJ_RD | ΔAJ | ΔOA | Dropped changed frames |
|---|---:|---:|---:|---:|
| natural | -0.0019 | +0.1130 | -0.0158 | 22206 |
| translate_L16 | -0.0009 | +0.1506 | -0.0728 | 28522 |
| occluder_L16 | -0.0007 | +0.1216 | +0.0049 | 22360 |

### V23-Risk minus V22Q

| Setting | ΔAJ_RD | ΔAJ | ΔOA |
|---|---:|---:|---:|
| natural | -0.0006 | +0.0222 | -0.0157 |
| translate_L16 | -0.0005 | +0.0385 | -0.0281 |
| occluder_L16 | -0.0005 | +0.0318 | -0.0004 |

Interpretation:

```text
V23-RiskScore improves AJ beyond V22Q on all three fresh settings.
However, it pays a small AJ_RD cost and a small-to-moderate OA cost, especially on translate_L16.
```

---

## 5. Paired video-level results

Paired outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/risk_score_paired_stats/rgb_fresh20_49_natural_event001_or_dist025.json
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/risk_score_paired_stats/fresh20_49_translate_L16_event001_or_dist025.json
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/risk_score_paired_stats/fresh20_49_occluder_L16_event001_or_dist025.json
```

### V23-Risk minus V1, AJ

| Setting | Mean ΔAJ | 95% CI | Wins / Losses / Ties |
|---|---:|---:|---:|
| natural | +0.113017 | [0.077070, 0.148921] | 28 / 2 / 0 |
| translate_L16 | +0.150618 | [0.107355, 0.195561] | 28 / 2 / 0 |
| occluder_L16 | +0.121530 | [0.075460, 0.161246] | 29 / 1 / 0 |

### V23-Risk minus V22Q, AJ

| Setting | Mean ΔAJ | 95% CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.022184 | [0.001938, 0.045829] | 17 / 13 / 0 | 0.5847 |
| translate_L16 | +0.038453 | [0.014432, 0.065106] | 23 / 7 / 0 | 0.0052 |
| occluder_L16 | +0.031799 | [0.000120, 0.060434] | 23 / 7 / 0 | 0.0052 |

Interpretation:

```text
AJ improvement over V1 is very stable.
AJ improvement over V22Q is stable and statistically supported on translate_L16 and occluder_L16.
Natural improves on average but wins/losses are mixed.
```

### V23-Risk minus V22Q, AJ_RD

| Setting | Mean ΔAJ_RD | 95% CI | Wins / Losses / Ties |
|---|---:|---:|---:|
| natural | -0.001063 | [-0.002530, 0.000150] | 11 / 16 / 3 |
| translate_L16 | -0.000440 | [-0.001110, 0.000147] | 14 / 13 / 3 |
| occluder_L16 | -0.000563 | [-0.001297, 0.000017] | 10 / 13 / 7 |

Interpretation:

```text
AJ_RD cost relative to V22Q is small, but consistently negative in mean.
```

### V23-Risk minus V22Q, OA

| Setting | Mean ΔOA | 95% CI | Wins / Losses / Ties |
|---|---:|---:|---:|
| natural | -0.015719 | [-0.046884, 0.015648] | 9 / 20 / 1 |
| translate_L16 | -0.028070 | [-0.072545, 0.011269] | 6 / 24 / 0 |
| occluder_L16 | -0.000360 | [-0.046837, 0.040619] | 17 / 12 / 1 |

Interpretation:

```text
OA cost is the main weakness. Translate_L16 shows the clearest OA degradation relative to V22Q.
```

---

## 6. Current conclusion

V23-RiskScore succeeds as a diagnostic:

```text
The interpretable risk signals from feature analysis are real.
A simple rule improves AJ beyond V22Q on all three fresh settings.
```

But it should not replace V1 or V22Q as final method yet:

```text
V1 still has the best AJ_RD.
V22Q has a better recovery-stability balance than this RiskScore policy because V23-Risk pays extra AJ_RD/OA cost.
```

Best current positioning:

```text
V1 Learned = AJ_RD-oriented main method.
V22Q = best current stability-oriented extension.
V23-RiskScore = interpretable diagnostic showing that event gate + coordinate agreement can further improve AJ, but with extra AJ_RD/OA cost.
```

---

## 7. Recommended next step

The selected RiskScore policy is slightly too aggressive.

Next step should be a smaller, constrained policy search around `event001_or_dist025`, not a new MLP:

```text
Goal:
  recover most of the V23-Risk AJ gain
  reduce AJ_RD/OA cost
```

Most promising variants:

```text
1. event001_or_dist050:
   drop if event_gate_prob < 0.01 OR base_override_dist_norm >= 0.50

2. event0005_or_dist025:
   already tested on dev: AJ 79.9597, OA 94.0495
   more conservative than selected policy

3. event001_or_dist025_and_v1lt050:
   drop distance-risk frames only when v1_prob < 0.50, but always drop event_gate_prob < 0.01

4. event001_or_dist025_and_segmaxlt050:
   drop distance-risk frames only when segment_prob_max < 0.50

5. score_ge_1.5 with tuned OA constraint:
   dev AJ lower than selected policy, but OA better
```

Selection criterion on dev7-9:

```text
AJ_RD >= 0.3212
OA >= 94.00
maximize AJ
```

If a conservative policy preserves V22Q-level AJ_RD/OA while giving some AJ gain, then run fresh.

If not, V22Q remains the best stability variant, and the next real improvement likely requires appearance verifier / external tracker teacher.
