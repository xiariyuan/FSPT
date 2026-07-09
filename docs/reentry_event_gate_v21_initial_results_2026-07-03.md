# ReEntry-VisCalibrator V2.1 Event Gate — Initial Results — 2026-07-03

## 1. Goal

V2.1 introduces an event-level no-action gate before the frame-level ReEntry-VisCalibrator V1 recovery.

The goal is not to recover more frames. The goal is:

```text
keep most AJ_RD recovery from V1
block unnecessary / harmful recovery events
improve standard AJ
```

This is motivated by failure cases where Base is already correct but deterministic or learned recovery still modifies visibility and hurts standard tracking.

---

## 2. New code

```text
utils/reentry_event_gate_features.py
scripts/build_reentry_event_gate_dataset.py
scripts/train_reentry_event_gate.py
scripts/eval_reentry_viscalibrator_v2_event_gate.py
```

All scripts passed:

```bash
python -m py_compile \
  utils/reentry_event_gate_features.py \
  scripts/build_reentry_event_gate_dataset.py \
  scripts/train_reentry_event_gate.py \
  scripts/eval_reentry_viscalibrator_v2_event_gate.py
```

---

## 3. Dataset construction

Train split:

```text
base:     outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_offline_rgb_dev0_6.pt
override: outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_online_rgb_dev0_6.pt
```

Validation split:

```text
base:     outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_offline_rgb_dev7_9.pt
override: outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_online_rgb_dev7_9.pt
```

V1 model:

```text
outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1_dev0_6_clean.pt
```

V1 frame threshold:

```text
0.10
```

Event dataset stats:

| Split | Events | Feature dim | Positive rate | Mean utility | Positive utility sum | Negative utility sum |
|---|---:|---:|---:|---:|---:|---:|
| dev0-6 train | 8666 | 129 | 0.3763 | -1.8754 | 21786.0 | -38038.2 |
| dev7-9 val | 3441 | 129 | 0.3104 | -3.2505 | 5783.8 | -16968.8 |

Interpretation:

```text
Most candidate events are harmful under the local utility approximation.
Therefore, an event gate is meaningful in principle.
```

---

## 4. Gate model

Model:

```text
small MLP event classifier
input: 129 event-level aggregated features
output: allow / no-op probability
```

Training output:

```text
outputs/paper_discovery_2026-06-27/reentry_event_gate/models/reentry_event_gate_v21_dev0_6.pt
```

Best epoch under utility-sweep objective:

```text
epoch = 2
training-selected threshold = 0.45
```

Validation utility at threshold 0.45:

```text
total_utility = 2046.0
utility_recall = 0.3537
accuracy = 0.7693
blocked_bad_rate = 0.8061
kept_good_rate = 0.6873
allow_rate = 0.3470
```

Important finding:

```text
The utility-selected threshold 0.45 is too conservative for final AJ_RD.
It blocks many negative-utility events, but it also removes too much re-entry recovery.
```

---

## 5. Dev7-9 final-metric threshold sweep

V1 clean on dev7-9:

```text
AJ_RD_256 = 0.3213
AJ_256    = 79.7517
OA_256    = 94.0018
```

Gate threshold sweep:

| Gate threshold | AJ_RD_256 | AJ_256 | OA_256 | Allow rate | Recovered frames |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 50427 |
| 0.005 | 0.3216 | 79.9035 | 94.0431 | 0.9149 | 48001 |
| 0.010 | 0.3216 | 79.9493 | 94.0191 | 0.8945 | 47095 |
| 0.020 | 0.3192 | 80.0113 | 93.9369 | 0.8605 | 45536 |
| 0.030 | 0.3189 | 80.0762 | 93.8824 | 0.8341 | 44235 |
| 0.040 | 0.3173 | 80.1266 | 93.7980 | 0.8053 | 42842 |
| 0.050 | 0.3146 | 80.1675 | 93.7029 | 0.7809 | 41541 |
| 0.100 | 0.3033 | 80.3087 | 93.3235 | 0.6774 | 36141 |
| 0.150 | 0.2972 | 80.3965 | 93.1440 | 0.6042 | 32358 |
| 0.200 | 0.2832 | 80.4346 | 92.9733 | 0.5400 | 29041 |
| 0.250 | 0.2689 | 80.4424 | 92.8133 | 0.4891 | 26421 |
| 0.300 | 0.2630 | 80.4648 | 92.7081 | 0.4438 | 24124 |
| 0.350 | 0.2569 | 80.4631 | 92.5600 | 0.4034 | 21981 |
| 0.400 | 0.2541 | 80.4618 | 92.4252 | 0.3691 | 20080 |
| 0.450 | 0.2513 | 80.4552 | 92.3485 | 0.3470 | 18923 |

Selected V2.1 threshold for first clean fresh test:

```text
gate_threshold = 0.01
```

Reason:

```text
On dev7-9 it preserves AJ_RD_256 relative to V1 while improving AJ by +0.1976.
```

---

## 6. Fresh20-49 test results, gate threshold 0.01

### Query-weighted main metrics

| Setting | V1 AJ_RD | V2.1 AJ_RD | ΔAJ_RD | V1 AJ | V2.1 AJ | ΔAJ | V1 OA | V2.1 OA | ΔOA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 0.4533 | 0.4522 | -0.0011 | 79.1867 | 79.2618 | +0.0751 | 92.9234 | 92.9192 | -0.0042 |
| translate_L16 | 0.5354 | 0.5352 | -0.0002 | 74.8798 | 74.9784 | +0.0986 | 92.2390 | 92.1954 | -0.0436 |
| occluder_L16 | 0.6671 | 0.6669 | -0.0002 | 77.1320 | 77.2086 | +0.0766 | 92.2273 | 92.2313 | +0.0040 |

### Gate stats on fresh

| Setting | Candidate windows | Gate allow rate | Recovered frames |
|---|---:|---:|---:|
| natural | 31218 | 0.9471 | 446783 |
| translate_L16 | 32499 | 0.9238 | 456683 |
| occluder_L16 | 37288 | 0.9558 | 545810 |

Interpretation:

```text
V2.1 at threshold 0.01 is a light-touch gate.
It blocks roughly 4--8% of candidate windows, not a hard filter.
This is enough to consistently improve AJ while preserving most AJ_RD.
```

---

## 7. Natural threshold 0.005 sanity check

Because natural at threshold 0.01 loses AJ_RD by -0.0011, threshold 0.005 was also tested on natural:

| Gate threshold | AJ_RD_256 | AJ_256 | OA_256 | Allow rate | Recovered frames |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.4524 | 79.2300 | 92.9183 | 0.9623 | 452476 |
| 0.010 | 0.4522 | 79.2618 | 92.9192 | 0.9471 | 446783 |

Interpretation:

```text
0.005 preserves slightly more AJ_RD but gives smaller AJ gain.
0.01 remains the stronger AJ-stability variant.
```

---

## 8. Paired video-level V2.1 vs V1 analysis

Paired video stats were computed with:

```text
scripts/paired_video_stats_reentry_methods.py
```

Outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_event_gate/paired_video_stats/rgb_fresh20_49_natural_gate001.json
docs/reentry_event_gate_paired_video_stats_rgb_fresh20_49_natural_gate001_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_event_gate/paired_video_stats/fresh20_49_translate_L16_gate001.json
docs/reentry_event_gate_paired_video_stats_fresh20_49_translate_L16_gate001_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_event_gate/paired_video_stats/fresh20_49_occluder_L16_gate001.json
docs/reentry_event_gate_paired_video_stats_fresh20_49_occluder_L16_gate001_2026-07-03.md
```

### V2.1 minus V1, AJ_RD_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | -0.000450 | [-0.001750, 0.000793] | 10 / 13 / 7 | 0.6776 |
| translate_L16 | -0.000200 | [-0.000683, 0.000240] | 12 / 13 / 5 | 1.0000 |
| occluder_L16 | -0.000170 | [-0.000723, 0.000357] | 10 / 15 / 5 | 0.4244 |

Interpretation:

```text
AJ_RD change is very small and not statistically meaningful at the video level.
```

### V2.1 minus V1, AJ_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.075054 | [0.047758, 0.103948] | 27 / 1 / 2 | 0.00000022 |
| translate_L16 | +0.098605 | [0.068130, 0.131983] | 29 / 1 / 0 | 0.00000006 |
| occluder_L16 | +0.076583 | [0.053629, 0.100120] | 28 / 1 / 1 | 0.00000011 |

Interpretation:

```text
AJ improvement is highly consistent across videos.
This validates the event gate as a stability-improving variant.
```

### V2.1 minus V1, OA_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | -0.004182 | [-0.036859, 0.029632] | 14 / 14 / 2 | 1.0000 |
| translate_L16 | -0.043588 | [-0.081158, -0.008135] | 11 / 19 / 0 | 0.2005 |
| occluder_L16 | +0.003972 | [-0.023614, 0.032627] | 15 / 14 / 1 | 1.0000 |

Interpretation:

```text
OA is mixed and mostly near-neutral, except translate shows a small drop.
```

---

## 9. Current conclusion

V2.1 should not yet replace V1 as a strictly better method because:

```text
AJ_RD is slightly lower than V1 in fresh query-weighted metrics.
OA is mixed.
```

But V2.1 is a successful stability variant because:

```text
AJ improves consistently and significantly across videos.
AJ_RD loss is tiny and statistically weak.
The module learns a real no-action / risk signal.
```

Best current positioning:

```text
V1 Learned = best AJ_RD-oriented final method.
V2.1 Event-Gated = stability-oriented variant that improves AJ with negligible AJ_RD cost.
```

Paper-safe wording:

```text
Adding an event-level no-action gate consistently improves standard AJ over the frame-level calibrator, while only marginally changing AJ_RD. This confirms that many candidate re-entry events are harmful and that learning when not to intervene is a promising direction.
```

---

## 10. Recommended next step

The next technical step should be V2.2, not more threshold tuning:

```text
V2.2 = event gate + dynamic interval decoder / interval post-processing
```

Reason:

```text
V2.1 confirms the no-action signal exists.
But hard event gating alone trades off AJ_RD and AJ.
The next step should reduce false-visible frames more locally inside allowed events, instead of blocking entire candidate windows.
```

Concrete next options:

```text
1. Keep V2.1 gate threshold low, e.g. 0.01.
2. Add within-window interval selection from V1 probabilities.
3. Only remove low-confidence edge frames, not the whole event.
4. Evaluate whether this keeps V1 AJ_RD while preserving V2.1 AJ gains.
```
