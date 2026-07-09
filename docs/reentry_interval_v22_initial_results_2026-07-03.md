# ReEntry-VisCalibrator V2.2 Interval Post-Processing — Initial Results — 2026-07-03

## 1. Goal

V2.1 Event Gate showed that learning when not to intervene is useful, but whole-event gating is too coarse. V2.2 tests a lighter idea:

```text
Do not block all recovery in a candidate window.
Instead, keep the V1 recovery core and trim short / low-confidence recovery fragments.
```

The goal is:

```text
keep V1 AJ_RD
retain or improve V2.1 AJ gains
reduce harmful recovery frames inside candidate windows
```

---

## 2. New code

```text
scripts/eval_reentry_viscalibrator_v22_interval.py
```

The script:

```text
1. loads base/offline and override/online caches;
2. loads the V1 frame-level ReEntry-VisCalibrator;
3. optionally loads the V2.1 event gate;
4. computes V1 recovery probabilities per candidate window;
5. builds the raw V1 recovery mask;
6. splits the mask into contiguous temporal segments;
7. trims low-confidence edges and short segments;
8. optionally blocks the lowest-confidence events using the event-gate score;
9. writes a new cache and evaluates AJ_RD / AJ / OA.
```

Compile check:

```bash
python -m py_compile scripts/eval_reentry_viscalibrator_v22_interval.py
```

passed.

---

## 3. Dev7-9 V2.2 sweep

Baseline V1 on dev7-9:

```text
AJ_RD_256 = 0.3213
AJ_256    = 79.7517
OA_256    = 94.0018
```

### First interval-only sweep

| Config | Description | AJ_RD | AJ | OA | Final/raw recovery rate | Removed/raw recovery rate |
|---|---|---:|---:|---:|---:|---:|
| sanity | min_len=1, edge=0.10 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 0.0000 |
| B | min_len=2, edge=0.10 | 0.3208 | 79.7549 | 93.9964 | 0.9964 | 0.0036 |
| C | min_len=2, edge=0.12 | 0.3208 | 79.7763 | 93.9992 | 0.9882 | 0.0118 |
| D | min_len=2, edge=0.15 | 0.3203 | 79.8144 | 93.9832 | 0.9728 | 0.0272 |
| E | min_len=3, edge=0.12 | 0.3204 | 79.7830 | 93.9949 | 0.9839 | 0.0161 |
| F | adaptive gate trimming | 0.3208 | 79.7864 | 94.0164 | 0.9873 | 0.0127 |

Interpretation:

```text
Simple interval trimming works but is too weak.
It removes only 1--3% of recovered frames in moderate settings.
It slightly improves AJ but cannot match the V2.1 gate AJ gain.
```

### Stronger interval-only sweep

| Config | Description | AJ_RD | AJ | OA | Final/raw recovery rate | Removed/raw recovery rate |
|---|---|---:|---:|---:|---:|---:|
| G | min_len=2, edge=0.20 | 0.3203 | 79.8985 | 93.9073 | 0.9375 | 0.0625 |
| H | min_len=2, edge=0.25 | 0.3198 | 79.9707 | 93.8146 | 0.9026 | 0.0974 |
| I | min_len=2, edge=0.30 | 0.3185 | 80.0344 | 93.7287 | 0.8676 | 0.1324 |
| J | min_len=3, edge=0.20 | 0.3199 | 79.9057 | 93.8967 | 0.9326 | 0.0674 |
| K | min_len=3, edge=0.25 | 0.3195 | 79.9754 | 93.8082 | 0.8984 | 0.1016 |
| L | min_len=3, edge=0.30 | 0.3181 | 80.0376 | 93.7192 | 0.8633 | 0.1367 |
| M | stronger adaptive | 0.3204 | 79.8777 | 93.9900 | 0.9533 | 0.0467 |

Interpretation:

```text
Higher edge thresholds improve AJ but steadily reduce AJ_RD and OA.
This is the same recovery-stability tradeoff as event gating, just at frame/segment level.
```

### Combined event block + interval trimming

Because event-gate selection was more precise than edge trimming alone, V2.2 was extended with:

```text
--gate-block-threshold
```

This blocks only the lowest-confidence events, then applies interval trimming to the remaining events.

| Config | Description | AJ_RD | AJ | OA | Gate-blocked windows | Final/raw recovery rate | Removed/raw recovery rate |
|---|---|---:|---:|---:|---:|---:|---:|
| N | block=0.01, min_len=1, edge=0.10 | 0.3216 | 79.9493 | 94.0191 | 363 | 0.9339 | 0.0661 |
| O | block=0.01, min_len=2, edge=0.10 | 0.3211 | 79.9510 | 94.0134 | 363 | 0.9312 | 0.0688 |
| P | block=0.01, min_len=2, edge=0.12 | 0.3212 | 79.9606 | 94.0123 | 363 | 0.9273 | 0.0727 |
| Q | block=0.01, min_len=2, edge=0.15 | 0.3212 | 79.9763 | 94.0033 | 363 | 0.9210 | 0.0790 |
| R | block=0.005, min_len=2, edge=0.12 | 0.3211 | 79.9171 | 94.0333 | 293 | 0.9442 | 0.0558 |
| S | block=0.005, min_len=2, edge=0.15 | 0.3211 | 79.9412 | 94.0095 | 293 | 0.9341 | 0.0659 |

Selected first V2.2 test configuration:

```text
V22Q = block=0.01, min_len=2, edge_threshold=0.15, core_threshold=0.10
```

Reason:

```text
It gives the best AJ among configurations that keep dev7-9 AJ_RD close to V1.
```

---

## 4. Fresh20-49 query-weighted results

### V1 vs V2.1 vs V22Q

| Setting | V1 AJ_RD | V2.1 AJ_RD | V22Q AJ_RD | V1 AJ | V2.1 AJ | V22Q AJ | V1 OA | V2.1 OA | V22Q OA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 0.4533 | 0.4522 | 0.4520 | 79.1867 | 79.2618 | 79.2775 | 92.9234 | 92.9192 | 92.9233 |
| translate_L16 | 0.5354 | 0.5352 | 0.5350 | 74.8798 | 74.9784 | 74.9919 | 92.2390 | 92.1954 | 92.1943 |
| occluder_L16 | 0.6671 | 0.6669 | 0.6669 | 77.1320 | 77.2086 | 77.2218 | 92.2273 | 92.2313 | 92.2326 |

### V22Q minus V1

| Setting | ΔAJ_RD | ΔAJ | ΔOA |
|---|---:|---:|---:|
| natural | -0.0013 | +0.0908 | -0.0001 |
| translate_L16 | -0.0004 | +0.1121 | -0.0448 |
| occluder_L16 | -0.0002 | +0.0898 | +0.0053 |

### V22Q minus V2.1

| Setting | ΔAJ_RD | ΔAJ | ΔOA |
|---|---:|---:|---:|
| natural | -0.0002 | +0.0157 | +0.0041 |
| translate_L16 | -0.0002 | +0.0135 | -0.0011 |
| occluder_L16 | +0.0000 | +0.0132 | +0.0013 |

Interpretation:

```text
V22Q improves AJ beyond V2.1 in all three settings.
The improvement is small but consistent.
AJ_RD remains slightly below V1 and usually slightly below V2.1, except occluder is effectively tied.
```

---

## 5. Paired video-level statistics

Paired outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_interval_v22/paired_video_stats/rgb_fresh20_49_natural_v22Q.json
docs/reentry_interval_v22_paired_video_stats_rgb_fresh20_49_natural_v22Q_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_interval_v22/paired_video_stats/fresh20_49_translate_L16_v22Q.json
docs/reentry_interval_v22_paired_video_stats_fresh20_49_translate_L16_v22Q_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_interval_v22/paired_video_stats/fresh20_49_occluder_L16_v22Q.json
docs/reentry_interval_v22_paired_video_stats_fresh20_49_occluder_L16_v22Q_2026-07-03.md
```

### V22Q minus V1, AJ_RD_256

| Setting | Mean diff | 95% CI | Wins / Losses / Ties | p |
|---|---:|---:|---:|---:|
| natural | -0.000683 | [-0.002137, 0.000607] | 10 / 17 / 3 | 0.2478 |
| translate_L16 | -0.000413 | [-0.001090, 0.000193] | 12 / 14 / 4 | 0.8450 |
| occluder_L16 | -0.000113 | [-0.000723, 0.000520] | 11 / 15 / 4 | 0.5572 |

Interpretation:

```text
AJ_RD loss versus V1 remains tiny and statistically weak at the video level.
```

### V22Q minus V1, AJ_256

| Setting | Mean diff | 95% CI | Wins / Losses / Ties | p |
|---|---:|---:|---:|---:|
| natural | +0.090832 | [0.060083, 0.123911] | 28 / 2 / 0 | 0.00000087 |
| translate_L16 | +0.112165 | [0.079220, 0.147983] | 29 / 1 / 0 | 0.00000006 |
| occluder_L16 | +0.089732 | [0.064421, 0.115571] | 29 / 1 / 0 | 0.00000006 |

Interpretation:

```text
V22Q improves AJ over V1 even more consistently than V2.1.
```

### V22Q minus V2.1, AJ_256

| Setting | Mean diff | 95% CI | Wins / Losses / Ties | p |
|---|---:|---:|---:|---:|
| natural | +0.015779 | [0.009213, 0.024269] | 29 / 1 / 0 | 0.00000006 |
| translate_L16 | +0.013561 | [0.008595, 0.019382] | 28 / 2 / 0 | 0.00000087 |
| occluder_L16 | +0.013148 | [0.009072, 0.017850] | 29 / 1 / 0 | 0.00000006 |

Interpretation:

```text
The interval stage gives a small but highly stable additional AJ gain on top of V2.1.
```

### V22Q minus V2.1, AJ_RD_256

| Setting | Mean diff | 95% CI | Wins / Losses / Ties | p |
|---|---:|---:|---:|---:|
| natural | -0.000233 | [-0.000777, 0.000400] | 4 / 17 / 9 | 0.0072 |
| translate_L16 | -0.000213 | [-0.000490, 0.000060] | 8 / 12 / 10 | 0.5034 |
| occluder_L16 | +0.000057 | [-0.000127, 0.000287] | 9 / 12 / 9 | 0.6636 |

Interpretation:

```text
The extra interval trimming has a small AJ_RD cost in natural, but this cost is tiny in magnitude.
In translate and occluder, AJ_RD is effectively tied with V2.1.
```

---

## 6. Current conclusion

V22Q is not a strict replacement for V1 because:

```text
V1 still has the best AJ_RD in query-weighted results.
```

V22Q is stronger than V2.1 as a stability variant because:

```text
It consistently improves AJ beyond V2.1.
It keeps OA roughly tied or slightly better than V2.1.
Its extra AJ_RD cost is tiny.
```

Best current positioning:

```text
V1 Learned = best AJ_RD-oriented final method.
V2.1 Event-Gated = first stability-oriented extension.
V22Q Interval = stronger stability-oriented extension with additional AJ gain.
```

Paper-safe wording:

```text
A light event gate combined with interval trimming consistently improves standard AJ over the frame-level calibrator and the event-gated variant, while changing AJ_RD only marginally. This suggests that harmful recovery is concentrated in a small subset of events and low-confidence segment edges.
```

---

## 7. Technical interpretation

The V2.2 experiment tells us:

```text
1. Whole-event risk is useful.
2. Segment-edge risk is also useful.
3. Deterministic trimming improves AJ, but it still cannot fully avoid AJ_RD loss.
4. The remaining challenge is to distinguish harmful edge frames from true first re-entry frames.
```

This means deterministic interval trimming has likely reached its useful limit.

---

## 8. Recommended next step

Do not keep hand-tuning deterministic thresholds.

The next meaningful direction is:

```text
V2.3 = learned frame-keep/risk decoder on top of V1 proposals.
```

Key idea:

```text
V1 predicts which frames might be recovered.
V2.3 predicts which of those proposed recovery frames should be kept.
```

Training target should be local utility of keeping each V1-proposed frame:

```text
positive: V1 proposes recovery and keeping it improves local AJ / AJ_RD
negative: V1 proposes recovery but keeping it creates false-visible or coordinate-risk error
```

This is more targeted than V1 because the model only learns on proposed recovery frames, not all candidate frames.

Expected benefit:

```text
keep V1 AJ_RD better than V2.2
retain most of V22Q AJ gain
reduce false-visible edges without deleting true first re-entry frames
```
