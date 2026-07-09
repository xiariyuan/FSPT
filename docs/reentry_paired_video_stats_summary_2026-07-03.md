# ReEntry Paired Video Statistics Summary — 2026-07-03

## Purpose

This document checks whether the learned `ReEntry-VisCalibrator V1 clean` improvement is stable across videos or only driven by a small number of videos.

Protocol:

```text
learned model trained on dev0-6
threshold selected on dev7-9: 0.10
frozen evaluation on fresh20-49
paired unit: video
n = 30 videos per setting
```

Compared methods:

```text
base: offline/base
rule_w8: original ReEntry-VisGuard-W8P2
det_w16: deterministic positive-only W16P2
learned: ReEntry-VisCalibrator V1 clean
```

Scripts / outputs:

```text
scripts/paired_video_stats_reentry_methods.py
outputs/paper_discovery_2026-06-27/reentry_paired_video_stats/rgb_fresh20_49_natural.json
docs/reentry_paired_video_stats_rgb_fresh20_49_natural_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_paired_video_stats/fresh20_49_translate_L16.json
docs/reentry_paired_video_stats_fresh20_49_translate_L16_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_paired_video_stats/fresh20_49_occluder_L16.json
docs/reentry_paired_video_stats_fresh20_49_occluder_L16_2026-07-03.md
```

---

## 1. Video-weighted means

These are per-video means, so they differ slightly from query-weighted main-table numbers.

### Natural

| Method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 |
|---|---:|---:|---:|
| base | 0.332953 | 79.5944 | 91.4636 |
| rule_w8 | 0.400610 | 79.1110 | 92.8853 |
| det_w16 | 0.401743 | 79.1098 | 92.8919 |
| learned | 0.403863 | 79.1867 | 92.9234 |

### Translate_L16

| Method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 |
|---|---:|---:|---:|
| base | 0.508040 | 75.1940 | 90.7805 |
| rule_w8 | 0.555583 | 74.7705 | 92.2245 |
| det_w16 | 0.556617 | 74.7679 | 92.2329 |
| learned | 0.556967 | 74.8798 | 92.2390 |

### Occluder_L16

| Method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 |
|---|---:|---:|---:|
| base | 0.641727 | 77.6280 | 90.8918 |
| rule_w8 | 0.675440 | 77.0436 | 92.1748 |
| det_w16 | 0.676133 | 77.0443 | 92.1858 |
| learned | 0.676603 | 77.1320 | 92.2273 |

---

## 2. Learned vs base

### AJ_RD_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.070910 | [0.053046, 0.090450] | 29 / 0 / 1 | 0.000000 |
| translate_L16 | +0.048927 | [0.035637, 0.062147] | 29 / 1 / 0 | 0.00000006 |
| occluder_L16 | +0.034877 | [0.025290, 0.045427] | 28 / 2 / 0 | 0.00000087 |

Interpretation:

```text
The learned final method is consistently and strongly better than base on AJ_RD across videos.
```

### AJ_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | -0.407698 | [-0.781588, 0.003384] | 5 / 25 / 0 | 0.000325 |
| translate_L16 | -0.314227 | [-0.654693, 0.070744] | 5 / 25 / 0 | 0.000325 |
| occluder_L16 | -0.495979 | [-0.864729, -0.073027] | 4 / 26 / 0 | 0.000059 |

Interpretation:

```text
Relative to base, learned recovery trades a small amount of standard AJ for large AJ_RD gains.
```

---

## 3. Learned vs original rule W8P2

### AJ_RD_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.003253 | [0.000470, 0.007533] | 20 / 7 / 3 | 0.019157 |
| translate_L16 | +0.001383 | [0.000077, 0.002720] | 19 / 10 / 1 | 0.136046 |
| occluder_L16 | +0.001163 | [0.000383, 0.002087] | 22 / 6 / 2 | 0.003719 |

### AJ_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.075735 | [0.040759, 0.119485] | 24 / 6 / 0 | 0.001431 |
| translate_L16 | +0.109288 | [0.070446, 0.153344] | 28 / 2 / 0 | 0.000001 |
| occluder_L16 | +0.088395 | [0.049147, 0.136179] | 29 / 1 / 0 | 0.00000006 |

Interpretation:

```text
Compared with the original W8P2 rule, learned V1 is consistently better on standard AJ and generally better on AJ_RD.
```

This supports the claim:

```text
The learned calibrator improves the recovery-vs-stability tradeoff over the original rule.
```

---

## 4. Learned vs deterministic positive-only W16P2

This is the strongest simple rule control.

### AJ_RD_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.002120 | [-0.000667, 0.006697] | 12 / 10 / 8 | 0.831812 |
| translate_L16 | +0.000350 | [-0.000687, 0.001390] | 14 / 11 / 5 | 0.690038 |
| occluder_L16 | +0.000470 | [-0.000127, 0.001143] | 14 / 9 / 7 | 0.404873 |

### AJ_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.076907 | [0.041826, 0.121013] | 26 / 2 / 2 | 0.000003 |
| translate_L16 | +0.111854 | [0.072697, 0.156610] | 28 / 1 / 1 | 0.00000011 |
| occluder_L16 | +0.087764 | [0.047420, 0.137238] | 27 / 2 / 1 | 0.00000162 |

### OA_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| natural | +0.031521 | [0.001323, 0.069896] | 15 / 13 / 2 | 0.850554 |
| translate_L16 | +0.006135 | [-0.050207, 0.062163] | 17 / 12 / 1 | 0.458258 |
| occluder_L16 | +0.041502 | [0.007191, 0.084568] | 19 / 10 / 1 | 0.136046 |

Interpretation:

```text
Against the strongest deterministic W16 rule, learned V1 has only a small AJ_RD advantage and the video-level AJ_RD advantage is not statistically strong. However, learned V1 has a clear and consistent AJ advantage across all three settings.
```

This is the most important nuance for the paper.

---

## 5. Paper-safe conclusion

Strong claim supported:

```text
ReEntry-VisCalibrator V1 is the best overall version: it preserves the large AJ_RD gain over base and improves standard AJ over both the original W8P2 rule and the stronger deterministic W16 variant.
```

Careful claim needed:

```text
Its additional AJ_RD gain over deterministic W16P2 is small and not yet statistically strong at the video level. The learned module's clearest contribution is improving the recovery-vs-standard-AJ tradeoff.
```

Best wording:

```text
The deterministic positive-only variant already captures most of the AJ_RD recovery. The learned calibrator further improves the stability side of the tradeoff: it matches or slightly improves AJ_RD while consistently improving standard AJ across videos.
```

Do not write:

```text
The learned module dramatically improves AJ_RD over all rule variants.
```

---

## 6. Recommended main-table strategy

Use query-weighted main-table metrics for primary reporting, but add this paired-video analysis as a stability check.

Main table should include:

```text
Base
Original local rule W8P2
Deterministic positive-only W16P2
ReEntry-VisCalibrator V1 clean
```

In the text:

```text
Compared to deterministic W16P2, the learned calibrator yields modest AJ_RD changes but consistently higher standard AJ: +0.077, +0.112, and +0.088 video-mean AJ points on natural, translate_L16, and occluder_L16, respectively.
```

This is honest and strong enough.
