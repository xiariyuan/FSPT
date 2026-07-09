# ReEntry-VisCalibrator Paper Tables V2 — 2026-07-03

## Purpose

This document reorganizes the current results into paper-ready tables after the learned ReEntry-VisCalibrator clean protocol.

Final naming used here:

| Paper name | Meaning | Learned? | Coordinates | Visibility |
|---|---|---|---|---|
| Base | offline/base tracker | No | base | base |
| Global Override | global override tracker | No | override | override |
| Global Vis-Only | base coordinates + global override visibility | No | base | override globally |
| Rule W8P2 | original local replacement rule | No | base | override in local W8P2 window |
| Ours-Det | deterministic positive-only local recovery, W16P2 | No | base | base OR override-visible in W16P2 window |
| Ours-Learned | ReEntry-VisCalibrator V1 clean | Yes | base | learned visibility recovery |

Clean learned protocol:

```text
train: dev0-6
threshold selection: dev7-9
selected threshold: 0.10
test: fresh20-49 natural / translate_L16 / occluder_L16
```

---

## Table 1. Main query-weighted results

These are the primary paper numbers. Use AJ_RD_256 as the main re-entry metric.

| Setting | Method | Learned? | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ |
|---|---|---:|---:|---:|---:|
| RGB fresh20-49 natural | Base | No | 0.3816 | 79.5944 | 91.4636 |
| RGB fresh20-49 natural | Global Override | No | 0.4121 | 44.6933 | 55.7585 |
| RGB fresh20-49 natural | Global Vis-Only | No | 0.4282 | 45.7904 | 55.7585 |
| RGB fresh20-49 natural | Rule W8P2 | No | 0.4510 | 79.1110 | 92.8853 |
| RGB fresh20-49 natural | Ours-Det | No | 0.4524 | 79.1098 | 92.8919 |
| RGB fresh20-49 natural | Ours-Learned | Yes | **0.4533** | **79.1867** | **92.9234** |
| fresh20-49 translate_L16 | Base | No | 0.4788 | 75.1940 | 90.7805 |
| fresh20-49 translate_L16 | Global Override | No | 0.4981 | 43.0978 | 56.7847 |
| fresh20-49 translate_L16 | Global Vis-Only | No | 0.5133 | 44.0643 | 56.7847 |
| fresh20-49 translate_L16 | Rule W8P2 | No | 0.5336 | 74.7705 | 92.2245 |
| fresh20-49 translate_L16 | Ours-Det | No | 0.5348 | 74.7679 | 92.2329 |
| fresh20-49 translate_L16 | Ours-Learned | Yes | **0.5354** | **74.8798** | **92.2390** |
| fresh20-49 occluder_L16 | Base | No | 0.6311 | 77.6280 | 90.8918 |
| fresh20-49 occluder_L16 | Global Override | No | 0.6146 | 43.1490 | 58.0983 |
| fresh20-49 occluder_L16 | Global Vis-Only | No | 0.6469 | 44.7880 | 58.0983 |
| fresh20-49 occluder_L16 | Rule W8P2 | No | 0.6659 | 77.0436 | 92.1748 |
| fresh20-49 occluder_L16 | Ours-Det | No | 0.6667 | 77.0443 | 92.1858 |
| fresh20-49 occluder_L16 | Ours-Learned | Yes | **0.6671** | **77.1320** | **92.2273** |

Key readout:

```text
Ours-Learned is the best overall version in the main table.
It improves AJ_RD over Base by +0.0717, +0.0566, and +0.0360 on natural, translate_L16, and occluder_L16 respectively.
It also improves over the original Rule W8P2 and deterministic Ours-Det in all three final settings.
```

---

## Table 2. Ours-Learned gains over Base

| Setting | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---|---:|---:|---:|
| RGB fresh20-49 natural | 0.0717 | -0.4077 | 1.4598 |
| fresh20-49 translate_L16 | 0.0566 | -0.3142 | 1.4585 |
| fresh20-49 occluder_L16 | 0.0360 | -0.4960 | 1.3355 |

Paper-safe wording:

```text
Compared with the offline/base tracker, ReEntry-VisCalibrator improves AJ_RD_256 by +0.0717, +0.0566, and +0.0360 on the natural, translate_L16, and occluder_L16 settings, while keeping the standard-AJ drop below 0.5 points.
```

---

## Table 3. Learned calibrator gains over rule variants

| Setting | Baseline | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---|---|---:|---:|---:|
| RGB fresh20-49 natural | Rule W8P2 | 0.0023 | 0.0757 | 0.0381 |
| RGB fresh20-49 natural | Ours-Det | 0.0009 | 0.0769 | 0.0315 |
| fresh20-49 translate_L16 | Rule W8P2 | 0.0018 | 0.1093 | 0.0145 |
| fresh20-49 translate_L16 | Ours-Det | 0.0006 | 0.1119 | 0.0061 |
| fresh20-49 occluder_L16 | Rule W8P2 | 0.0012 | 0.0884 | 0.0525 |
| fresh20-49 occluder_L16 | Ours-Det | 0.0004 | 0.0877 | 0.0415 |

Interpretation:

```text
The learned calibrator consistently improves over the original W8P2 rule on all three metrics.
Against the stronger deterministic positive-only variant, the AJ_RD gain is small but positive, while AJ is consistently better.
```

---

## Table 4. Video-level paired statistics

These are video-weighted paired statistics over 30 videos per setting. They should be used as stability evidence, not as replacements for query-weighted main metrics.

### Ours-Learned vs Base, AJ_RD_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties |
|---|---:|---:|---:|
| RGB fresh20-49 natural | 0.0709 | [0.0530, 0.0905] | 29 / 0 / 1 |
| fresh20-49 translate_L16 | 0.0489 | [0.0356, 0.0621] | 29 / 1 / 0 |
| fresh20-49 occluder_L16 | 0.0349 | [0.0253, 0.0454] | 28 / 2 / 0 |

### Ours-Learned vs Ours-Det, AJ_256

| Setting | Mean diff | 95% bootstrap CI | Wins / Losses / Ties |
|---|---:|---:|---:|
| RGB fresh20-49 natural | 0.0769 | [0.0418, 0.1210] | 26 / 2 / 2 |
| fresh20-49 translate_L16 | 0.1119 | [0.0727, 0.1566] | 28 / 1 / 1 |
| fresh20-49 occluder_L16 | 0.0878 | [0.0474, 0.1372] | 27 / 2 / 1 |

Paper-safe wording:

```text
At the video level, the learned method improves AJ_RD over Base on 29/30, 29/30, and 28/30 videos for natural, translate_L16, and occluder_L16. Compared with the deterministic variant, its clearest advantage is standard AJ: it improves AJ on 26/30, 28/30, and 27/30 videos, respectively.
```

---

## Recommended paper narrative

Use this hierarchy:

```text
Core principle: coordinate-preserving local visibility recovery.
Ours-Det: deterministic non-learned version that proves the principle is strong.
Ours-Learned: learned calibrator that further improves the recovery-stability tradeoff.
```

Strong claim supported:

```text
ReEntry-VisCalibrator is the best overall version in our evaluated CoTracker-family settings, improving AJ_RD, AJ, and OA over both the original rule and the deterministic positive-only variant in the query-weighted main table.
```

Careful nuance:

```text
The deterministic variant already captures most of the AJ_RD recovery. The learned calibrator mainly improves the stability side of the tradeoff: it matches or slightly improves AJ_RD while consistently improving standard AJ across videos.
```
