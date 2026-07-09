# V25 Official-Safe Threshold Sweep — DAVIS Strided + Original — 2026-07-04

## 1. Goal

After the official protocol audit showed that the default ReEntry V1/V22Q improves AJ_RD but hurts official-style AJ/OA, this sweep tests whether a stricter V1 threshold can preserve official leaderboard-style metrics while retaining some AJ_RD gain.

Protocol:

```text
Dataset: TAPVid-DAVIS
Query mode: strided
Metric resolution: original
Base cache: CoTracker3 offline strided_original
Override cache: CoTracker3 online strided_original
Model: reentry_viscalibrator_v1_dev0_6_clean.pt
Thresholds: 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99
```

Artifacts:

```text
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/v25_threshold_sweep/
```

---

## 2. Offline reference

Official-style original-resolution offline baseline:

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 0.3870 |

---

## 3. Threshold sweep results

| V1 threshold | AJ | OA | δ_avg | AJ_RD | ΔAJ | ΔOA | ΔAJ_RD | recovered frames |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 50.7569 | 91.2704 | 63.5892 | 0.4141 | -0.7816 | -0.8839 | +0.0271 | 39,809 |
| 0.30 | 50.8000 | 91.2821 | 63.5892 | 0.4136 | -0.7384 | -0.8722 | +0.0266 | 39,508 |
| 0.40 | 50.8522 | 91.3187 | 63.5892 | 0.4131 | -0.6863 | -0.8357 | +0.0261 | 38,993 |
| 0.50 | 50.9103 | 91.4152 | 63.5892 | 0.4118 | -0.6282 | -0.7391 | +0.0248 | 37,759 |
| 0.60 | 51.0521 | 91.6501 | 63.5892 | 0.4092 | -0.4864 | -0.5043 | +0.0222 | 34,459 |
| 0.70 | 51.3570 | 91.9982 | 63.5892 | 0.3988 | -0.1815 | -0.1562 | +0.0118 | 26,932 |
| 0.80 | 51.5648 | 92.2106 | 63.5892 | 0.3900 | +0.0263 | +0.0563 | +0.0030 | 15,024 |
| 0.90 | 51.5630 | 92.1914 | 63.5892 | 0.3875 | +0.0246 | +0.0370 | +0.0005 | 4,140 |
| 0.95 | 51.5420 | 92.1607 | 63.5892 | 0.3870 | +0.0035 | +0.0063 | +0.0000 | 773 |
| 0.99 | 51.5385 | 92.1543 | 63.5892 | 0.3870 | +0.0000 | +0.0000 | +0.0000 | 0 |

---

## 4. Best official-safe point

The best trade-off point is:

```text
V25-safe threshold = 0.80
```

Why:

```text
AJ:    +0.0263 pp vs offline
OA:    +0.0563 pp vs offline
AJ_RD: +0.0030 absolute vs offline
```

This is the first threshold where official-style AJ/OA are non-negative while AJ_RD remains slightly positive.

However, the gain is very small and should not be overstated.

---

## 5. Paired-video check for threshold 0.80

Overall paired-video differences, V1 threshold 0.80 minus offline:

```text
AJ:
  mean = +0.0263 pp
  95% bootstrap CI = [-0.0517, +0.1253]
  wins/losses/ties = 12/13/5

OA:
  mean = +0.0563 pp
  95% bootstrap CI = [-0.1060, +0.2473]
  wins/losses/ties = 14/10/6

δ_avg:
  mean = +0.0000 pp
  wins/losses/ties = 0/0/30
```

Long-occlusion subset paired differences, threshold 0.80 minus offline:

```text
AJ:
  n = 13 videos with long-occ subset
  mean = -0.6908 pp
  95% bootstrap CI = [-2.1613, +0.1716]
  wins/losses/ties = 2/4/7

OA:
  n = 13 videos with long-occ subset
  mean = -1.2025 pp
  95% bootstrap CI = [-3.2166, +0.0236]
  wins/losses/ties = 2/4/7
```

AJ_RD at threshold 0.80:

```text
Offline true AJ_RD = 0.3870
V25-safe true AJ_RD = 0.3900
Δ true AJ_RD = +0.0030

Offline true AJ_RD_256 = 0.5546
V25-safe true AJ_RD_256 = 0.5601
Δ true AJ_RD_256 = +0.0055
```

---

## 6. Interpretation

The sweep shows that a stricter ReEntry threshold can make the method nearly official-safe.

But it also shows the trade-off sharply:

```text
Low thresholds:
  Large AJ_RD gain, but official AJ/OA drop.

High thresholds:
  Official AJ/OA preserved, but AJ_RD gain nearly disappears.

Threshold 0.80:
  Small positive AJ/OA and small positive AJ_RD, but not statistically strong.
```

Therefore, threshold 0.80 should be described as:

```text
an official-style safe operating point
```

not as:

```text
a strong leaderboard improvement
```

---

## 7. Recommended paper usage

Safe wording:

```text
Under DAVIS strided+original evaluation, increasing the ReEntry confidence threshold to 0.80 yields a conservative operating point that preserves official-style AJ/OA while retaining a small AJ_RD gain. This supports the interpretation that ReEntry can be made leaderboard-safe, but the leaderboard-style gain is marginal.
```

Avoid:

```text
V25 significantly improves the official leaderboard.
V25 is a strong official metric improvement.
V25 solves the full leaderboard objective.
```

---

## 8. Next technical step

The current threshold-only sweep finds only a tiny official-safe gain. The next technical direction should be:

```text
Official-metric-aware selective ReEntry
```

Instead of only thresholding V1 probability, the method should estimate whether a proposed visibility flip helps or hurts official AJ/OA.

Candidate V26 direction:

```text
Train or derive a frame-level keep/drop rule using standard-Jaccard utility, not only re-entry utility.
```

Practical next experiment:

```text
Build a DAVIS strided_original frame-keep dataset with labels based on official AJ/OA utility.
Train a conservative classifier that fires only when both:
  1. re-entry utility is positive, and
  2. standard-Jaccard harm is unlikely.
```
