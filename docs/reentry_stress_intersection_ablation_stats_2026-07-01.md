# ReEntry-TAP Strictness Supplements: Intersection Query, L16 Ablation, and Stress Statistics — 2026-07-01

## Goal

This document records three low-cost but high-value supplements for the ReEntry-TAP stress results:

```text
1. intersection-query severity audit
2. L16 stress ablation
3. per-video statistical robustness
```

These experiments do not rerun trackers. They reuse existing offline / online / B2 caches.

---

## 1. Intersection-query severity audit

### Motivation

The original severity curves use the same source videos and frozen construction rules, but retained query sets differ slightly across L because some query frames are removed when they become invalid under stress.

To rule out query-set variation as the source of the trend, we take the intersection of source queries that are retained under all three severities L8/L16/L32, then re-evaluate all methods.

### Translate intersection query set

```text
common source queries = 11808
```

| L | #queries | re-entry rate | B2 vs offline ΔAJ_RD | B2 vs offline ΔAJ | online vs offline ΔAJ_RD | online vs offline ΔAJ |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 11808 | 0.237890 | +0.0780 | -0.1127 | +0.0464 | -33.5104 |
| 16 | 11808 | 0.237890 | +0.0766 | -0.0481 | +0.0481 | -32.3509 |
| 32 | 11808 | 0.238144 | +0.0838 | +0.0782 | +0.0525 | -30.5219 |

### Moving-occluder intersection query set

```text
common source queries = 10825
```

| L | #queries | re-entry rate | B2 vs offline ΔAJ_RD | B2 vs offline ΔAJ | online vs offline ΔAJ_RD | online vs offline ΔAJ |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 10825 | 0.424942 | +0.0443 | -0.4532 | +0.0105 | -35.2690 |
| 16 | 10825 | 0.424942 | +0.0446 | -0.5532 | +0.0047 | -34.7157 |
| 32 | 10825 | 0.423649 | +0.0437 | -0.7156 | -0.0215 | -33.9696 |

### Interpretation

The main ReEntry-TAP severity conclusions remain unchanged under strict intersection-query evaluation.

```text
translate: B2-W16-P2 still improves AJ_RD by about +0.077 to +0.084 with near-zero AJ cost.
occluder:  B2-W16-P2 still improves AJ_RD by about +0.044 with <1 AJ cost.
```

This addresses the concern that L8/L16/L32 trends are caused by different retained query sets.

---

## 2. L16 stress ablation

### Motivation

Show that the method is not merely using the online branch globally. The key mechanism is local windowed override.

Variants:

```text
offline:         base tracker only
online_global:   online / override branch globally
b2_fullpost_p1:  trigger once, then override to the end
b2_w8_p2:        local W8 override, persistent-visible trigger P2
b2_w16_p1:       local W16 override, no P2 persistence
b2_w16_p2:       main method
b2_w32_p2:       local W32 override, persistent-visible trigger P2
```

### Translate L16 ablation

| Method | AJ_RD_256 | AJ_256 | ΔAJ_RD vs offline | ΔAJ vs offline |
|---|---:|---:|---:|---:|
| offline | 0.4708 | 75.8656 | — | — |
| online_global | 0.5194 | 43.9771 | +0.0486 | -31.8885 |
| b2_fullpost_p1 | 0.5311 | 75.2650 | +0.0603 | -0.6006 |
| b2_w8_p2 | 0.5498 | 75.8454 | +0.0790 | -0.0202 |
| b2_w16_p1 | 0.5487 | 75.8343 | +0.0779 | -0.0313 |
| b2_w16_p2 | 0.5485 | 75.8344 | +0.0777 | -0.0312 |
| b2_w32_p2 | 0.5448 | 75.7849 | +0.0740 | -0.0807 |

### Moving-occluder L16 ablation

| Method | AJ_RD_256 | AJ_256 | ΔAJ_RD vs offline | ΔAJ vs offline |
|---|---:|---:|---:|---:|
| offline | 0.6422 | 77.6880 | — | — |
| online_global | 0.6477 | 43.6800 | +0.0055 | -34.0080 |
| b2_fullpost_p1 | 0.6560 | 76.0911 | +0.0138 | -1.5969 |
| b2_w8_p2 | 0.6897 | 77.2316 | +0.0475 | -0.4564 |
| b2_w16_p1 | 0.6877 | 77.1701 | +0.0455 | -0.5179 |
| b2_w16_p2 | 0.6873 | 77.1682 | +0.0451 | -0.5198 |
| b2_w32_p2 | 0.6822 | 77.0227 | +0.0400 | -0.6653 |

### Interpretation

The ablation confirms the mechanism:

```text
1. online_global collapses standard AJ by 32–34 points.
2. full-post override avoids global collapse but still costs more AJ and gives weaker AJ_RD than windowed variants.
3. local windows W8/W16/W32 are all effective.
4. W8 and W16 are the strongest operating points under L16 stress.
5. P2 has small effect under stress, but remains useful as the conservative main setting because it reduced false triggers in natural DAVIS/RGB audits.
```

For the paper, B2-W16-P2 remains a safe main method because it was selected before these stress supplements and is consistent with natural-data audits. The stress ablation can also mention that W8-P2 is a strong stress-only variant, but should not replace the frozen main method unless retuning is allowed.

---

## 3. Per-video statistical robustness

### Translate: B2-W16-P2 vs offline

| L | mean ΔAJ_RD | 95% CI | positive videos | sign-test p | mean ΔAJ | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | +0.0705 | [0.0488, 0.0931] | 10/10 | 0.001953 | -0.0767 | [-0.5745, 0.4227] |
| 16 | +0.0682 | [0.0499, 0.0870] | 10/10 | 0.001953 | -0.0312 | [-0.4929, 0.4504] |
| 32 | +0.0752 | [0.0533, 0.0995] | 10/10 | 0.001953 | +0.0782 | [-0.3913, 0.5595] |

### Moving-occluder: B2-W16-P2 vs offline

| L | mean ΔAJ_RD | 95% CI | positive videos | sign-test p | mean ΔAJ | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | +0.0427 | [0.0300, 0.0599] | 10/10 | 0.001953 | -0.4282 | [-0.9170, -0.0012] |
| 16 | +0.0427 | [0.0313, 0.0597] | 10/10 | 0.001953 | -0.5198 | [-0.9738, -0.1154] |
| 32 | +0.0415 | [0.0276, 0.0607] | 10/10 | 0.001953 | -0.7156 | [-1.1753, -0.2895] |

### Interpretation

The AJ_RD improvement is video-consistent across all six stress settings:

```text
B2-W16-P2 improves AJ_RD over offline on 10/10 videos for every translate and occluder severity.
```

AJ cost behaves differently by stress family:

```text
translate: AJ cost is statistically small / near zero, with CI crossing zero.
occluder:  AJ cost is consistently negative but remains below 1 AJ point in mean.
```

This supports the main claim: B2-W16-P2 reliably improves re-entry recovery, while standard tracking cost is minimal for translate and controlled under occluder.

## Artifacts

```text
scripts/eval_reentry_intersection_severity.py
scripts/eval_reentry_stress_ablation_l16.py
scripts/audit_reentry_stress_statistical_robustness.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/intersection_query_severity_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/stress_ablation_L16_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/stress_statistical_robustness_summary.json
```
