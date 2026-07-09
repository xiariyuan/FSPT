# ReEntry-TAP Frozen L16 Validation on RGB fresh20-49 — 2026-07-01

## Goal

Address the data-scale concern by extending frozen stress validation from RGB fresh20-29 to the full RGB fresh20-49 split.

Protocol:

```text
Design / development split: RGB dev0-9
Frozen validation split:   RGB fresh20-49
Stress families:           translate_exit_reenter_L16 and moving_occluder_L16
Parameters:                frozen from dev protocol, no retuning
```

## Construction

Fresh20-49 was built by combining:

```text
fresh20-29 frozen L16 stress
fresh30-49 frozen L16 stress
```

Fresh30-49 source cache:

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh30_49/offline_source_rgb_stacking_fresh30_49.pt
20 videos
24760 source queries
```

## Stress sanity

| Split | Stress | videos | kept queries | re-entry queries | re-entry events | re-entry query rate | NaN | visible OOB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fresh30-49 | translate L16 | 20 | 24071 | 5708 | 9211 | 0.237132 | 0 | 0 |
| fresh30-49 | occluder L16 | 20 | 23151 | 9457 | 13900 | 0.408492 | 0 | 0 |
| fresh20-49 | translate L16 | 30 | 36176 | 8104 | 12728 | 0.224016 | 0 | 0 |
| fresh20-49 | occluder L16 | 30 | 34808 | 14020 | 19948 | 0.402781 | 0 | 0 |

## Fresh30-49 main results

| Stress | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---|---:|---:|---:|---:|
| translate L16 | offline | 0.4678 | 74.5750 | 90.3310 | 85.0337 |
| translate L16 | online | 0.4903 | 42.9931 | 56.6756 | 67.0123 |
| translate L16 | B2-W16-P2 | 0.5195 | 74.1432 | 91.8746 | 85.0833 |
| occluder L16 | offline | 0.6152 | 76.7416 | 90.1854 | 86.8442 |
| occluder L16 | online | 0.6100 | 42.9494 | 58.0898 | 71.9815 |
| occluder L16 | B2-W16-P2 | 0.6447 | 75.8234 | 91.6865 | 86.6495 |

### Fresh30-49 gains

| Stress | Comparison | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---|---|---:|---:|---:|
| translate L16 | online vs offline | +0.0225 | -31.5819 | -33.6554 |
| translate L16 | B2 vs offline | +0.0517 | -0.4318 | +1.5436 |
| translate L16 | B2 vs online | +0.0292 | +31.1501 | +35.1990 |
| occluder L16 | online vs offline | -0.0052 | -33.7922 | -32.0956 |
| occluder L16 | B2 vs offline | +0.0295 | -0.9182 | +1.5011 |
| occluder L16 | B2 vs online | +0.0347 | +32.8740 | +33.5967 |

## Fresh20-49 main results

| Stress | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---|---:|---:|---:|---:|
| translate L16 | offline | 0.4788 | 75.1940 | 90.7805 | 85.4834 |
| translate L16 | online | 0.4981 | 43.0978 | 56.7847 | 67.1146 |
| translate L16 | B2-W16-P2 | 0.5310 | 74.8457 | 92.2166 | 85.5166 |
| occluder L16 | offline | 0.6311 | 77.6280 | 90.8918 | 87.4234 |
| occluder L16 | online | 0.6146 | 43.1490 | 58.0983 | 72.0255 |
| occluder L16 | B2-W16-P2 | 0.6588 | 76.7991 | 92.1682 | 87.2293 |

### Fresh20-49 gains

| Stress | Comparison | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---|---|---:|---:|---:|
| translate L16 | online vs offline | +0.0193 | -32.0962 | -33.9958 |
| translate L16 | B2 vs offline | +0.0522 | -0.3483 | +1.4361 |
| translate L16 | B2 vs online | +0.0329 | +31.7479 | +35.4319 |
| occluder L16 | online vs offline | -0.0165 | -34.4790 | -32.7935 |
| occluder L16 | B2 vs offline | +0.0277 | -0.8289 | +1.2764 |
| occluder L16 | B2 vs online | +0.0442 | +33.6501 | +34.0699 |

## Fresh20-49 event provenance

| Stress | natural | stress-induced | mixed | stress-induced rate |
|---|---:|---:|---:|---:|
| translate L16 | 7316 | 2284 | 61 | 0.236414 |
| occluder L16 | 5364 | 10072 | 719 | 0.623460 |

## Fresh20-49 stress-induced-only AJ_RD_256

| Stress | offline | online | B2-W16-P2 | B2-offline | B2-online |
|---|---:|---:|---:|---:|---:|
| translate L16 | 0.7618 | 0.7478 | 0.7802 | +0.0184 | +0.0324 |
| occluder L16 | 0.7415 | 0.7118 | 0.7541 | +0.0126 | +0.0423 |

## Fresh20-49 per-video robustness

### B2-W16-P2 vs offline

| Stress | mean ΔAJ_RD | 95% CI | positive videos | sign-test p | mean ΔAJ | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| translate L16 | +0.0451 | [0.0317, 0.0586] | 27/30 | 0.000008 | -0.3483 | [-0.7660, 0.0917] |
| occluder L16 | +0.0263 | [0.0163, 0.0369] | 26/30 | 0.000059 | -0.8289 | [-1.2419, -0.4102] |

## Interpretation

This experiment directly addresses the dataset-size concern.

Before this supplement, frozen validation was limited to RGB fresh20-29:

```text
10 videos
translate L16 + occluder L16
```

After this supplement, frozen validation covers RGB fresh20-49:

```text
30 videos
36176 translate query instances
34808 occluder query instances
12728 translate re-entry events
19948 occluder re-entry events
```

The result pattern remains stable:

```text
translate L16:
B2-W16-P2 improves AJ_RD by +0.0522 with -0.3483 AJ cost.

occluder L16:
B2-W16-P2 improves AJ_RD by +0.0277 with -0.8289 AJ cost.

online branch:
standard AJ collapses by 32--34 points.
```

Stress-induced-only analysis also remains positive:

```text
translate stress-induced-only: +0.0184 AJ_RD over offline
occluder stress-induced-only:  +0.0126 AJ_RD over offline
```

This strengthens the paper's claim that ReEntry-TAP is not merely a small dev-set observation and that B2-W16-P2 generalizes to a larger frozen validation split.

## Paper wording

Recommended wording:

```text
We design the ReEntry-TAP stress protocol on RGB dev0-9 and freeze all construction parameters. We then validate L16 translate and moving-occluder stress on RGB fresh20-49, covering 30 videos, 36K translate queries, 35K occluder queries, and over 32K re-entry events. The same tradeoff persists: the online branch severely damages standard AJ, while B2-W16-P2 consistently improves AJ_RD with limited standard-AJ cost.
```

## Artifacts

```text
scripts/filter_rgb_cache_by_video_range.py
scripts/build_reentry_stress_rgb_from_cache.py
scripts/eval_reentry_frozen_validation.py
scripts/audit_reentry_frozen_validation_statistical_robustness.py
scripts/merge_reentry_frozen_validation_splits.py
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh30_49/
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh30_49/
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/
```
