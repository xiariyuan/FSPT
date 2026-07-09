# ReEntry-TAP Frozen Mini Validation on RGB fresh20-29 — 2026-07-01

## Goal

Validate the frozen ReEntry-TAP stress protocol on a fresh split that was not used to design the stress settings.

Frozen settings:

```text
translate_exit_reenter_L16:
  amplitude = 0.4W
  t0 = 40
  ramp = 8
  fill = frame_mean

moving_occluder_L16:
  vertical full-height bar
  speed = 4 px/frame
  t0 = 40
  fill = frame_mean
```

Dataset:

```text
RGB fresh20-29
10 videos
```

## Stress sanity

| Stress | kept queries | re-entry query rate | re-entry events | mean occ length | median occ length | NaN | visible OOB |
|---|---:|---:|---:|---:|---:|---:|---:|
| translate L16 | 12105 | 0.197935 | 3517 | 29.2568 | 19.0 | 0 | 0 |
| occluder L16 | 11657 | 0.391439 | 6048 | 22.9200 | 16.0 | 0 | 0 |

Note: translate L16 is slightly below the original dev sanity target of 20% re-entry-query rate (0.197935), but this is a frozen validation split. We do not retune the stress parameters. There are no NaNs, no visible out-of-bound points, and query-frame visibility is 100%.

## Main results

| Stress | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---|---:|---:|---:|---:|
| translate L16 | offline | 0.5049 | 76.4320 | 91.6794 | 86.3829 |
| translate L16 | online | 0.5166 | 43.3072 | 57.0030 | 67.3190 |
| translate L16 | B2-W16-P2 | 0.5583 | 76.2507 | 92.9006 | 86.3832 |
| occluder L16 | offline | 0.6640 | 79.4009 | 92.3044 | 88.5818 |
| occluder L16 | online | 0.6241 | 43.5482 | 58.1152 | 72.1135 |
| occluder L16 | B2-W16-P2 | 0.6879 | 78.7506 | 93.1315 | 88.3889 |

## Key gains

### B2-W16-P2 vs offline

| Stress | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---|---:|---:|---:|
| translate L16 | +0.0534 | -0.1813 | +1.2212 |
| occluder L16 | +0.0239 | -0.6503 | +0.8271 |

### Online vs offline

| Stress | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---|---:|---:|---:|
| translate L16 | +0.0117 | -33.1248 | -34.6764 |
| occluder L16 | -0.0399 | -35.8527 | -34.1892 |

### B2-W16-P2 vs online

| Stress | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---|---:|---:|---:|
| translate L16 | +0.0417 | +32.9435 | +35.8976 |
| occluder L16 | +0.0638 | +35.2024 | +35.0163 |

## Trigger statistics

| Stress | precision | recall | triggered re-entry | GT re-entry | false-trigger tracks |
|---|---:|---:|---:|---:|---:|
| translate L16 | 0.650822 | 0.908598 | 2177 | 2396 | 1168 |
| occluder L16 | 0.876132 | 0.911462 | 4159 | 4563 | 588 |

## Event provenance on fresh20-29

| Stress | natural | stress-induced | mixed | stress-induced rate |
|---|---:|---:|---:|---:|
| translate L16 | 1987 | 772 | 14 | 0.278399 |
| occluder L16 | 1466 | 3422 | 208 | 0.671507 |

Occluder L16 on fresh20-29 is even cleaner than the dev occluder stress: 67.15% of eligible events are directly stress-induced.

## Stress-induced-only AJ_RD_256

| Stress | offline | online | B2-W16-P2 | B2-offline | B2-online |
|---|---:|---:|---:|---:|---:|
| translate L16 | 0.8253 | 0.7798 | 0.8334 | +0.0081 | +0.0536 |
| occluder L16 | 0.7794 | 0.7140 | 0.7790 | -0.0004 | +0.0650 |

## Interpretation

Frozen validation supports the main ReEntry-TAP story.

On the fresh split, B2-W16-P2 improves full AJ_RD while preserving standard AJ:

```text
translate L16: AJ_RD +0.0534, AJ -0.1813
occluder L16:  AJ_RD +0.0239, AJ -0.6503
```

Online remains a poor tradeoff:

```text
translate L16: AJ_RD +0.0117, AJ -33.1248
occluder L16:  AJ_RD -0.0399, AJ -35.8527
```

Stress-induced-only analysis is more nuanced:

```text
translate stress-induced: B2 is slightly above offline and clearly above online.
occluder stress-induced: B2 is essentially tied with offline and clearly above online.
```

This means the fresh full AJ_RD gain is not only from stress-induced events. It also comes from natural re-entry events inside the frozen stress videos. This is acceptable and should be stated precisely:

```text
Frozen validation confirms the overall re-entry/standard-tracking tradeoff and B2-W16-P2's favorable operating point. Stress-induced-only gains remain positive for translate, while occluder stress-induced events show B2 preserving offline-level AJ_RD and strongly outperforming the online branch.
```

## Decision

Frozen mini validation succeeds as a generalization check. It should be included in the paper as evidence that the ReEntry-TAP protocol and B2-W16-P2 behavior are not restricted to RGB dev0-9.

Recommended wording:

```text
After freezing the stress construction parameters on RGB dev0-9, we validate translate-L16 and occluder-L16 on RGB fresh20-29. The same pattern holds: B2-W16-P2 improves full AJ_RD over offline while preserving standard AJ, whereas the online branch severely damages standard tracking.
```

## Artifacts

```text
scripts/build_reentry_stress_rgb_from_cache.py
scripts/eval_reentry_fresh20_29_frozen_validation.py
scripts/audit_reentry_stress_event_provenance.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/fresh20_29_frozen_validation_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/event_provenance_combined_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/translate_L16/
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/occluder_L16/
```
