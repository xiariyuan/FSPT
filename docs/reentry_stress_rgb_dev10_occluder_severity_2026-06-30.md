# ReEntry-TAP RGB dev10 Moving-Occluder Severity Curve — 2026-06-30

## Goal

Evaluate whether the ReEntry-TAP conclusion also holds for occlusion-induced re-entry, not only out-of-frame/translation-induced re-entry.

This is development-only on RGB dev0-9. No RGB fresh20-49 validation data is used.

## Stress construction

```text
stress_type = moving_occluder
L = 8, 16, 32
vertical full-height moving bar
speed = 4 px/frame
fill = frame_mean
t0 = 40
```

The occluder modifies only visibility and pixels; GT coordinates remain unchanged. Query points occluded at query time are removed.

## Sanity results

| L | kept queries | re-entry query rate | re-entry events | mean occ length | median occ length | NaN | visible OOB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 11992 | 0.407939 | 7830 | 12.8294 | 8.0 | 0 | 0 |
| 16 | 11612 | 0.412418 | 7523 | 16.6121 | 16.0 | 0 | 0 |
| 32 | 10825 | 0.423649 | 6918 | 24.9498 | 32.0 | 0 | 0 |

All three levels pass sanity. Compared with translate stress, moving-occluder stress creates a denser re-entry setting: about 40%+ of queries contain re-entry.

## Main table

| L | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---:|---|---:|---:|---:|---:|
| 8 | CoTracker3 offline | 0.6417 | 78.5238 | 90.9933 | 88.1378 |
| 8 | CoTracker3 online | 0.6547 | 44.1678 | 57.0715 | 72.4975 |
| 8 | B2-W16-P2 | 0.6875 | 78.0956 | 92.7550 | 88.0913 |
| 16 | CoTracker3 offline | 0.6422 | 77.6880 | 90.6437 | 87.6125 |
| 16 | CoTracker3 online | 0.6477 | 43.6800 | 58.2391 | 72.0218 |
| 16 | B2-W16-P2 | 0.6873 | 77.1682 | 92.2855 | 87.5420 |
| 32 | CoTracker3 offline | 0.6388 | 76.2831 | 90.1738 | 86.8624 |
| 32 | CoTracker3 online | 0.6173 | 42.3135 | 60.5086 | 70.9439 |
| 32 | B2-W16-P2 | 0.6825 | 75.5675 | 91.6044 | 86.7668 |

## Key gains

### B2-W16-P2 vs CoTracker3 offline

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0458 | -0.4282 | +1.7617 |
| 16 | +0.0451 | -0.5198 | +1.6418 |
| 32 | +0.0437 | -0.7156 | +1.4306 |

### Online vs CoTracker3 offline

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0130 | -34.3560 | -33.9218 |
| 16 | +0.0055 | -34.0080 | -32.4046 |
| 32 | -0.0215 | -33.9696 | -29.6652 |

### B2-W16-P2 vs online

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0328 | +33.9278 | +35.6835 |
| 16 | +0.0396 | +33.4882 | +34.0464 |
| 32 | +0.0652 | +33.2540 | +31.0958 |

## Trigger statistics

| L | precision | recall | triggered re-entry | GT re-entry | false-trigger tracks |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.850898 | 0.958913 | 4691 | 4892 | 822 |
| 16 | 0.849594 | 0.940071 | 4502 | 4789 | 797 |
| 32 | 0.843718 | 0.912342 | 4184 | 4586 | 775 |

## Interpretation

Moving-occluder stress confirms that ReEntry-TAP is not merely a translation/out-of-frame artifact.

The online branch does not solve occluder re-entry. It gives small AJ_RD gains for L8/L16 and becomes worse than offline at L32, while collapsing standard AJ by about 34 points:

```text
L8:  online AJ_RD +0.0130, AJ -34.3560
L16: online AJ_RD +0.0055, AJ -34.0080
L32: online AJ_RD -0.0215, AJ -33.9696
```

B2-W16-P2 is consistently better than both branches in AJ_RD while retaining offline-level AJ:

```text
L8:  B2 vs offline AJ_RD +0.0458, AJ -0.4282
L16: B2 vs offline AJ_RD +0.0451, AJ -0.5198
L32: B2 vs offline AJ_RD +0.0437, AJ -0.7156
```

Unlike translate stress, the AJ cost is slightly larger under occluder stress, but it remains below 1 AJ point across all L. This is a strong result: occluder-induced re-entry is harder, yet B2-W16-P2 still preserves most standard tracking while improving re-entry reliability.

## Decision

Moving-occluder stress succeeds. ReEntry-TAP now has two validated stress families:

```text
1. translate_exit_reenter: out-of-frame re-entry
2. moving_occluder: occlusion-induced re-entry
```

This is sufficient to upgrade the paper direction from a single-method heuristic study to a controlled re-entry reliability stress-test plus intervention study.

## Artifacts

```text
scripts/build_reentry_occluder_rgb_dev10.py
scripts/export_cotracker_reentry_stress_cache.py
scripts/eval_reentry_occluder_severity.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_severity_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L8/
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L32/
```
