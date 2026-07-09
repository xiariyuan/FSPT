# ReEntry-TAP RGB dev10 Translate Severity Curve — 2026-06-30

## Goal

Extend the successful translate-L16 smoke test into a controlled severity curve over out-of-view duration:

```text
translate_exit_reenter L = 8, 16, 32
```

This is still development-only on RGB dev0-9. No RGB fresh20-49 validation data is used.

## Stress sanity

| L | kept queries | re-entry query rate | re-entry events | mean occ length | median occ length | NaN | visible OOB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 12173 | 0.237904 | 5187 | 16.2468 | 11.0 | 0 | 0 |
| 16 | 12054 | 0.238095 | 5139 | 17.2633 | 13.0 | 0 | 0 |
| 32 | 11808 | 0.238144 | 5025 | 19.4038 | 13.0 | 0 | 0 |

All three stress levels pass sanity: query-frame visibility is 100%, no NaNs, no visible out-of-bound points, and about 23.8% of queries contain re-entry.

## Main table

| L | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---:|---|---:|---:|---:|---:|
| 8 | CoTracker3 offline | 0.4709 | 77.0301 | 90.9260 | 86.9612 |
| 8 | CoTracker3 online | 0.5188 | 44.0925 | 56.6368 | 68.6245 |
| 8 | B2-W16-P2 | 0.5513 | 76.9534 | 92.7714 | 87.1168 |
| 16 | CoTracker3 offline | 0.4708 | 75.8656 | 90.7947 | 86.1429 |
| 16 | CoTracker3 online | 0.5194 | 43.9771 | 57.0539 | 67.2819 |
| 16 | B2-W16-P2 | 0.5485 | 75.8344 | 92.6729 | 86.3117 |
| 32 | CoTracker3 offline | 0.4614 | 74.0875 | 90.0318 | 84.9765 |
| 32 | CoTracker3 online | 0.5139 | 43.5656 | 57.7568 | 64.8982 |
| 32 | B2-W16-P2 | 0.5452 | 74.1657 | 92.0251 | 85.1856 |

## Key gains

### B2-W16-P2 vs CoTracker3 offline

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0804 | -0.0767 | +1.8454 |
| 16 | +0.0777 | -0.0312 | +1.8782 |
| 32 | +0.0838 | +0.0782 | +1.9933 |

### Online vs CoTracker3 offline

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0479 | -32.9376 | -34.2892 |
| 16 | +0.0486 | -31.8885 | -33.7408 |
| 32 | +0.0525 | -30.5219 | -32.2750 |

### B2-W16-P2 vs online

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0325 | +32.8609 | +36.1346 |
| 16 | +0.0291 | +31.8573 | +35.6190 |
| 32 | +0.0313 | +30.6001 | +34.2683 |

## Trigger stats

| L | precision | recall | triggered re-entry | GT re-entry | false-trigger tracks |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.683453 | 0.918508 | 2660 | 2896 | 1232 |
| 16 | 0.678363 | 0.929617 | 2668 | 2870 | 1265 |
| 32 | 0.678227 | 0.924964 | 2601 | 2812 | 1234 |

## Interpretation

The translate severity curve strongly supports the ReEntry-TAP direction.

Across all three levels, CoTracker3 online improves AJ_RD over offline but collapses standard AJ by roughly 30--33 points:

```text
L8:  online AJ_RD +0.0479, AJ -32.9376
L16: online AJ_RD +0.0486, AJ -31.8885
L32: online AJ_RD +0.0525, AJ -30.5219
```

B2-W16-P2 avoids this collapse while improving re-entry recovery more than online:

```text
L8:  B2 vs offline AJ_RD +0.0804, AJ -0.0767
L16: B2 vs offline AJ_RD +0.0777, AJ -0.0312
L32: B2 vs offline AJ_RD +0.0838, AJ +0.0782
```

Notably, B2-W16-P2 also outperforms online in AJ_RD while retaining offline-level standard AJ:

```text
L8:  B2 vs online AJ_RD +0.0325, AJ +32.8609
L16: B2 vs online AJ_RD +0.0291, AJ +31.8573
L32: B2 vs online AJ_RD +0.0313, AJ +30.6001
```

This is stronger than the original smoke result because the phenomenon is stable across stress duration.

## Decision

The translate severity curve is successful. ReEntry-TAP should continue.

Next recommended experiment:

```text
Moving-occluder stress L = 8, 16, 32
```

The translate stress captures out-of-frame re-entry. The next step should test occlusion-induced re-entry to show that the reliability problem is not specific to camera translation.

## Artifacts

```text
scripts/build_reentry_stress_rgb_dev10.py
scripts/export_cotracker_reentry_stress_cache.py
scripts/eval_reentry_translate_severity.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_severity_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L8/
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L32/
```
