# CT-Offline Oracle Local Refinement Audit

**Date**: 2026-06-18
**Protocol**: strided+original
**Smoke**: 5 videos, 128 re-entry queries
**Grid stride**: 2px

## Baseline: CoTracker3 Offline Raw Prediction

| Metric | Value |
|---|---:|
| n | 128 |
| Median error | 4.0 px |
| `<4px` | 49.2% |
| `<8px` | 84.4% |

## Oracle Local Refinement Results

| Radius | Oracle median | Oracle `<4px` | `<4px` improv | Oracle `<8px` | `<8px` improv | Overall pass |
|---|---:|---:|---:|---:|---:|---:|
| 8px | 0.8px | 89.1% | +39.8pp | 89.8% | +5.5pp | PASS |
| 16px | 0.8px | 91.4% | +42.2pp | 91.4% | +7.0pp | PASS |
| 32px | 0.8px | 97.7% | +48.4pp | 98.4% | +14.1pp | PASS |

## Long-Occlusion (occ >= 20)

| Long-occ baseline `<4px`: 10.0% |

| Radius | Oracle median | Oracle `<4px` | `<4px` improv | Pass |
|---|---:|---:|---:|---:|
| 8px | 0.7px | 95.0% | +85.0pp | PASS |
| 16px | 0.7px | 95.0% | +85.0pp | PASS |
| 32px | 0.7px | 100.0% | +90.0pp | PASS |

## Decision
**3/3 radii pass both thresholds → headroom exists.**
Proceed to learned verifier / local refiner.
