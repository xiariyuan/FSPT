# ReEntry-TAP RGB dev10 Combined Stress Summary — 2026-06-30

## Scope

This document summarizes controlled re-entry stress evaluation on RGB dev0-9 for two stress families:

```text
1. translate_exit_reenter: out-of-frame re-entry
2. moving_occluder: occlusion-induced re-entry
```

Both are development-only. No RGB fresh20-49 validation data is used.

## Translate stress: B2-W16-P2 vs offline

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0804 | -0.0767 | +1.8454 |
| 16 | +0.0777 | -0.0312 | +1.8782 |
| 32 | +0.0838 | +0.0782 | +1.9933 |

## Moving-occluder stress: B2-W16-P2 vs offline

| L | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
|---:|---:|---:|---:|
| 8 | +0.0458 | -0.4282 | +1.7617 |
| 16 | +0.0451 | -0.5198 | +1.6418 |
| 32 | +0.0437 | -0.7156 | +1.4306 |

## Translate stress: online vs offline

| L | ΔAJ_RD_256 | ΔAJ_256 |
|---:|---:|---:|
| 8 | +0.0479 | -32.9376 |
| 16 | +0.0486 | -31.8885 |
| 32 | +0.0525 | -30.5219 |

## Moving-occluder stress: online vs offline

| L | ΔAJ_RD_256 | ΔAJ_256 |
|---:|---:|---:|
| 8 | +0.0130 | -34.3560 |
| 16 | +0.0055 | -34.0080 |
| 32 | -0.0215 | -33.9696 |

## Main conclusion

ReEntry-TAP now has a strong controlled-stress evidence chain.

Translate stress shows that online/re-entry-strong tracking improves AJ_RD but catastrophically damages standard tracking, while B2-W16-P2 improves AJ_RD more than online and preserves offline-level AJ.

Moving-occluder stress shows a harder but consistent pattern: online does not reliably improve AJ_RD under occlusion-induced re-entry and still collapses standard AJ, while B2-W16-P2 consistently improves AJ_RD by about +0.044 to +0.046 with less than 1 AJ point cost.

## Paper-level interpretation

These two stress families support the upgraded paper direction:

```text
ReEntry-TAP: a controlled re-entry reliability stress test and local intervention framework for Tracking Any Point.
```

Safe claim:

```text
B2-W16-P2 is a lightweight inference-time intervention that improves re-entry reliability under both natural and controlled stress-induced re-entry while preserving most standard tracking quality.
```

Do not claim:

```text
B2-W16-P2 solves all long-term TAP failures.
B2-W16-P2 is a new SOTA point tracker.
ReEntry-TAP replaces natural benchmarks.
```

## Recommended next step

Do not add a third stress family yet. The next most valuable step is event-provenance and stress-induced-only AJ_RD:

```text
1. classify re-entry events as natural / stress-induced / mixed;
2. report AJ_RD on stress-induced events only;
3. use this to prevent the criticism that gains come from natural re-entry already present in RGB.
```
