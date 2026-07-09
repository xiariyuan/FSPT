# B2-W16 False-Trigger-Cost Guard Development Sweep — 2026-06-29

## Decision

This was a development-only sweep on DAVIS + RGB dev first-10. It did not use RGB heldout-10 for tuning.

The safest unified false-trigger-cost reducer is `w16_persist2`: require the override branch to be visible for 2 consecutive frames before triggering. The effect is modest but consistent: it reduces false-trigger tracks and slightly improves standard AJ, with very small AJ_RD loss.

A more aggressive candidate, `w16_base_invis_t_persist2`, is rejected as a unified method: it looks good on RGB dev but substantially hurts DAVIS re-entry recall and AJ_RD.

## Protocol

```text
Development sweep only on DAVIS + RGB dev first-10. Do not use heldout-10 for tuning.
```

## Main candidates

### RGB dev first-10

| variant | AJ_RD_256 | AJ_256 | AJ_RD gain vs offline | AJ delta vs offline | precision | recall | false-trigger tracks |
|---|---:|---:|---:|---:|---:|---:|---:|
| w16_base | 0.4865 | 79.903 | 0.1102 | -0.1691 | 0.635225 | 0.947303 | 1311 |
| w16_persist2 | 0.4863 | 79.9035 | 0.11 | -0.1686 | 0.640478 | 0.93361 | 1263 |
| w16_persist3 | 0.4863 | 79.9109 | 0.11 | -0.1612 | 0.645367 | 0.921992 | 1221 |
| w16_base_invis_t_persist2 | 0.4901 | 79.9148 | 0.1138 | -0.1573 | 0.633059 | 0.83112 | 1161 |
| w16_max2 | 0.4095 | 80.2214 | 0.0332 | 0.1493 | 0.635225 | 0.947303 | 1311 |
| w16_persist2_max1 | 0.3887 | 80.1987 | 0.0124 | 0.1266 | 0.640478 | 0.93361 | 1263 |

### DAVIS

| variant | AJ_RD_256 | AJ_256 | AJ_RD gain vs offline | AJ delta vs offline | precision | recall | false-trigger tracks |
|---|---:|---:|---:|---:|---:|---:|---:|
| w16_base | 0.6266 | 68.9512 | 0.072 | -1.0998 | 0.529996 | 0.963177 | 1183 |
| w16_persist2 | 0.6251 | 69.0119 | 0.0705 | -1.0391 | 0.548643 | 0.948736 | 1081 |
| w16_persist3 | 0.6209 | 69.0928 | 0.0663 | -0.9582 | 0.55815 | 0.93213 | 1022 |
| w16_base_invis_t_persist2 | 0.6198 | 68.9766 | 0.0652 | -1.0744 | 0.458174 | 0.605054 | 991 |
| w16_max2 | 0.6102 | 69.0599 | 0.0556 | -0.9911 | 0.529996 | 0.963177 | 1183 |
| w16_persist2_max1 | 0.5871 | 69.3375 | 0.0325 | -0.7135 | 0.548643 | 0.948736 | 1081 |

## Interpretation

`w16_persist2` relative to `w16_base`:

```text
RGB dev: AJ_RD -0.0002, AJ +0.0005, false tracks -48
DAVIS:   AJ_RD -0.0015, AJ +0.0607, false tracks -102
```

This is not a dramatic improvement, but it is a clean, prediction-only guard that behaves consistently across both development domains.

Rejected options:

- `w16_base_invis_t_persist2`: best RGB dev AJ_RD, but DAVIS recall collapses from 0.963177 to 0.605054 and AJ_RD drops from 0.6266 to 0.6198.
- `max_windows` variants: recover standard AJ but destroy much of the re-entry gain, especially on DAVIS.
- `persist3`: improves DAVIS AJ more than persist2, but loses more AJ_RD; it is an ablation candidate, not the safest default.

## Frozen candidate for fresh held-out

The next fresh-heldout candidate is:

```text
B2-W16-P2
trigger = base invisible run >= 1 and override visible for 2 consecutive frames
override window = 16 frames
```

Do not test it on already-consumed RGB heldout-10 as a tuning step. The next valid evaluation should use a fresh split such as RGB videos 20-29.

## Artifacts

```text
scripts/sweep_b2_w16_false_cost_guards_dev.py
outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/summary.json
```
