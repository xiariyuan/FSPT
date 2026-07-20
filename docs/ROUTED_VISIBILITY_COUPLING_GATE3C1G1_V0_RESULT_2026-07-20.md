# Route-D Visibility-Coupling Gate 3C1G1 v0 Result — 2026-07-20

## Decision

```text
COMPLETED_EXACT_REPLAY
STOP_GATE3C1G1_VISIBILITY_MODEL
```

The preregistered 5-outer/4-inner source-video nested OOF evaluation completed on 44 exposed action videos, 89 sealed actions, 801 frame rows, and 66 causal features. Primary and fresh-process replay reproduce the outer scores, outer masks, final selection, and complete scientific payload exactly.

## Nested outer-OOF result

| Metric | Result |
|---|---:|
| GT-visible AUC | 0.545960 |
| GT-visible AP | 0.525180 |
| GT-visible recall | 28.89% |
| Occluded false-positive rate | 26.52% |
| Predicted-visible precision | 52.70% |
| Utility16 precision | 38.74% |

Complete action-video metrics:

| View | AJ | delta_avg | OA |
|---|---:|---:|---:|
| Native | 0.252786 | 0.373921 | 0.830306 |
| Actual Gate 3C1F2 modified | 0.252827 | 0.376646 | 0.834934 |
| Nested-OOF calibrated | 0.252777 | 0.376646 | 0.833518 |

```text
AJ vs actual modified: -0.0051 points
95% CI: [-0.0788, +0.0583] points

AJ vs native: -0.0009 points
OA vs actual modified: -0.1417 points
OA vs native: +0.3212 points
```

## Stability

The five outer folds selected five different candidates:

```text
{
  "hgb_soft_utility_l2_3": 1,
  "log_utility16_c0p1": 1,
  "hgb_visible_l2_10": 1,
  "hgb_utility16_l2_10": 1,
  "hgb_soft_utility_l2_10": 1
}
```

Only two outer-fold inner selections were feasible. The full-data selection chose `hgb_soft_utility_l2_10` at threshold `0.225`, but that inner selection was itself infeasible under the frozen recall/FPR/OA constraints. This bundle is retained only as failed evidence and is not authorized for confirmation.

## Interpretation

Replacing all action-frame visibility with a learned score is rejected. The richer 66-D evidence did not generalize at row level, and broad replacement sacrificed recall and OA without producing AJ gain. The next admissible structural hypothesis is narrower: preserve all visibility decisions except native-occluded to modified-visible activation transitions, and learn only whether each new activation should be accepted. This directly targets the 268 recovered visible frames versus 186 newly introduced occluded false positives identified by the Gate 3C1F2 diagnostic.

No new raw population, DAVIS, Kinetics, final holdout, or official Kinetics 1,144 was read.
