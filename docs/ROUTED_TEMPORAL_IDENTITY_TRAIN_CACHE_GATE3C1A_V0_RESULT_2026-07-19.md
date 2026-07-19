# Route-D temporal identity train-cache Gate 3C1A v0 result — 2026-07-19

## Decision

```text
AUTHORIZE_GATE3C1B_SELECTOR_PREREGISTRATION
```

The gradient-train-only cache passed every completion and isolation check.

## Qualified population

| item | value |
| --- | ---: |
| source indices | 64--383 |
| required sidecars | 320 |
| validated sidecars | 320 |
| videos with failures | 308 |
| videos with valid empty tensors | 12 |
| frozen natural-failure rows | 3,325 |

The expected, completed, and index-row memberships are each exactly the
contiguous range 64--383. Every sidecar path, file hash, tensor hash, feature
shape, finite-value check, and pre-teacher candidate-coordinate hash validated.

## Training-only support diagnosis

| frozen candidate rule | supported within 12 px | recall |
| --- | ---: | ---: |
| native only | 54 / 3,325 | 1.6241% |
| native + static M1 top 8 | 1,725 / 3,325 | 51.8797% |
| native + complete M1 top 128 oracle | 3,185 / 3,325 | 95.7895% |

The complete bank contains a valid 12-pixel hypothesis for 1,460 more rows
than the static top-eight rule, an absolute support gap of 43.9098 percentage
points. Only 140/3,325 rows lack a 12-pixel hypothesis anywhere in the bank.
This establishes a large ranking target on the gradient-training population.
It does not establish that causal features can recover that target.

## Isolation

Checkpoint-selection indices 384--447, fit-only audit indices 448--511,
original model-validation indices 48--63, calibration, final holdout, DAVIS,
Kinetics, and the frozen official Kinetics 1,144 protocol were not read.

## Claim boundary

These numbers are teacher-selected support headroom from gradient-training
data. They are not selector performance, coordinate refinement, state
restoration, natural-union utility, model validation, or external improvement.
The next authorized action is only to preregister a causal candidate-retention
mechanism before opening checkpoint-selection data.
