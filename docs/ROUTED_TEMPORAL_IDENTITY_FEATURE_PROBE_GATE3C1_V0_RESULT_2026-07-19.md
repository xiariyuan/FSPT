# Route-D temporal identity feature probe Gate 3C1 v0 result — 2026-07-19

## Status

```text
COMPLETED_EXACT_REPLAY
```

## Claim boundary

This is a single-gradient-train-sample feature-shape and resource result. It is
not selector accuracy, tracking improvement, model-validation evidence, or an
external result.

## Frozen sample

- expanded source index: 64;
- video name: `2238`;
- eligible natural-failure rows: 10;
- frozen probe row: point index 9;
- original query frame: 0.

The selected native trajectory has a future mean error of 113.3631 px. Its
native frame-15 commit error is 86.7079 px. The frozen M1 native-plus-128 bank
contains a candidate 10.6436 px from the frame-15 teacher, so this row has
12-pixel support. These teacher distances were read only after the candidate
coordinate hash and causal feature tensors were frozen.

This one-row oracle observation must not be aggregated or claimed as a model
gain.

## Measured shapes

| Tensor | Shape |
|---|---|
| CoTracker frame-15 pyramid | [128,96,128], [128,48,64], [128,24,32], [128,12,16] |
| reverse candidate tracklets | [129,16,2] |
| DINOv3 feature video | [16,384,14,14] |
| sampled descriptor sequence | [129,16,384] |
| temporal identity features | [129,16,9] |
| static/list features | [129,14] |
| validity mask | [129] |

All 129 candidates are valid. Candidate zero is the mandatory native proposal.
The final float32 temporal and static features plus the boolean mask occupy
81,657 bytes per failure row.

## Runtime and memory

| Stage | Time | Peak allocated CUDA memory |
|---|---:|---:|
| native tracking and M1 candidate bank | 1.9903 s | 1564.83 MiB |
| 129 reverse candidate tracklets | 0.1872 s | 1698.44 MiB |
| DINOv3 and feature assembly | 10.2088 s | 266.91 MiB |

The measured feature representation and one-row candidate microbatch fit well
inside the available 24 GiB GPU. Full-cache execution should reuse one DINO
encoding per video and microbatch reverse tracklets by failure row; it should
not materialize raw 384-dimensional descriptors in the training cache.

## Determinism

A fresh-process replay produced exact equality for sample identity, candidate
bank diagnostics, every shape, storage size, integrity flags, and every feature
tensor hash.

- candidate coordinate hash before teacher:
  `9dfafa1910c8145ccf20312763d9c079508d81015089669507859f75c5aafe8a`
- temporal feature hash:
  `ead419c4134840768d2ec60420f466734c48197a85bf8b14b3ecbb9926803de1`
- static feature hash:
  `0444940137629ee4b85ccda2c557dc3c740bcdc2bd2e64ab9334151181077ee9`
- validity-mask hash:
  `642a48616eb151ff1126f67fcdcef96f5abb6e798243d9b33e3dea60b4e10d78`
- combined tensor hash digest:
  `9460db07c74ef5b7c12180d1a05da9a5a24b8bc28459099b22833faf78a67a74`

## Decision

```text
FEATURE_SHAPE_AND_MEMORY_FEASIBLE
PREREGISTER_GATE3C1_CACHE_AND_LISTWISE_SELECTOR
```

The next protocol must keep indices 448--511 unread until every feature
definition, model parameter, loss weight, checkpoint rule, top-eight rule, and
pass threshold is frozen. Candidate support and final state rollout remain
separate gates.
