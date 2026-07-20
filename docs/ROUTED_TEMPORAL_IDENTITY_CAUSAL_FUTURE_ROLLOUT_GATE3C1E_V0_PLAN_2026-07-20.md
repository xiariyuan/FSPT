# Route-D causal top-1 future rollout Gate 3C1E v0 — 2026-07-20

## Status

```text
PREREGISTERED_NOT_RUN
future rollout metrics unread
DAVIS/Kinetics unread for this route
```

## Question

Gate 3C1D v1 established that a frozen deployable selector reduces frame-15
commit error on three raw-record-disjoint Kubric partitions. Gate 3C1E asks the
next necessary question:

> When the actual causal selector—not a teacher—chooses a non-native coordinate,
> does writing that coordinate and deterministically rebuilding all four
> CoTracker memory levels improve visible future frames 16--23?

This gate is required before any claim about complete trajectories or any
comparison with CoTracker3 paper AJ, delta-average, or OA.

## Frozen parent

The only authorized parent is the exact-replay Gate 3C1D v1 model-validation
result:

```text
partition: model_validation_v1, indices 384--511
failure rows: 1,379
frozen bundle SHA256:
878f122bdb647a88e45b4b3ff38f56c9674207a5ea043bcf3bafab9285a72dc2
frozen policy:
  support >= 0.30
  value   >= 1.0 px
  harm    <= 0.20
parent decision:
AUTHORIZE_GATE3C1E_CAUSAL_TOP1_FUTURE_ROLLOUT_PREREGISTRATION
```

No model, feature, shortlist, policy, or threshold may change.

## Causal preflight

Before CoTracker is initialized, the runner must:

1. validate the parent result, replay payload, bundle hash, cache index, sidecars,
   manifest, backbone checkpoint, and all implementation hashes;
2. reconstruct all 1,379 candidate feature rows from causal observed-frame
   tensors only;
3. reload the frozen Gate 3C1D bundle and recompute support, expected distance,
   value, harm, selected shortlist slot, action, and output candidate;
4. require exact per-row equality with the sealed Gate 3C1D model-validation
   replay records;
5. stop before backbone initialization on any drift.

Teacher distance and future tensors are inaccessible until this complete causal
action identity is frozen.

## State actions

Three future trajectories are evaluated for every natural-failure row:

### Native

No coordinate, probability, visibility, or memory write.

### Coordinate only

For rows where the frozen Gate 3C1D policy acts, write only the selected
coordinate at frame 15. All non-action rows remain exact native no-ops.

### Coordinate plus memory

Use the same action mask and coordinate, then deterministically re-extract all
four CoTracker feature/support memory levels from observed frame 15. Probability
and visibility are not written.

Future frames are exactly 16--23. Metrics are computed only on ground-truth
visible future frames, matching the earlier Gate 3C1C mechanism audit.

## Metrics

Report separately over all 1,379 failure rows and over the frozen action rows:

- mean visible-future L2 error;
- average threshold utility over 1, 2, 4, 8, and 16 px;
- severe error rate above 16 px;
- mean error reduction versus native;
- source-video clustered bootstrap confidence intervals;
- positive/nonnegative point fractions;
- harmful future fraction, defined as mean future error exceeding native by more
  than 4 px;
- nonnegative source-video fraction;
- coordinate-plus-memory gain over coordinate-only;
- exact native replay parity and fresh-process replay digests.

## Preregistered gates

```text
selector decisions exactly reproduce Gate 3C1D parent: required
commit mean error reduction:                         >= 2.0 px
all-row full-state future error reduction:           >= 1.0 px
video-cluster future-error CI lower:                  >= 0.25 px
all-row threshold-utility gain:                       >= 0.01
video-cluster utility CI lower:                       >= 0.00
action-row future error reduction:                    >= 3.0 px
action-row positive fraction:                         >= 0.55
all-row severe-16px reduction:                        >= 0.02
all-row harmful-future fraction:                      <= 0.03
action-row harmful-future fraction:                   <= 0.10
nonnegative source-video fraction:                    >= 0.75
memory error gain beyond coordinate-only:             >= 0.5 px
memory error-gain CI lower:                           >= 0.0 px
memory utility gain beyond coordinate-only:           >= 0.005
memory-better source-video fraction:                  >= 0.60
native replay maximum absolute coordinate drift:      <= 1e-4 px
fresh-process exact replay:                           required
```

A primary result never authorizes continuation by itself. Only exact replay that
passes every gate may issue:

```text
AUTHORIZE_GATE3C1F_OFFICIAL_TAPVID_BENCHMARK_PREREGISTRATION
```

Any failed gate issues:

```text
STOP_GATE3C1E_CAUSAL_FUTURE_ROLLOUT
```

No threshold rescue or policy adjustment is allowed after reading results.

## Implementation freeze

```text
runner SHA256:
8686a7811514cb6c75b731618f139bc988949ce28d28e58a94103f80535a751b

config SHA256:
2d2f50089983a3fc31409b667c55b2e6f5479b544a8d27991b1e166cc530a901
```

The configuration additionally pins all imported top-1, cache-verification,
CoTracker interface, state-restoration, and future-metric implementations.

## Claim boundary

A passing Gate 3C1E would establish that the deployable causal action improves
short-horizon future trajectories on the renewed Kubric model-validation
population and that deterministic memory reinstatement contributes beyond a
coordinate overwrite.

It would not yet establish a CoTracker3 paper-table improvement. That requires a
separately preregistered Gate 3C1F run using the same official TAP-Vid dataset,
query mode, raster, checkpoint, visibility outputs, aggregation, and evaluator as
the baseline.

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked during Gate 3C1E.
