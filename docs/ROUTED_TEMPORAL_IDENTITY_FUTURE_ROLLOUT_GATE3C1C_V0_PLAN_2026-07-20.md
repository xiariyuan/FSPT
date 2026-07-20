# Route-D temporal identity future rollout Gate 3C1C v0 — 2026-07-20

## Status

```text
COMPLETED_PASS
```

## Question

Gate 3C1B established that query closure plus DINO identity preserves a better
native-plus-eight frame-15 shortlist. Gate 3C1C asks a narrower state-mechanism
question:

> When a teacher chooses the nearest coordinate only after that causal shortlist
> is frozen, does writing the coordinate and deterministically rebuilding the
> four CoTracker memory levels improve frames 16--23, and does memory rebuilding
> contribute beyond a coordinate-only write?

This gate does not attempt deployable top-1 selection. The teacher-nearest choice
is an oracle used only to test whether the retained shortlist reaches a useful
state-reinstatement basin.

## Data boundary

The primary and replay runs use only the already exposed fit-only internal audit
indices 448--511. They do not read original model-validation indices 48--63,
calibration, final holdout, DAVIS, Kinetics, or official Kinetics 1,144.

The frozen Gate 3C1B audit cache contains 64 videos and 753 natural-failure rows.
The source membership, cache index hash, selector config, backbone checkpoint,
and all parent result hashes are pinned in
`configs/routeD_temporal_identity_future_rollout_gate3c1c_v0.yaml`.

## Frozen selection sequence

For each natural-failure row:

1. read only the causal temporal/static features from frames 0--15;
2. recompute the frozen Gate 3C1B score;
3. retain candidate zero plus eight scored non-native candidates;
4. freeze and hash the selected candidate indices;
5. only then reveal the exact frame-15 teacher coordinate;
6. choose the nearest valid candidate inside the frozen shortlist;
7. never use frames 16--23 to choose a candidate or state action.

## Frozen state actions

All actions start from the same exact native frame-15 snapshot and keep native
visibility/confidence logits unchanged.

### Native

No state field is changed.

### Coordinate only

Write only the teacher-nearest shortlist coordinate at frame 15. Keep all native
track features and support memories unchanged.

### Coordinate plus memory

Write the same coordinate and deterministically re-extract all four CoTracker
track-feature/support levels from the observed frame-15 pyramid using stride 4,
support radius 3, and the frozen scaled-online backbone.

All three branches continue through frames 16--23 with identical code and video
frames.

## Primary hypotheses

Gate 3C1C passes only if:

1. the frozen shortlist still has at least 60% support within 12 px and median
   commit error at most 10 px;
2. coordinate-plus-memory materially improves future error, threshold utility,
   positive-point fraction, and severe-tail rate relative to native;
3. coordinate-plus-memory materially outperforms coordinate-only, establishing
   an independent memory-reinstatement contribution;
4. the separately recomputed native branch matches the cached native future
   error to within 0.0001 px;
5. every scientific digest replays exactly in a fresh process.

## Frozen pass gates

```text
shortlist recall within 12 px                         >= 0.60
shortlist median commit error                         <= 10 px
full-state mean future-error reduction vs native      >= 8 px
full-state equal-video error CI lower                 >= 2 px
full-state threshold-utility gain vs native            >= 0.08
full-state equal-video utility CI lower                >= 0.02
full-state positive-point fraction                     >= 0.65
full-state severe >=16 px rate reduction               >= 0.15
memory incremental error reduction vs coordinate-only >= 2 px
memory incremental utility gain vs coordinate-only     >= 0.02
memory better-video fraction                           >= 0.60
native cached/recomputed max absolute error            <= 0.0001 px
exact replay                                           required
```

```text
pass -> AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION
fail -> STOP_GATE3C1C_STATE_ACTION_ROUTE
```

A pass authorizes only writing a separate protocol for original model-validation
indices 48--63. It does not directly open that partition and does not authorize a
deployable coordinate selector or external evaluation.

## Claim boundary

Possible positive results are teacher-nearest shortlist state-action evidence.
They are not causal top-1 selection performance, end-to-end tracker improvement,
model-validation performance, or external benchmark improvement.


## Final result

All preregistered state-action and integrity gates passed with exact replay.
The formal decision is:

```text
AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION
```

See `docs/ROUTED_TEMPORAL_IDENTITY_FUTURE_ROLLOUT_GATE3C1C_V0_RESULT_2026-07-20.md`.
