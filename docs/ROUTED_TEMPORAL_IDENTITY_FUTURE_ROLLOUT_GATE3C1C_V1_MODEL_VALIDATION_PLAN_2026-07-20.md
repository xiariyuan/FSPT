# Route-D temporal identity future rollout Gate 3C1C v1 model validation — 2026-07-20

## Status

```text
COMPLETED_PASS
```

Original model-validation indices 48--63 remain unread at preregistration time.

## Purpose

Gate 3C1B established independent shortlist gains on indices 384--511. Gate
3C1C v0 then showed on the fit-only audit that a teacher-nearest coordinate
inside the frozen shortlist becomes useful only when the four CoTracker memory
levels are deterministically reinstated.

Gate 3C1C v1 performs one unchanged confirmation on the original model-validation
indices 48--63. It does not tune the shortlist score, state action, metric,
bootstrap rule, or threshold.

## Frozen protocol

The following are inherited without modification from Gate 3C1C v0:

- the zero-parameter query-closure identity shortlist;
- native plus eight retained candidates;
- teacher revelation only after shortlist indices are frozen and hashed;
- teacher-nearest selection restricted to that shortlist;
- native, coordinate-only, and coordinate-plus-four-level-memory actions;
- unchanged native visibility/confidence logits;
- continuation over frames 16--23;
- all metric definitions, bootstrap seeds, and pass thresholds.

The candidate cache is generated only after the committed Gate 3C1C v0 exact
replay authorization is verified. It must contain exactly indices 48--63 and
report original-model-validation access while all external/final data remain
unread.

## Frozen gates

The pass gates are byte-for-byte numerically identical to Gate 3C1C v0:

```text
shortlist recall within 12 px                         >= 0.60
shortlist median commit error                         <= 10 px
full-state mean future-error reduction vs native      >= 8 px
full-state equal-video error CI lower                 >= 2 px
full-state threshold-utility gain vs native            >= 0.08
full-state equal-video utility CI lower                >= 0.02
full-state positive-point fraction                     >= 0.65
full-state severe >16 px rate reduction                >= 0.15
memory incremental error reduction vs coordinate-only >= 2 px
memory incremental utility gain vs coordinate-only     >= 0.02
memory better-video fraction                           >= 0.60
native cached/recomputed max absolute error            <= 0.0001 px
exact replay                                           required
```

```text
pass -> AUTHORIZE_GATE3C1D_CAUSAL_TOP1_SELECTOR_PREREGISTRATION
fail -> STOP_QUERY_CLOSURE_STATE_ACTION_BEFORE_TOP1
```

## Claim boundary

Even a positive result remains a teacher-nearest state-action oracle. It would
confirm shortlist reachability and memory reinstatement on original
model-validation, but would not provide a deployable top-1 coordinate decision,
calibration, final-holdout evidence, or external benchmark improvement.

A pass authorizes only preregistration of Gate 3C1D, whose purpose is to replace
the teacher-nearest oracle with a causal top-1 selector under native-safe risk
control.

## Locked data

During preregistration, indices 48--63, calibration, final holdout, DAVIS,
Kinetics, and official Kinetics 1,144 remain unread or not rerun. The committed
cache builder must verify Gate 3C1C v0 authorization before index 48 can be
opened.


## Final result

All unchanged state-action and exact-replay gates passed on indices 48--63.
The formal decision is:

```text
AUTHORIZE_GATE3C1D_CAUSAL_TOP1_SELECTOR_PREREGISTRATION
```

See `docs/ROUTED_TEMPORAL_IDENTITY_FUTURE_ROLLOUT_GATE3C1C_V1_MODEL_VALIDATION_RESULT_2026-07-20.md`.
