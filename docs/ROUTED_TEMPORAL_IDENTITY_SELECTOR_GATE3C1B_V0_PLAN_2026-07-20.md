# Route-D query-closure identity selector Gate 3C1B v0 — 2026-07-20

## Status

```text
PREREGISTERED_NOT_RUN_ON_CHECKPOINT_SELECTION
```

Indices 384--511 remain unread at preregistration time.

## Frozen hypothesis

A correct frame-15 candidate must satisfy two conditions using only frames
0--15:

1. its reverse trajectory returns to the known query coordinate;
2. its sampled DINOv3 identity remains consistent with the query descriptor.

The training-only mechanism audit falsified a broader path-energy hypothesis.
Visibility, confidence, descriptor self-continuity, velocity, acceleration, and
jerk frequently reward a smooth but identity-wrong trajectory. They are excluded
from the primary score rather than assigned intuitive monotonic weights.

## Primary selector

For every failure row, normalize each signal across the 129 valid candidates and
compute

```text
score = -2 * z(reverse cycle error at the query frame)
        + z(query identity cosine at the query frame)
        + z(mean query identity cosine over frames 0--15)
        + z(minimum query identity cosine over frames 0--15)
```

Candidate zero is mandatory. The selector retains candidate zero plus the eight
highest-scoring non-native candidates. Stable ties use lower frozen M1 rank.
Candidate coordinates are never modified.

The selector has zero trainable parameters, no optimizer, no loss, no epoch, and
no checkpoint. A learned residual is forbidden in v0. Consequently indices
384--447 are an independent confirmation partition, not a hyperparameter or
checkpoint-selection partition in the usual learned-model sense.

## Training-only design evidence

The frozen rule was chosen using only the completed gradient-train cache on
indices 64--383. This evidence is architectural design evidence, not a formal
Gate 3C1B result.

| Rule | 12 px retained support | Recall |
|---|---:|---:|
| native + first eight M1 candidates | 1,725 / 3,325 | 51.8797% |
| cycle only | 1,821 / 3,325 | 54.7669% |
| identity only | 1,838 / 3,325 | 55.2782% |
| frozen closure + identity primary | 1,970 / 3,325 | 59.2481% |
| complete native + 128 oracle | 3,185 / 3,325 | 95.7895% |

The primary rule improves pooled support by 7.3684 percentage points, improves
8-pixel recall from 31.6090% to 38.1955%, and reduces median retained-candidate
distance from 11.4892 px to 9.8567 px. The equal-video bootstrap mean support
gain is 8.5321 points with 95% CI [6.0415, 11.0977]. A fresh full-cache replay
reproduced every selected-index, score, metric, and video-record digest exactly.

These values remain training-only and cannot be reported as independent selector
performance.

## Frozen evaluation order

1. Commit and push this protocol, implementation, cache builders, and tests.
2. Build only the checkpoint-selection cache on indices 384--447.
3. Run primary evaluation and a fresh exact replay.
4. If any checkpoint gate fails, stop without reading 448--511.
5. Only an exact replay pass may authorize the one-shot fit-only audit cache on
   indices 448--511.
6. Run the audit once and replay it exactly.
7. A Gate 3C1B audit pass authorizes only preregistration of future state rollout.

The audit cache builder itself verifies the checkpoint replay authorization file;
a manual command cannot bypass step 4.

## Checkpoint-selection gate: indices 384--447

All conditions are required after exact replay:

- pooled 12-pixel support gain over static M1 is at least +4 points;
- absolute 12-pixel retained support is at least 56%;
- equal-video bootstrap CI lower bound for support gain is positive;
- median minimum-distance improvement is at least 0.75 px;
- 8-pixel recall gain is positive;
- at least 60% of videos are nonnegative;
- all deterministic digests replay exactly.

```text
pass -> AUTHORIZE_GATE3C1B_FIT_ONLY_AUDIT_448_511
fail -> STOP_GATE3C1B_WITHOUT_AUDIT
```

## One-shot audit gate: indices 448--511

All conditions are required after exact replay:

- pooled 12-pixel support gain over static M1 is at least +5 points;
- absolute 12-pixel retained support is at least 57%;
- equal-video bootstrap CI lower bound for support gain is positive;
- median minimum-distance improvement is at least 1.0 px;
- 8-pixel recall gain is positive;
- at least 60% of videos are nonnegative;
- all deterministic digests replay exactly.

```text
pass -> AUTHORIZE_GATE3C1C_FUTURE_ROLLOUT_PREREGISTRATION
fail -> STOP_QUERY_CLOSURE_IDENTITY_SELECTOR
```

## Claim boundary

Gate 3C1B measures whether a causal selector preserves a useful frame-15
candidate in a nine-candidate shortlist. It does not establish that selecting or
committing any candidate improves frames 16--23. Coordinate selection, future
rollout, state writeback, original model validation, calibration, final holdout,
DAVIS, Kinetics, and official Kinetics 1,144 remain separate and locked.
