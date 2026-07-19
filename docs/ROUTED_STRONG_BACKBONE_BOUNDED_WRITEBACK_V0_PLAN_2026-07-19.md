# Route-D strong-backbone bounded coordinate writeback v0 plan — 2026-07-19

## 1. Decision being preregistered

P0j identifies variant C as the strongest safety-feasible strong-backbone
configuration:

```text
frozen formal P0g CMCP core
+ trained rank-32 LMRA
+ trained local comparator
+ frozen CoTracker3 native trajectory
```

C reaches `+0.8861` AJ on the complete 16-video Kubric model-validation split,
with paired CI `[+0.6489,+1.0694]` and harmful non-native rate `0.9396%`.
P0k asks one narrower question:

> Can a single fixed, bounded coordinate-only writeback convert variant C's
> current-frame corrections into additional future tracking value without
> violating its existing safety boundary?

This is the only state-writeback experiment authorized by P0j. It is not a
parameter sweep and does not reopen model architecture selection.

## 2. Frozen system

The following are byte-frozen:

- CoTracker3 scaled-online checkpoint and true-streaming execution;
- P0j variant C checkpoint;
- LMRA, CMCP, comparator, token normalization, candidate extraction and NMS;
- native candidate 0, query mode, 256x256 raster and first-visible queries;
- fit/model-validation identities;
- output-only variant C as the comparison baseline;
- all external and final-holdout locks.

No component receives gradients in P0k.

## 3. Fixed intervention

For a frame where variant C selects a non-native candidate, define

```text
delta = selected_coordinate - current_native_coordinate
write_delta = delta * min(1, 8px / ||delta||_2)
write_coordinate = current_native_coordinate + write_delta
```

The `8px` bound is frozen from the existing `1/2/4/8/16px` utility grid. It is
not selected from model-validation results.

The hook may write only:

```text
CoTracker3 online_coords_predicted
```

It may not write:

```text
online_vis_predicted
online_conf_predicted
online_track_feat
online_track_support
LMRA / CMCP / comparator state
```

The current-frame output is copied before state mutation. State mutation is
allowed only for frames in the second half of the current 16-frame online
window, because only that overlap is consumed as initialization by the next
8-frame streaming step.

## 4. Causal candidate state

Variant C is evaluated online over all points jointly. It must not be run one
query at a time, because CoTracker3 spatial attention couples the active query
set.

The proposal branch retains its original causal memories:

1. immutable query-frame support;
2. previous native feature;
3. fixed-alpha `0.9` EMA native feature.

At a write frame, previous-native and EMA memories use the pre-writeback native
feature. The write hook changes only the CoTracker coordinate state inherited by
the next window. Future proposal memories may change only through the genuinely
changed future native trajectory.

The point-independent frozen feature maps are loaded from the already verified
float16 CMCP cache. This keeps the online output-only path numerically aligned
with the formal C evaluation while the live CoTracker state supplies the native
trajectory.

## 5. Interface gate before model validation

Use fit source index `0` only. Run two independent output-only processes and one
bounded-write process.

Output-only must match formal cached C exactly for:

```text
native coordinates, visibility and confidence
candidate coordinates and validity masks
selected candidate indices
selected coordinates
dynamic decision summaries
```

The bounded-write process must additionally prove:

```text
candidate 0 remains exact native
all writes occur only in the declared overlap
all write norms are <= 8px
visibility/confidence state is unchanged by the hook
no output is changed before the first eligible write
future native state diverges only after an applied write
independent full-video replay is exact
```

A parity failure is an implementation failure. Model validation remains locked
until every interface gate passes.

## 6. Formal model-validation comparison

Run the complete 16-video model-validation partition once with the fixed hook.
Compare closed loop against output-only variant C, not against native alone.

All gates are required:

```text
closed-loop AJ gain over C >= +0.10 point
paired-video closed-loop-minus-C AJ CI lower bound > 0
closed-loop delta-average gain > 0
16px severe-error rate not worse than C
harmful non-native rate <= 1%
exact candidate-0/native parity on every video
exact seed-17 independent replay
```

The `+0.10` threshold matches the preregistered P0j component non-redundancy
magnitude. A statistically positive but smaller result is recorded as
insufficient for the paper mainline.

## 7. Stop rules

If P0k fails, do not sweep:

```text
write bound
write region
confirmation length
candidate threshold
visibility/confidence writeback
LMRA rank
CMCP checkpoint
NMS / top-K / EMA
```

A failure closes coordinate-only state writeback for the current paper. Variant C
remains the strong-backbone output-level model.

## 8. Locked data

```text
fit source 0: interface audit only
model validation 48--63: one fixed formal comparison
calibration: unread
final holdout: unread
TAP-Vid-DAVIS: unread
official Kinetics 1,144: unread and not rerun
```

## 9. Completed interface result — 2026-07-19

The fit-source-0 interface audit fails before bounded writeback or model-validation
access. Formal P0j-C is reconstructed exactly and the final live native state
matches the frozen cache exactly, but the provisional commit-time stream does
not match formal C:

```text
candidate-coordinate max component difference: 56.3684px
selected-coordinate max component difference:  36.5723px
selected-index mismatches:                       2 / 1536 rows
selected-coordinate mismatches:                502 / 1536 rows
eligible native rows later revised:             50.0%
maximum eligible native revision:               40.9733px
```

The fixed write hook was not executed. Model validation and every locked dataset
remain unread. Formal decision:

```text
STOP_P0K_BEFORE_MODEL_VALIDATION_COMMIT_STATE_MISMATCH
```

Do not sweep write timing, bounds, confirmation, or state fields. See
`docs/ROUTED_STRONG_BACKBONE_BOUNDED_WRITEBACK_INTERFACE_RESULT_2026-07-19.md`.
