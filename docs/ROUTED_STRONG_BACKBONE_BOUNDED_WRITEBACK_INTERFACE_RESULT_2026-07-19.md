# Route-D strong-backbone bounded coordinate writeback interface result — 2026-07-19

## 1. Formal decision

The preregistered P0k interface gate fails before model-validation access:

```text
STOP_P0K_BEFORE_MODEL_VALIDATION_COMMIT_STATE_MISMATCH
```

The fixed 8px coordinate-only hook was **not executed**. The complete 16-video
model-validation partition was **not read**. Calibration, final holdout, DAVIS,
and official Kinetics also remain unread.

The failure is not a numerical-reconstruction bug. The formal P0j-C output-only
path is reproduced exactly. The failure is a temporal-state contract mismatch:
formal C acts on the finalized CoTracker overlap state, whereas next-window
writeback must decide from the provisional state before that overlap is revised.

## 2. Frozen interface

```text
fit source index:              0
video:                         1680
frames / points:               24 / 64
variant-C checkpoint SHA-256:  7babb76e3407497832d0bc0fca4557df64b2d09450ebc60766e32a70816ab52f
P0k config SHA-256:             d3db199673295bfd294975dfbe05fd7516e3fd801003c16254a3b26e0f394ee6
feature-index SHA-256:          bfb10726a317b7bdb1a7993ab1e417440dc561dad03c748a2c5e958f2b5bac76
```

The audit tracks all 64 points jointly under the original CoTracker3
true-streaming execution. It does not substitute per-query inference.

## 3. Checks that pass

### 3.1 Formal P0j-C reconstruction

The new framewise implementation exactly reproduces the existing formal C cache
path:

| Tensor | Exact | Max abs |
|---|:---:|---:|
| candidate coordinates | yes | 0 |
| candidate valid mask | yes | n/a |
| selected indices | yes | n/a |
| selected coordinates | yes | 0 |

This establishes that LMRA normalization, query/previous/EMA memories, frozen
CMCP state, local tokens, comparator decisions, and causal summaries are
implemented consistently with P0j.

### 3.2 Native true-streaming parity

After a complete no-write live run, the final native state exactly matches the
frozen base cache:

```text
coordinates: exact
visibility probability: exact
confidence probability: exact
```

Thus the mismatch is not caused by a different backbone, query set, raster,
support grid, or streaming schedule.

### 3.3 Reproducibility

Two no-write predictor instances inside each audit are exact. A second independent
process reproduces all report fields and all 22
nested tensors/scalars exactly.

```text
primary report SHA-256: edbff80c75b287a6143155e0745904de39532a1d07f55a1c94d0bf1b8c3e28fd
replay report SHA-256:  e7d3db81bfc260503563754dbc97f4a864b096a82f2fd1e37580dc190bf1c5a0
```

Torch archive byte hashes differ because serialization containers are not
canonical; tensor-keyed content is exact.

## 4. Commit-time mismatch

The deployable provisional commit-time stream does not equal formal C:

| Field | Exact | Largest difference |
|---|:---:|---:|
| candidate coordinates | no | 56.3684px per component |
| candidate valid mask | yes | 0 |
| selected indices | no | categorical mismatch |
| selected coordinates | no | 36.5723px per component |
| dynamic summary | no | 1.0000 |

Structural counts over all 1,536 point-frame rows:

```text
selected-index mismatches:              2 / 1536
selected-coordinate mismatches:         502 / 1536
selected-coordinate mismatch fraction:  32.6823%
candidate-coordinate max L2 difference: 60.5410px
```

The small selected-index mismatch count does not make the route deployable.
Candidate 0 is the native coordinate, so even rows that retain index 0 inherit a
different provisional native position.

## 5. Why the writeback gate fails

The write-eligible overlap contains 1,024 point-frame rows. Before the next
window finalizes that overlap:

```text
native rows later revised:        50.00%
maximum native revision:          40.9733px
selected-index mismatches:        2 / 1024
formal non-native actions:        29
commit-time non-native actions:   29
```

Although the aggregate non-native action counts happen to match, the action
identity and coordinate state do not. A hook that writes formal C decisions at
the next-window commit point would therefore use information produced only after
the state it intends to modify has already been revised.

This violates the preregistered exact interface gate. Running model validation
would not repair the causal contract and would constitute post-failure tuning.

## 6. Claim correction

P0j variant C remains a valid result under its frozen contract:

> finalized-state output-only evaluation on Kubric model validation.

P0k establishes that C cannot be reinterpreted as a zero-latency controller for
next-window CoTracker state writeback. The paper must not claim a closed-loop
strong-backbone gain from C.

The safest wording is:

> Late metric adaptation plus local safety comparison improves finalized
> strong-backbone outputs, while direct state feedback is blocked by the
> provisional-versus-final overlap-state mismatch of the online tracker.

## 7. Stop decision

Coordinate-only writeback is closed for the current paper. Do not sweep:

```text
write timing or latency
8px bound
confirmation length
visibility/confidence writes
candidate threshold
LMRA rank
CMCP checkpoint
NMS / top-K / EMA
```

The strong-backbone main result remains output-only variant C. Any future
closed-loop method would require training and evaluating explicitly on the
commit-time provisional state under a newly frozen protocol; that is a new
research route, not a P0k rescue.

## 8. Canonical evidence

```text
docs/generated/ROUTED_STRONG_BACKBONE_BOUNDED_WRITEBACK_INTERFACE_SUMMARY_2026-07-19.json
SHA-256: 4fbccf4e17d212d46f3cf41d7a6b5218400264fd1188dc6ad342687c07a75475
```
