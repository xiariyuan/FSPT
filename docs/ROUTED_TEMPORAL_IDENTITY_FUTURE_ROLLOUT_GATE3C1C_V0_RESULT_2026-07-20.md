# Route-D temporal identity future rollout Gate 3C1C v0 result — 2026-07-20

## Formal status

```text
COMPLETED_PASS
AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION
```

The preregistered teacher-nearest shortlist state-action audit completed on the
already exposed fit-only indices 448--511 and reproduced exactly in a fresh
process.

## Commit support

The frozen query-closure identity shortlist contains native plus eight
non-native candidates. Only after the shortlist indices were frozen and hashed
did the audit reveal the frame-15 teacher and choose the nearest retained
candidate.

| Metric | Result |
|---|---:|
| failure rows | 753 |
| mean commit error | 12.9910 px |
| median commit error | 9.0194 px |
| recall within 4 px | 10.2258% |
| recall within 8 px | 42.3639% |
| recall within 12 px | 64.0106% |

## Future rollout actions

All three branches start from the exact same native frame-15 snapshot and keep
native visibility/confidence logits unchanged.

| Action | Mean future error | Severe >16 px | Threshold utility |
|---|---:|---:|---:|
| native | 34.4136 px | 0.97792 | 0.00501 |
| coordinate only | 33.2747 px | 0.97356 | 0.00624 |
| coordinate + four-level memory | **14.9708 px** | **0.30138** | **0.22518** |

Coordinate-only provides only a small improvement. The state-restored branch is
substantially better.

## Full state versus native

```text
mean future-error reduction:       +19.4428 px
video-cluster 95% CI:              [+17.4422, +20.4299] px
threshold-utility gain:            +0.22017
utility video-cluster 95% CI:      [+0.20416, +0.24277]
positive-point fraction:            98.2736%
severe >16 px rate reduction:      +0.67654
```

## Memory contribution beyond coordinate-only

```text
incremental future-error reduction: +18.3039 px
video-cluster 95% CI:               [+16.2614, +18.9950] px
incremental threshold utility:      +0.21895
utility video-cluster 95% CI:       [+0.20213, +0.24059]
positive-point fraction:             97.6096%
memory-better video fraction:        98.4127%
```

This establishes that the improvement is not a coordinate overwrite artifact.
Deterministic four-level memory reinstatement is the dominant mechanism.

## Integrity

- native recomputation matches the cached native future mean error exactly;
- selected shortlist indices, teacher-selected candidates, all three future
  coordinate streams, point records, video records, and the scientific payload
  reproduce exactly;
- original model-validation indices 48--63 remain unread;
- calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144
  remain unread or not rerun.

## Claim boundary

The positive result uses a teacher to choose the nearest coordinate inside the
causal shortlist. It proves shortlist reachability and state-restoration value,
not deployable top-1 selection or independent model-validation improvement.

The next authorized action is only to preregister an exact, unchanged run on the
locked original model-validation indices 48--63.

Machine-readable authority:

`docs/generated/ROUTED_TEMPORAL_IDENTITY_FUTURE_ROLLOUT_GATE3C1C_V0_RESULT_2026-07-20.json`
