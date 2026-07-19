# Route-D CSRR training and continuation interface result — 2026-07-19

## Decision

```text
ALLOW_FORMAL_GATE2_PRIMARY_TRAINING
```

Two result-free interface smokes pass after the formal runner implementation was
committed.

## Deterministic backward smoke

```text
source index:                    8
rows:                            4
trainable parameters:            19,685
all parameter gradients finite: true
model-validation read:           false
```

The smoke executes the complete model, structured re-extraction, all frozen loss
terms, and backward propagation. The initial aggregate loss was finite. No
fit-internal validation metric was computed.

The original CoTracker 3D `grid_sample` backward emitted a CUDA nondeterminism
warning. Before formal training, commit `e9e055d` replaced only the training
backward sampler with an analytically equivalent single-frame bilinear gather.
The frozen feature values are detached and gradients flow through deterministic
bilinear weights into coordinates. A known planar field test gives exact value
`17.5`, gradient `[1,10]`, and repeated backward equality. Official CoTracker
sampling remains in the no-gradient validation path.

The corrected smoke reproduces the same finite loss without a nondeterministic
backward warning.

## Causal continuation smoke

```text
source index:                    32
cached rows:                     6
zero-action state parity:        exact
coordinate-only final shape:     [1,24,64,2]
coordinate+probability shape:    [1,24,64,2]
full-state final shape:           [1,24,64,2]
learned-gate final shape:         [1,24,64,2]
ground-truth metrics computed:   false
model-validation read:           false
```

This verifies that every frozen C/P/F control and the learned-gate union can be
applied to the exact cached commit state and consumed by the next CoTracker
window. It does not expose or select using fit-internal performance.

## Locked boundary

Only cache sources `8` and `32` were loaded. Source indices `48–63`, calibration,
final holdout, DAVIS, and Kinetics remain unread. Official Kinetics 1,144 was not
rerun.
