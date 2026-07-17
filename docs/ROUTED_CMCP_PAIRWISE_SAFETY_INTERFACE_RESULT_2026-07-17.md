# Route-D CMCP local pairwise safety interface result — 2026-07-17

## 1. Scope

This is a zero-step interface audit on authorized Kubric fit video 0. The formal
P0g-c epoch-1 CMCP proposal generator is frozen. No comparator learning result,
model-validation claim, calibration result, final-holdout result, DAVIS result,
or Kinetics rerun is included.

## 2. Frozen proposal generator

```text
checkpoint model-state SHA-256:
fc3044eb6daa1fb2416164fc0afe58ff5fa4eb4442811ce9c0b4917fbbdb57c0

candidate count: native + five CMCP peaks
NMS radius: one feature cell
EMA alpha: 0.9
generator parameters requiring gradients: zero
```

The full 64-point, 24-frame candidate-coordinate tensor has frozen SHA-256:

```text
c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300
```

Candidate 0 remains exactly equal to the frozen CoTracker3 native coordinate.

## 3. Local comparator contract

Each candidate receives an 88-dimensional local token:

```text
CMCP hidden feature:                   64
nine-channel recurrent input:          9
dense utility/risk/proposal values:     3
native visibility/confidence/joint:     3
native-relative displacement:           3
candidate rank:                         1
candidate-vs-native score margin:       1
previous causal decision summary:       4
-----------------------------------------
total:                                 88
```

A shared encoder and two-layer candidate-set Transformer predict monotonic
1/2/4/8/16px correctness probabilities, catastrophic risk, pairwise
preference, and abstention-to-native. The comparator has 409,224 trainable
parameters. Its output heads are native-safe initialized.

## 4. Real interface audit

```text
partition: fit
source index: 0
video: 1680
points: 64
frames: 24
candidate count: 6
```

All active rows satisfy:

```text
candidate 0 == frozen native coordinate
selected candidate index == 0
selected coordinate == frozen native coordinate
```

The comparator receives no GT for token construction or selection and cannot
modify the frozen candidate-coordinate tensor.

## 5. Determinism

The complete video is evaluated twice within each process and in two independent
processes. Candidate coordinates, candidate scores, masks, frame-valid masks,
selected indices, and selected coordinates are bit-identical.

```text
frozen candidate-coordinate SHA-256:
c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300

zero-step selected-coordinate SHA-256:
416f90ac9df0d9c48aefc25dceb667ff6e5a46efada81f3677ad4f3c22a2da96
```

## 6. Decision

```text
ALLOW_FROZEN_CMCP_TOKEN_CACHE_AND_COMPARATOR_ONLY_TRAINING
```

The next step may export frozen local candidate tokens for fit and
model-validation, then train only the comparator on fit. The CMCP generator,
proposal coordinates, NMS, EMA, top-K, backbone, state writeback, calibration,
final holdout, DAVIS, and Kinetics remain locked.
