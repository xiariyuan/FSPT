# Route-D CMCP interface result — 2026-07-17

## 1. Scope

This is a zero-step interface audit on authorized Kubric fit video 0. It does
not contain learned proposal performance and does not authorize external or
final-holdout evaluation.

## 2. Implemented module

The Causal Multi-Memory Correlation Proposal Generator (CMCP) moves learning
before top-K candidate discretization. It consumes three causal dense
correlation fields:

1. immutable query-anchor correlation;
2. previous-native-feature correlation;
3. fixed-alpha (`0.9`) EMA-memory correlation.

These are fused with a native-centered motion prior and two previous proposal
evidence maps by a causal ConvGRU. A dense utility head, dense catastrophic-risk
head, and native-fallback head produce native plus five stable NMS proposals.

Formal configuration:

```text
hidden channels:       64
trainable parameters:  262,019
proposal top-K:        5
NMS radius:            1 feature cell
motion sigma:          2 feature cells
native logit bias:     2.0
feature map:            128 x 96 x 128 channels/height/width
```

The final utility/risk heads are zero-initialized and native fallback is
positively biased, giving exact zero-step native behavior.

## 3. Real one-video audit

```text
partition: fit
source index: 0
video: 1680
points: 64
frames: 24
CoTracker feature maps: 24 x 128 x 96 x 128
```

The frozen CoTracker3 native coordinate, visibility, confidence, joint
probability, and visibility decision tensors exactly match the previously
qualified stage-0 sidecar.

All query-after-frame rows satisfy:

```text
candidate 0 coordinate == frozen native coordinate
candidate 0 valid == true
selected candidate index == 0
selected coordinate == frozen native coordinate
```

No GT is used to construct memories, correlations, proposal maps, candidates,
or selections.

## 4. Determinism

Within the first run, the first point chunk is recomputed independently. The
following tensors are bit-identical:

- all three correlation maps;
- motion prior;
- proposal score map;
- native logit;
- candidate coordinates, scores, and masks;
- selected candidate index and coordinate;
- causal frame-valid mask.

The entire 64-point video is then rerun in a separate process. Every versioned
output tensor hash is identical across the two runs.

Key hashes:

```text
correlation first-chunk SHA-256:
09088c91a0e6387c28c0275588e8e1982849956875bfe10c999996571279a73d

proposal-score first-chunk SHA-256:
edea98338a940efe0814c0e57b328d7b556ba244354899c72e586200c62b0efe

selected-coordinate SHA-256:
416f90ac9df0d9c48aefc25dceb667ff6e5a46efada81f3677ad4f3c22a2da96
```

## 5. Decision

```text
ALLOW_CMCP_FEATURE_MAP_CACHE_AND_FIT_ONLY_PROPOSAL_TRAINING
```

The next step is to cache each frozen CoTracker feature map once in float16 for
fit and model-validation, then construct multi-memory correlations online during
CMCP training. This avoids storing three point-specific correlation volumes per
video while preserving exact causal inputs.

MUSR selector training, state writeback, calibration, final_holdout, DAVIS, and
Kinetics remain locked until the proposal-only gate passes.
