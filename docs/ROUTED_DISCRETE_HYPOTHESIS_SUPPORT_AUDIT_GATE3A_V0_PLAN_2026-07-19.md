# Route-D discrete-hypothesis support audit Gate 3A v0 — 2026-07-19

## Status

```text
PREREGISTERED_NOT_RUN
```

## Question

Does the frozen Gate 2 fused correlation map already contain a discrete candidate
near the correct frame-15 position, with the current global softmax expectation
being the principal source of localization failure?

Gate 3A is an audit, not a training stage. It freezes the exact Gate 2 epoch-6
checkpoint and existing sidecars, extracts a deterministic top-K candidate set
without GT, and only then uses the frame-15 teacher coordinate to score candidate
support.

## Why this gate precedes another model

Gate 2 proves that action-only repair is insufficient: even perfect future-aware
selection of the current full-state action has maximum union utility gain
`0.0228125`, below the frozen `0.05` gate. The current restorer must produce more
accurate hypotheses before a learned action policy can matter.

The audit distinguishes two cases:

1. **candidate support is good, readout is bad** — coordinate-oracle top-K
   restoration passes; train a causal discrete candidate selector next;
2. **candidate support is bad** — even nearest-candidate oracle fails; redesign
   support representation and temporal matching before any selector training.

## Frozen candidate construction

For each existing failure row:

1. run the frozen Gate 2 model and retain its `64 x 64` fused logits;
2. add the native commit coordinate as candidate 0;
3. extract eight nonnative peaks with deterministic 3-cell NMS;
4. refine each peak with a 5x5 local softmax at temperature 0.05;
5. remove candidates within 4 input pixels of an earlier higher-score candidate;
6. freeze candidate coordinates and hashes;
7. only then reveal the GT frame-15 teacher coordinate for audit scoring.

No future outcome selects a candidate. The coordinate oracle chooses the frozen
candidate nearest the commit teacher and then executes the same structured
memory re-extraction and causal continuation as Gate 2.

## Controls

```text
N: native no action
S: current Gate 2 global soft-expectation full state
T: discrete top-1 local-refined full state
O: nearest-candidate coordinate-oracle full state
```

`S` must reproduce the Gate 2 selected-checkpoint output exactly. `O` measures
candidate-set support, not deployable selector performance.

## Pass condition

The audit passes only if the frozen candidate set supports both accurate commit
localization and future rollout:

```text
recall within 8 px >= 0.75
median commit error <= 4 px
future mean-error reduction vs native >= 8 px
error-reduction CI lower >= 2 px
future utility gain vs native >= 0.12
utility CI lower >= 0.03
positive-point fraction >= 0.65
severe-rate reduction >= 0.15
utility gain over current soft expectation >= 0.05
```

A pass authorizes only fit-only discrete selector training. It does not authorize
an action policy or reading model-validation `48–63`.

## Replay

Primary and fresh-process replay must match exactly for candidate coordinates,
NMS ranks, selected oracle candidate identities, all initial states, all future
rollout tensors, and aggregate reports. JSON output paths are the only permitted
difference.

## Locked boundary

Source indices `48–63`, calibration, final holdout, DAVIS, and Kinetics remain
unread. The official 1,144-video Kinetics result must not be rerun.
