# Route-D late feature metric residual adapter v0 plan — 2026-07-17

## 1. Motivation

P0h produces a safe and statistically positive comparator result, but its best
model-validation AJ gain is only about `+0.245` points, below the preregistered
`+0.5` magnitude gate. Harmful non-native selection is below `0.32%` and the
paired AJ confidence interval is positive, while beneficial-candidate recall is
only about `3.34%`. The frozen CMCP pool still has about `+20.04` AJ oracle
headroom on every validation video.

The next controlled change must improve the local correlation evidence available
to the safe comparator without allowing the native tracker trajectory to drift.
It must not be a threshold, NMS, EMA, top-K, or abstention sweep.

## 2. Architecture: Late Metric Residual Adapter (LMRA)

Apply a trainable residual metric adapter after the frozen CoTracker3
`fnet.conv3` output and before all query/previous/EMA correlations:

```text
z = normalize(f + W_up(GELU(W_down(f))))
W_down: 128 -> 32, 1x1 convolution
W_up:    32 -> 128, 1x1 convolution
```

`W_up` is zero-initialized, so zero-step adapted features are exactly the frozen
feature maps. The adapter has 8,352 trainable parameters. This is the only late
CoTracker feature component that may change.

The original CoTracker online trajectory, visibility, confidence, query times,
and candidate-0 coordinates remain byte-frozen. The adapter is a proposal and
local-evidence branch; it does not feed back into the native updateformer.

## 3. Joint trainable components

Initialize from the exact formal checkpoints:

- P0g epoch-1 CMCP dense proposal generator;
- P0h epoch-3 local pairwise safety comparator;
- zero-initialized rank-32 LMRA.

Train on fit only:

- LMRA;
- CMCP proposal generator;
- local safety comparator.

The frozen native trajectory branch and all other CoTracker parameters receive
no gradients. Hard NMS coordinates are detached, while dense proposal losses
train LMRA/CMCP and local candidate losses train LMRA/CMCP/comparator at the
selected locations.

## 4. Loss and safety contract

Use the already frozen P0g dense proposal losses and P0h comparator losses, plus:

- residual feature-distortion penalty against the frozen normalized map;
- exact zero-step P0h metric/candidate parity;
- safety-first checkpoint selection: harmful non-native `<=1%` and severe 16px
  not worse, then maximum direct AJ.

No validation-selected threshold or layer/rank sweep is permitted. Adapter rank
is fixed at 32 before training.

## 5. Data and gates

```text
fit: LMRA + CMCP + comparator gradients
model_validation: checkpoint selection and formal gates only
calibration: unread
final_holdout: unread
DAVIS: unread
Kinetics official 1,144: unread and not rerun
```

Required gates:

```text
zero-step equality to formal P0h: exact
native trajectory and candidate-0 hashes: exact
adapter rank and trainable-layer identity: exact
seed-17 replay: exact
candidate oracle AJ gain: >= +3.0 points
direct AJ gain: >= +0.5 point
paired-video AJ CI lower bound: > 0
delta-average gain: > 0
16px severe-error rate: not worse
harmful non-native selection: <= 1%
```

If LMRA fails, do not unfreeze earlier encoder layers or sweep adapter rank on
model validation. The strong-backbone rescue route is then closed for the
current data scale, and the paper should retain the verified mechanism and
negative-gate narrative rather than claim competitive full-system performance.


## 6. Completed interface milestone

The 8,352-parameter rank-32 LMRA passes exact zero-step reconstruction on
Kubric fit video 0. The adapted feature map is byte-identical to the frozen map,
and the online LMRA -> CMCP -> comparator path reproduces the formal P0h
candidate coordinates, 88-D tokens, causal decisions, and selected coordinates.
The candidate-coordinate SHA-256 remains
`c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300`.
Independent full-video runs reproduce every tensor hash. Fit-only joint training
of LMRA, initialized CMCP, and initialized comparator is authorized.
