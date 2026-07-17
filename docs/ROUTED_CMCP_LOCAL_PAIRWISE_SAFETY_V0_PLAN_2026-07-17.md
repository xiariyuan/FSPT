# Route-D CMCP local pairwise safety comparator v0 plan — 2026-07-17

## 1. Motivation

Formal CMCP training produces a strong learned proposal pool on complete Kubric
model-validation: coordinate-oracle AJ improves by about `+20.04` points and all
16 videos are positive. Direct top-1 improves pooled AJ by about `+1.58` points
and reduces the 16px severe-error rate, but the paired-video AJ confidence
interval crosses zero and harmful non-native selection is about `28%`.

The failure is therefore not proposal availability. The current native fallback
head globally averages the entire ConvGRU hidden map before producing one native
logit. It cannot explicitly compare local evidence at the native coordinate with
local evidence at each learned proposal peak. This plan freezes the successful
proposal generator and changes only that comparison mechanism.

This is a new post-failure preregistration. It does not retroactively change the
P0g-c gates or reinterpret P0g-c as a pass.

## 2. Frozen components

The following are byte-frozen:

- CoTracker3 checkpoint and true-streaming execution;
- fit/model-validation identities and feature-map caches;
- CMCP query, previous-native, and fixed-alpha EMA memories;
- the nine-channel recurrent input, ConvGRU, dense utility/risk heads, and the
  exact seed-17 best proposal checkpoint;
- proposal top-K `5`, one-cell NMS, coordinate conversion, and candidate order;
- candidate 0 as frozen native continuation;
- all generated proposal coordinates and dense proposal-map outputs;
- utility thresholds, utility weights, seed 17, AdamW, and external-data locks.

No NMS, EMA, top-K, score threshold, or proposal-coordinate tuning is allowed.

## 3. Local pairwise safety comparator

For candidate 0 and each of the five frozen CMCP peaks at frame `t`, sample a
local token containing:

```text
64-D CMCP hidden feature at candidate location
9-D recurrent input evidence at candidate location
utility logit, risk logit, and proposal score
native visibility, confidence, and joint probability
candidate displacement from native, normalized x/y and magnitude
candidate rank and native-vs-candidate score margin
causal previous-frame candidate decision summary
```

A shared candidate encoder followed by a small candidate-set Transformer predicts:

- monotonic correctness probabilities at 1/2/4/8/16px;
- catastrophic-error probability;
- a pairwise candidate-vs-native preference logit;
- an explicit abstain-to-native probability.

The final scoring rule is fixed in the model graph. There is no
model-validation-selected threshold. Zero initialization must select native
exactly for every row.

## 4. Training protocol

```text
proposal generator: frozen at formal P0g-c best checkpoint
fit: comparator gradients only
model_validation: checkpoint selection and formal gates only
calibration: unread
final_holdout: unread
DAVIS: unread
Kinetics official 1,144: unread and not rerun
```

Supervision uses fit GT only:

- multi-threshold utility BCE;
- candidate-vs-native pairwise ranking;
- catastrophic-risk BCE;
- harmful non-native selections weighted 10x;
- abstain-to-native target when no candidate has strictly greater utility;
- native-safe no-harm margin.

The candidate generator is never updated during this stage, so proposal oracle
headroom and coordinates must remain exactly equal to P0g-c.

## 5. Gates

Complete model-validation must satisfy all:

```text
candidate-coordinate hashes equal frozen P0g-c: exact
zero-step native parity: exact
seed-17 replay of model-state hash/history/metrics: exact
candidate oracle AJ gain: unchanged and >= +3.0 points
direct top-1 AJ gain: >= +0.5 point
paired-video top-1 AJ CI lower bound: > 0
selected delta-average gain: > 0
16px severe-error rate: not worse
harmful non-native selection rate: <= 1%
```

If this comparator fails, do not tune a threshold on model-validation. The next
route is the preregistered zero-initialized rank-32 late feature metric residual
adapter after frozen CoTracker `fnet.conv3`, used only by the proposal/evidence
branch while the native trajectory remains byte-frozen.


## 6. Completed interface milestone

The frozen formal epoch-1 CMCP generator and the 409,224-parameter comparator
pass the real fit-video interface audit. Candidate 0 and zero-step selected
coordinates equal native for all active rows. The complete candidate-coordinate
tensor SHA-256 is
`c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300`,
and all candidate/selection tensors are bit-identical in independent full-video
runs. This authorizes frozen local-token cache export and comparator-only fit
training; it is not learned comparator performance.


## 7. Completed frozen-token cache milestone

Frozen 88-D token caches are complete for 48 fit and 16 model-validation videos.
The first 84 dimensions are static GT-free local evidence; the final four are
zero placeholders replaced online by the preceding comparator decision. The fit
and model-validation candidate-coordinate combined hashes are
`3f1be12649a8c4757c933c3f6d8c99b9f3d6bcd146262406557e139377ca57c2`
and `edf8ebe25cc11b04406efe27203a7d2bea4da8ce075160a27a11d8edda1fb0a1`.
Replay anchors 0/47/48/63 are exact. Comparator-only fit training is authorized.


## 8. Completed comparator-only result

Formal seed-17 comparator training is exactly reproducible. Best epoch 3 improves
model-validation AJ by `+0.2449` and delta average by `+0.4962`; paired AJ 95%
CI is `[+0.1324,+0.3693]`, 15/16 videos are positive, harmful non-native
selection is `0.3132%`, and the 16px severe-error rate improves. The frozen
candidate oracle remains exactly `+20.0375` AJ.

All safety and statistical gates pass, but the preregistered direct-AJ magnitude
gate `>= +0.5` fails. Beneficial-candidate recall is only `3.3362%`. P0h is
therefore closed without threshold tuning. P0i uses a zero-initialized rank-32
late feature metric residual adapter in the proposal/evidence branch while
keeping the native trajectory byte-frozen.
