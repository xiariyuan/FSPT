# Route-D strong-backbone DAVIS external transfer v0 plan — 2026-07-19

## 1. Decision being preregistered

P0l passes the complete identity-disjoint Kubric final holdout with the frozen
output-only variant C. P0m performs one frozen TAP-Vid-DAVIS first-query transfer
audit.

The model remains:

```text
frozen formal P0g CMCP
+ frozen rank-32 LMRA
+ frozen local safety comparator
+ frozen finalized CoTracker3 native state
+ output-only coordinate correction
```

No training, calibration, threshold selection or state writeback is permitted.

## 2. Historical-exposure boundary

DAVIS is not an untouched dataset for the project as a whole. Older recovery,
candidate and controller routes have previously used DAVIS for development and
diagnostics. P0m must therefore not describe DAVIS as an untouched final test.

The defensible claim is narrower:

> Variant C, its checkpoint, architecture, normalization and all P0m gates were
> frozen using Kubric only before the P0m evaluation. DAVIS is used once as a
> historically exposed but model-frozen zero-shot external transfer benchmark.

No existing DAVIS result may be used to alter variant C or the protocol below.

## 3. Pinned dataset and loader

```text
dataset:
/gemini/code/FSPT/datasets/tapvid_davis/tapvid_davis.pkl

size:
2,481,403,560 bytes

SHA-256:
8d4ed1b232fd27a78bc192e2e1ce943866ff640ab15e43db3bc7b10258bb48f5
```

The full 30-video pickle is evaluated in its pinned insertion order. The loader
is:

```text
TapVidDataset(
    dataset_type="davis",
    resize_to=[256, 256],
    queried_first=True,
    fast_eval=False,
)
```

No subset, fast-eval sample or video filtering is allowed.

## 4. Query, raster and visibility contract

The official first-visible sampler produces query points in `[t,y,x]` order.
All points in a video are passed jointly to CoTracker3.

```text
input raster: 256 x 256
metric raster: 256 x 256
query mode: first
support grid: disabled
native execution: true streaming, finalized state
prediction tracks stored: normalized [y,x]
official metric tracks: raster [x,y]
visibility: sigmoid(vis) * sigmoid(conf) > 0.6
```

Variant C changes coordinates only. It inherits native CoTracker visibility, so
OA is expected to remain unchanged; this expectation is not a success gate.

## 5. Cache-before-metrics rule

The external dataset is first sealed into a complete 30-video native/feature
cache. During cache construction, the code may report only:

- membership and completion;
- native/candidate-0 parity;
- feature quantization;
- hashes and extraction replay;
- runtime.

It may not calculate or display AJ, delta average, oracle gain, harmful rate or
per-video performance before all 30 videos are complete.

Full extraction replay is fixed at dataset indices `0`, `14` and `29`.

## 6. Frozen implementation

```text
variant-C checkpoint SHA-256:
7babb76e3407497832d0bc0fca4557df64b2d09450ebc60766e32a70816ab52f

combined model-state SHA-256:
64c3f6dae6ae34dc0223754aca3c76f78a1f48137d08bab6c3e01f2a806cd074

CoTracker3 checkpoint SHA-256:
205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218
```

The dataset loader, official metric port, metric wrapper, predictor, online model
and variant-C evaluator sources are pinned by SHA-256 in the YAML protocol.

## 7. Metrics

Report the complete aggregate and per-video paired rows for:

```text
AJ
average points within threshold
OA
1/2/4/8/16px profile
mean and median visible error
16px severe-error rate
candidate-coordinate oracle
selected non-native rate
harmful non-native rate
beneficial candidate availability and recall
```

Primary and replay evaluations must be exact.

## 8. Frozen gates

All gates are required:

```text
complete 30-video partition
candidate-0/native parity exact
candidate-oracle AJ gain >= +3.0 points
direct AJ gain >= +0.30 point
paired-video direct AJ CI lower bound > 0
delta-average gain > 0
16px severe-error rate not worse
pooled harmful non-native rate <= 1%
positive videos >= 18 / 30
exact primary/replay
```

The external direct-AJ magnitude gate is fixed at `+0.30`, lower than the
synthetic `+0.50` gate to account for a real-domain transfer shift while still
requiring a material gain. This value is frozen before any variant-C DAVIS
performance is generated.

## 9. Decisions

Pass:

```text
AUTHORIZE_STRONG_BACKBONE_EXTERNAL_TRANSFER_CLAIM
```

Allowed wording remains “frozen zero-shot transfer on a historically exposed
benchmark,” not “untouched final test.”

Fail:

```text
STOP_STRONG_BACKBONE_EXTERNAL_CLAIM_AND_RETAIN_SYNTHETIC_EVIDENCE_ONLY
```

A failure cannot trigger a threshold, candidate, checkpoint, calibration or
writeback rescue.

## 10. Kinetics lock

The completed official 1,144-video Kinetics result remains frozen historical
evidence. P0m does not rerun, rescore or tune against Kinetics.
