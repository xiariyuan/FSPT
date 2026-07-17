# Route-D causal multi-memory proposal generator v0 plan — 2026-07-17

## 1. Motivation

The frozen native-plus-local-top5 pool has `+6.8463 AJ` coordinate-oracle
headroom on model validation, yet structured, raw, and six-frame temporal
selectors recover `+0.0173`, `-0.0149`, and `+0.0041` AJ point. Temporal-v0 is
safe but recalls only `1.8443%` of beneficial global events.

This indicates that correctness is not recoverable after the frozen correlation
map has been reduced to five uncalibrated local maxima. The next controlled
change therefore moves learning before candidate discretization.

## 2. Distinct contribution

Implement a **Causal Multi-Memory Correlation Proposal Generator (CMCP)** on the
frozen CoTracker3 feature maps. This is not:

- direct coordinate-offset regression;
- another fixed-topK selector;
- a threshold, NMS, or window sweep;
- backbone fine-tuning;
- post-hoc state writeback.

CMCP learns a dense proposal field from causal correlation evidence and then
extracts candidate hypotheses. Native continuation remains candidate 0.

## 3. Frozen components

The following remain fixed:

- CoTracker3 checkpoint and true-streaming execution;
- query mode, 256x256 input and official first-query metrics;
- fit/model-validation identities and qualification protocol;
- native coordinate, visibility, confidence, and state sequence;
- seed 17, AdamW, learning rate, batch size, and evaluation gates;
- external-domain locks.

No final_holdout, calibration, DAVIS, or Kinetics result may be used for design or
selection.

## 4. Causal memories

At frame `t`, CMCP uses three support memories, all available causally:

1. **query anchor** — immutable normalized query-frame feature;
2. **previous native feature** — frozen feature sampled at the native coordinate
   at `t-1`;
3. **causal EMA memory** — fixed `alpha=0.9` feature EMA along the frozen native
   trajectory, initialized from the query anchor.

There is no confidence threshold or validation-selected memory update rule.
Before the query frame, all proposal supervision and memory reads are masked.

## 5. Dense proposal module

For each memory, compute a full normalized correlation map against the current
frozen CoTracker feature map. Stack:

- three current correlation maps;
- their pairwise differences;
- a native-centered motion-prior map;
- the previous two fused proposal-evidence maps.

A compact causal ConvGRU with a two-layer convolutional decoder outputs:

- a dense utility heatmap on the frozen feature grid;
- a dense catastrophic-risk heatmap;
- one native-fallback logit.

The initial decoder is native-safe: dense proposal logits are zero and the
native fallback wins exactly before training.

Candidate extraction is fixed before training:

```text
candidate 0: frozen native coordinate
candidates 1--5: stable top-5 local maxima of the learned utility-minus-risk map
NMS radius: one feature cell
coordinate conversion: exact feature-grid-to-input-raster mapping
```

No NMS radius, top-K, or search-region sweep is permitted on model validation.

## 6. Fit-only supervision

Train on fit only using:

- a Gaussian dense target centered at the visible GT coordinate;
- the same fixed 1/2/4/8/16px utility weights used by MUSR;
- focal heatmap loss for sparse localization;
- catastrophic-risk BCE for locations >=16px from GT;
- native-fallback supervision when no learned peak improves native utility;
- temporal consistency between adjacent causal proposal maps;
- explicit no-harm margin against the native coordinate.

This changes correlation supervision and candidate generation, not just the
selector consuming fixed hypotheses.

## 7. Evaluation order and gates

Before any MUSR selector or state-write training, evaluate CMCP proposal-only on
complete model validation.

Required gates:

```text
zero-step native coordinate parity: exact
candidate export deterministic replay: exact
new candidate-pool coordinate oracle AJ gain: >= +3.0 points
direct proposal top-1 AJ gain: >= +0.5 point
paired-video top-1 AJ CI lower bound: > 0
delta-average gain: > 0
16px severe-error rate: not worse
harmful non-native proposal rate: <= 1%
```

If the proposal-only gate passes, CMCP becomes the substantial trainable
candidate generator and MUSR may be reintroduced only as a second-stage
ablation. If it fails, do not tune NMS, memory alpha, or proposal thresholds on
model validation. The remaining route would require partial joint fine-tuning of
late CoTracker correlation layers under the same frozen data protocol.

## 8. Locked data

```text
fit: dense proposal gradients allowed
model_validation: checkpoint selection and gates only
calibration: unread
final_holdout: unread
TAP-Vid-DAVIS: unread
TAP-Vid-Kinetics official 1,144: unread and not rerun
```
