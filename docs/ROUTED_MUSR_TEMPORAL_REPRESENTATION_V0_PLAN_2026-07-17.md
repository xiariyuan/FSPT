# Route-D MUSR causal temporal representation v0 plan — 2026-07-17

## 1. Motivation

The frozen candidate pool has `+6.8463 AJ` coordinate-oracle headroom on model
validation. Neither the structured 64/32 representation (`+0.0173 AJ`) nor the
raw single-frame 601/160 representation (`-0.0149 AJ`) recovers meaningful AJ.
The raw-v1 selector recalls only `8.27%` of beneficial non-native events.

The events are not independent: a beneficial event persists to the next visible
frame with probability `69.31%`, versus `18.65%` after a non-beneficial frame.
The median run is two frames and the 90th percentile is six frames. This
pre-registers one scientifically distinct change: causal temporal evidence.

## 2. Frozen components

The following remain byte-identical:

- CoTracker3 checkpoint and true-streaming loop;
- query mode and 256x256 raster;
- native plus five local-correlation candidate coordinates;
- search radius, top-K, candidate order, masks, and source IDs;
- raw-v1 per-frame 601-D candidate and 160-D state features;
- utility thresholds and weights;
- fit/model-validation identities;
- seed 17, optimizer, learning rate, batch size, and Stage-A gates.

No candidate generator, threshold, writeback, abstention, or gate sweep is
allowed.

## 3. Temporal model: Causal Set-Evidence Accumulator

Candidate ranks are local correlation ranks and do not define persistent object
identities across frames. The temporal module must therefore be rank-invariant
for past candidate sets.

For each point and frame `t`:

1. encode the current six-candidate set with the existing shared candidate-set
   Transformer;
2. produce a rank-invariant frame evidence token by attention pooling over the
   valid candidate tokens, conditioned on the 160-D state token;
3. feed the current and previous five frame-evidence tokens into a causal
   temporal Transformer, giving a fixed six-frame receptive field;
4. condition each current-frame candidate token on the resulting temporal
   context;
5. predict the same monotonic multi-threshold utility, catastrophic risk, and
   native-vs-global score used in Stage A.

The six-frame window is fixed before training because it matches the observed
90th-percentile beneficial run length. It will not be swept on model validation.
Missing pre-query history is zero-padded and masked. No future frame is visible.

## 4. Training and evaluation

```text
training rows: causal point sequences from fit only
checkpoint selection: complete model_validation only
stage: selector-only Stage A
state-write heads: frozen
abstention head: frozen
calibration: unread
final holdout: unread
DAVIS: unread
Kinetics: unread and not rerun
```

Training clips preserve per-point temporal order. Evaluation processes each
video in causal frame order and emits only the current-frame selection.
Zero-initialized temporal conditioning must reproduce exact native selection
before training.

## 5. Gates

Complete model validation must satisfy all:

```text
AJ gain >= +0.5 point
paired-video AJ CI lower bound > 0
delta-average gain > 0
16px severe-error rate not worse
harmful global selection rate <= 1%
zero-step exact native parity
seed-17 exact replay of model-state hash, history, and metrics
```

If this fixed temporal model fails, do not sweep window sizes or thresholds on
model validation. The next route must change candidate generation or add
fit-only supervised correspondence between candidate hypotheses, not another
selector feature expansion.
