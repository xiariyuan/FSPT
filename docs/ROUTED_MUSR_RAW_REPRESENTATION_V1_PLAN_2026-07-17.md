# Route-D MUSR raw candidate representation v1 plan — 2026-07-17

## 1. Motivation

The frozen six-candidate pool provides `+6.85 AJ` coordinate-oracle headroom on
model validation, while the formal utility-aligned selector recovers only
`+0.017 AJ point`. The current 64-D candidate vector compresses 128-channel
CoTracker features into channel-bin means and scalar summaries. Earlier project
history and the new Stage-A result both indicate that handcrafted summaries are
insufficient for cross-video candidate discrimination.

## 2. Single controlled change

Candidate coordinates, search radius, top-K, query mode, backbone, partitions,
and all evaluation gates remain unchanged. Only the representation is upgraded.

### Candidate token

For every candidate:

```text
existing structured candidate vector:       64
full query support feature:                 128
full current candidate feature:             128
support-current difference:                 128
support-current elementwise product:        128
candidate-centered 5x5 correlation patch:    25
------------------------------------------------
total candidate feature dimension:          601
```

The correlation patch is sampled from the same causal query-support/current-frame
feature map used to generate candidates. It contains no GT or future frame.

### State token

```text
existing structured state vector:            32
full CoTracker online_track_feat level 0:    128
------------------------------------------------
total state feature dimension:               160
```

No support/template state is modified.

## 3. Model and training

Retain the current candidate-set Transformer, monotonic threshold utility,
catastrophic-risk head, native-safe initialization, and utility-aligned Stage-A
loss. Train selector-only; abstention and state-write heads stay frozen.

The representation must be normalized from fit only. Model validation selects a
checkpoint. Calibration is not read.

## 4. Gate

Raw selected coordinates on complete model validation must satisfy all:

```text
AJ gain >= +0.5 point
paired-video AJ CI lower bound > 0
delta-average gain > 0
16px severe-error rate not worse
harmful global selection rate <= 1%
zero-step exact native parity
seed-17 exact replay of model-state hash and metrics
```

If this representation fails, do not sweep patch sizes, rank thresholds, or
gate margins on model validation. The next scientifically distinct route would
be a sequence-level temporal candidate representation trained on fit only.

## 5. Locked data

```text
fit: allowed for gradients
model_validation: checkpoint selection only
calibration: locked until selector gate passes
final_holdout: locked
tapvid_davis: locked
tapvid_kinetics official 1,144: locked and not rerun
```

## 6. Completed result

Raw-v1 caches are complete for fit/model-validation at 48/16 videos. All
candidate-coordinate hashes remain unchanged and four replay anchors are
byte-identical. The formal seed-17 selector is exactly reproducible but fails:
AJ gain is `-0.0149` point with paired 95% CI `[-0.0806,+0.0502]`. Delta improves
`+0.1290`, severe-16px rate improves, and harmful selection remains below 1%,
but the AJ and positive-CI gates fail.

Simple marginal feature shift is small. The selector recalls only `8.27%` of
beneficial non-native events, while those events show strong temporal
persistence. Therefore raw-v1 is closed without patch-size, threshold, gate, or
writeback sweeps. The next route is the fixed causal temporal model in
`docs/ROUTED_MUSR_TEMPORAL_REPRESENTATION_V0_PLAN_2026-07-17.md`.
