# Route-D temporal identity feature probe Gate 3C1 v0 — 2026-07-19

## Status

```text
COMPLETED
```

## Purpose

Gate 3C0 passed and authorizes a causal feature cache on the expanded Kubric
population. Before fixing a selector architecture or launching a multi-video
cache, this probe measures the actual causal tensor shapes, runtime, and peak
GPU memory on exactly one gradient-train sample.

This is not a performance experiment. It cannot establish candidate recall,
tracking improvement, or publication evidence.

## Frozen membership

- expanded manifest index: 64;
- split: gradient train only;
- probe row: the first failure under the frozen Gate 2 natural-failure ranking;
- original query frame less than 8;
- target visible at frame 15;
- at least four visible GT frames in 16--23;
- native future mean error at least 16 px.

Teacher/future information may determine the design-only probe membership and
may be read after candidates are frozen for diagnostic distances. It may not
enter candidate generation, reverse tracklets, DINO observations, or feature
tensors.

## Frozen candidate and temporal evidence

The bank is mandatory native candidate zero plus 128 nonnative M1
geometry-native proposals, using the Gate 3C0 NMS and local refinement
parameters.

Each valid proposal is queried at reversed frame 15 and tracked through the
reversed observed clip to frame 0 with frozen CoTracker3. Frozen DINOv3
features are sampled along the resulting tracklet. The probe constructs:

- 9 temporal channels over 16 observed frames;
- 14 static/list channels;
- one validity mask;
- post-freeze teacher distance diagnostics only.

No learned parameter, selector score, threshold, fusion-weight search, future
frame feature, or external datum is permitted.

## Output

The machine-readable result must record:

- exact parent/config/checkpoint/weight hashes;
- selected source and point identity;
- candidate and valid-candidate counts;
- CoTracker and DINO feature shapes;
- temporal and static tensor shapes and hashes;
- feature storage bytes;
- stage runtimes and peak CUDA memory;
- finite-value checks;
- locked-data flags;
- candidate hash before teacher diagnostic access.

The output is descriptive. Architecture, loss, checkpoint selection, and Gate
3C1 pass thresholds remain unauthorized until this probe is complete.

## Completion

The probe completed and replayed exactly. The authoritative result is:

`docs/ROUTED_TEMPORAL_IDENTITY_FEATURE_PROBE_GATE3C1_V0_RESULT_2026-07-19.md`
