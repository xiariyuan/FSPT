# Route-D visibility-coupling cache Gate 3C1G0 v0 plan — 2026-07-20

## Status

```text
PREREGISTERED_EXPOSED_CACHE_NOT_BUILT
formal output root absent
```

## Motivation

Gate 3C1F2 failed the complete-video AJ and magnitude gates despite strong action-level coordinate recovery. The exact 44-video failure diagnostic established:

- modified coordinates under GT visibility improve action-video AJ by `+0.2921` points with a positive CI;
- native visibility suppresses recovered coordinates;
- modified visibility applied to native coordinates reduces AJ by `-0.2195` points;
- failure actions recover 220 visible false negatives and create only two occluded false positives;
- `other` actions recover six visible false negatives but create 176 occluded false positives.

A grouped-OOF probe using only 19 low-dimensional trajectory/probability/action signals was insufficient. The best regularized logistic visibility model achieved AUC `0.6233`, AP `0.5662`, and pooled affected-frame AJ `0.09816`, only slightly above the frozen modified-visibility baseline `0.09245`, while retaining a `68.94%` GT-occluded false-positive rate. The low-dimensional route is therefore rejected as the primary redesign.

## Purpose

Build one deterministic exposed-development cache containing richer causal identity and memory-consistency evidence for the 44 sealed Gate 3C1F2 action videos. This gate qualifies feature extraction and exact pipeline identity only. It does not select a model or authorize a confirmation claim.

## Frozen population

```text
videos:             44 sealed Gate 3C1F2 action videos
actions:            89 sealed actions
frames per action:  15--23 inclusive, nine frames
frame rows:         801
```

The source-video membership is taken exactly from the committed Gate 3C1F2 replay. Every cache video must reproduce all sealed entry, candidate, shortlist, action, coordinate, and visibility digests before any new feature is stored.

## Frozen 66-D causal feature schema

All features for frame `t` use only frame `t` or earlier observations.

### Geometry and trajectory

- native and modified normalized coordinates;
- modified-minus-native coordinate displacement;
- native and modified speed and acceleration;
- distance from the frame-15 commit coordinate;
- image-border margin;
- normalized frame position.

### Native and modified tracker state

- visibility, confidence, and joint probability;
- native-to-modified probability deltas;
- frozen binary visibility states.

### DINOv3 identity

- modified and native query-descriptor cosine;
- modified-minus-native query cosine;
- modified and native previous-frame descriptor cosine;
- modified and native frame-15 commit-descriptor cosine;
- modified/native current-descriptor cosine;
- causal running mean and minimum of modified query cosine.

### Four-level CoTracker identity

For each of four frozen feature-pyramid levels:

- modified descriptor versus modified commit descriptor;
- native descriptor versus native commit descriptor;
- modified descriptor versus native commit descriptor;
- modified versus native current descriptor;
- modified previous-frame descriptor cosine;
- native previous-frame descriptor cosine.

### Sealed action evidence

- entry probability;
- native joint probability at commit;
- selected support probability;
- normalized predicted value;
- predicted harm probability;
- normalized selected slot;
- normalized output candidate index.

## Prohibited feature inputs

The model feature tensor may not contain:

- GT visibility or coordinates;
- native or modified coordinate error;
- threshold-hit labels;
- Gate 3C1F2 failure/ambiguous/other category;
- any frame after the prediction frame.

GT visibility, coordinate errors, threshold hits, and complete-video metrics are stored only as labels/evaluation tensors.

## Stored evaluation view

Each video sidecar also seals:

- complete native and modified coordinates;
- native and modified visibility/confidence probabilities and binary visibility;
- GT tracks/occlusion and query points;
- action point/frame identities;
- per-frame GT visibility, native/modified coordinate error, and modified threshold hits.

These tensors allow later nested group-OOF visibility policies to be evaluated under exact complete-video AJ, delta_avg, and OA without rerunning GPU tracking.

## Determinism and qualification

The formal cache build must produce:

```text
44 videos
89 actions
801 frame rows
66 feature dimensions
all sealed pipeline digest checks true
all sidecars reload with exact tensor hashes
one deterministic cache index
```

The source-0 smoke build has already verified one video, three actions, 27 frame rows, all 66 channels, tensor reload, and sealed digest parity. The smoke output is outside the formal cache root and is not a scientific result.

## Authorized next step

A completed exact cache may authorize a separately preregistered nested source-video OOF model-development gate. Model family, targets, threshold selection, and complete-video safety gates must be frozen in that later plan before formal model results are read.

Any eventual design selected on this exposed population requires a new raw-record-disjoint confirmation population. DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.
