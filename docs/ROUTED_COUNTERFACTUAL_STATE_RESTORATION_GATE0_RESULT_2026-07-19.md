# Route-D counterfactual tracker-state restoration Gate 0 result — 2026-07-19

## 1. Formal decision

Gate 0 passes:

```text
AUTHORIZE_SEPARATE_FIT_ONLY_LEARNED_STATE_RESTORER_PREREGISTRATION
```

This is a causal interface result, not a tracking-performance result. It proves
that the complete CoTracker3 commit-time state can be manipulated and restored
exactly, and that state outside coordinates has independent influence on the
next-window future.

## 2. Frozen protocol

```text
backbone:       CoTracker3 scaled-online true-streaming
fit sample:     Kubric train source index 0, video 1680
frames / points:24 / 64
window / step:  16 / 8
snapshot:       after the first window, online_ind = 8
future audit:   newly exposed frames 16--23
trainable params:0
GT performance: not read
```

Protocol SHA-256:

```text
f5dedb16f2e46421c55b88c711c1edfeca8d8210c97e18de552dd273f4f0e0d8
```

## 3. Actual state size

The snapshot contains:

```text
online_track_feat:
  four levels, each [1,1,64,128]

online_track_support:
  four levels, each [1,49,64,128]

online coordinate state:
  [1,16,64,2]

online visibility/confidence state:
  [1,16,64] each

plus:
  online_ind, predictor queries, and point count
```

The persistent correlation support is therefore a much larger and richer state
than the coordinate trajectory alone.

## 4. Future effect of the frozen corruption

All values compare finalized frames 16--23 with the clean continuation over 506
active post-query point-frame rows.

| Variant | Mean coordinate difference | Maximum difference | Rows above 1px | Mean visibility difference | Mean confidence difference |
|---|---:|---:|---:|---:|---:|
| Exact no-op restore | 0.0000 px | 0.0000 px | 0.00% | 0.0000 | 0.0000 |
| Composite corruption | 17.2915 px | 50.0199 px | 96.44% | 0.3059 | 0.2753 |
| Restore coordinates only | 1.5018 px | 51.7497 px | 23.32% | 0.1888 | 0.0839 |
| Restore complete state | 0.0000 px | 0.0000 px | 0.00% | 0.0000 | 0.0000 |

The composite corruption modifies fixed overlap coordinates, visibility and
confidence logits, and 50% of persistent support channels. Restoring coordinates
removes most average displacement, but a substantial tail remains because the
tracker still consumes corrupted correlation support and probability state.

## 5. Exactness gates

```text
snapshot round-trip state exact:        PASS
no-op restored future exact:            PASS
full-state restored future exact:       PASS
composite mean future difference >=2px: PASS
composite rows above 1px >=25%:          PASS
coordinate-only mean difference >=0.5px:PASS
coordinate-only rows above 1px >=10%:   PASS
full restoration strictly better:       PASS
independent process replay exact:        PASS
```

Primary and replay Torch archives have different byte hashes because of archive
serialization metadata, but every nested tensor and scalar is exact.

## 6. Scientific interpretation

The result establishes a model-level fact missing from prior Route-D branches:

> The online tracker future is controlled by a structured latent state, and
> coordinate repair alone is not sufficient to restore that future.

This supports a state-restoration research direction rather than another
candidate-ranking module. In particular, `online_track_support` is directly used
by the next-window correlation computation, while visibility and confidence are
part of the update-transformer input.

The result does **not** establish that a restorer is learnable. Exact full
restoration uses an available clean snapshot, which is unavailable at inference.
It also does not establish improvement beyond native tracking: a denoiser trained
only to reproduce an uncorrupted native state could at best recover native
behavior.

## 7. Required next gate

Before training a state-restoration network, construct a synthetic fit-only
**oracle state teacher** that is better than native on naturally failed tracks.
The next audit must answer:

```text
Can a causal state transplant built from GT only during training improve future
rollout over native continuation?

Does a full state transplant outperform an oracle coordinate-only transplant?

Can the teacher target be represented compactly enough for a learned restorer?
```

A positive answer would justify supervised imitation of a beneficial latent
state. A negative answer would close the route before expensive training.

MegaDepth may later regularize real-image appearance, but old MegaDepth two-frame
pretraining cannot be treated as this contribution: it neither restores tracker
state nor optimizes next-window rollout.

## 8. Reproducibility

Canonical summary:

```text
docs/generated/ROUTED_COUNTERFACTUAL_STATE_RESTORATION_GATE0_SUMMARY_2026-07-19.json
```

Canonical summary SHA-256:

```text
4f024d4c8885e9d2697ecfcb03fa076bcc0fccbb6bda4dd5a828344e26538b39
```

Primary report SHA-256:

```text
30d2fe01329d2153544fbac2634e1f22e58b05c1024c8fea8ac394a557d8d87c
```

Replay report SHA-256:

```text
ebd0d6cd97fa88912ec84932f9a8d3aa44e951a19c550cb97e32fdfe9f91f2d8
```

Locked-data flags remain false for model validation, calibration, final holdout,
DAVIS, and Kinetics. Official Kinetics 1,144 was not read or rerun.
