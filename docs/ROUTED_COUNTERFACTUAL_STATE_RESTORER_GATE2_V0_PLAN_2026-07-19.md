# Route-D counterfactual structured state restorer Gate 2 v0 plan — 2026-07-19

## 1. Motivation

Gate 0 established that CoTracker3 future rollout depends causally on complete
online state rather than coordinates alone. Gate 1 then showed that a GT-derived
fresh-query state reduces natural fit-video future error from `60.7129 px` to
`5.8543 px`, while coordinate-only and coordinate-probability transplantation
remain near `42--43 px`.

The next question is therefore not whether a better state exists. It is whether
a causal model can infer a useful fresh-like state without GT and without future
frames.

Gate 2 evaluates **Counterfactual Structured Re-extraction Restoration (CSRR)**.
The model predicts a commit-time correction location and a no-op decision from
native state plus frozen current-frame features. It does not directly generate a
large support tensor. Instead, it invokes the frozen CoTracker feature sampler at
the predicted location to reconstruct a coherent four-level track feature and
7 x 7 support memory.

## 2. Why this is distinct from closed historical routes

This is not another output-only candidate selector:

- the action modifies the state consumed by the next CoTracker window;
- full future rollout is the validation target;
- the support memory is reconstructed through the frozen tracker feature
  geometry rather than predicted as unrelated values;
- learned coordinate-only and coordinate-probability variants are mandatory
  controls;
- exact no-op is part of the model semantics.

It is also not a repetition of the old MegaDepth route. That route trained a
separate FSPT tracker with two-frame coordinate and occlusion losses. It did not
operate on CoTracker commit state, did not distill a fresh-query state, and did
not validate counterfactual future continuation.

## 3. Data boundary

The 64-video Kubric development manifest remains divided as follows:

```text
0--7:   architecture-design exposed by Gate 1; excluded from Gate 2 gradients
8--31:  Gate 2 gradient training
32--47: fit-internal checkpoint selection and formal Gate 2 validation
48--63: locked model-validation; unread until Gate 2 passes
```

The split is frozen before any Gate 2 cache or result is read.

Calibration, final holdout, DAVIS, and official Kinetics remain locked. The
completed official 1,144-video Kinetics evaluation is not rerun.

## 4. Causal input and teacher boundary

At inference the model may use only:

```text
frames 0--15;
native overlap coordinates, visibility, and confidence;
original query metadata;
native four-level track feature and support memory;
frozen CoTracker frame-15 feature pyramid.
```

It may not read:

```text
GT coordinates or visibility;
frames 16--23;
fresh-query teacher state;
future native or corrected error.
```

GT is used only to construct training targets, fixed failure/clean evaluation
rows, and fit-internal scores.

## 5. Fixed examples

A failure row must satisfy:

```text
original query frame < 8;
GT visible at frame 15;
GT visible in at least four frames of 16--23;
native future mean error >= 16 px.
```

A clean no-op row uses the same visibility rules but requires native future mean
error <= 4 px. Rows in `(4,16)` are excluded rather than assigned a soft label.

Training uses at most 16 failure and 16 clean rows per video. Validation uses at
most 8 of each per video. Ranking and no-backfill behavior are fixed in the
configuration.

## 6. Structured architecture

For each point and each of four feature levels, CSRR attention-pools the native
7 x 7 support memory, projects 128 channels to 32, and computes full-frame cosine
similarity against the frozen frame-15 feature map. Four score maps are resized
to a common 64 x 64 grid and fused by a small convolutional head. A fixed
spatial-softmax temperature converts the fused map to the predicted coordinate.

The model additionally predicts:

```text
apply versus exact no-op;
visibility residual;
confidence residual.
```

For the full-state variant, frozen CoTracker sampling re-extracts all four track
features and all four 7 x 7 support memories at the predicted coordinate. Direct
dense support prediction is forbidden.

The total trainable parameter count must remain at or below 250,000.

## 7. Training objectives

Training combines:

- failure heatmap focal loss;
- failure coordinate Smooth-L1 loss;
- teacher track-feature cosine loss;
- teacher support-memory cosine loss;
- teacher visibility/confidence regression;
- apply/no-op BCE;
- clean coordinate-displacement penalty;
- clean memory-change penalty.

The frozen CoTracker is never fine-tuned. Actual future rollout is evaluated at
every epoch and is used by the fixed fit-internal checkpoint score, but no future
frame is an inference input.

## 8. Required controls

Every selected checkpoint is evaluated as:

```text
native;
learned coordinate only;
learned coordinate + probability;
learned full structured state re-extraction.
```

This determines whether any gain comes from state repair rather than simply a
better commit coordinate.

## 9. Two formal tests

### 9.1 Forced failure action

GT defines the fixed natural-failure evaluation rows, but the model receives no
GT input. The correction is forced to isolate restoration quality.

A pass requires material improvement over native and both learned controls,
positive paired confidence intervals, broad point/video support, and lower
severe-error rate.

### 9.2 Learned gate on failure + clean union

The frozen `0.5` apply threshold is used on the balanced failure/clean union. A
pass requires:

```text
failure apply recall >= 50%;
clean false apply rate <= 5%;
clean harmful rate <= 2%;
positive union utility and error improvement.
```

Exact no-op must preserve native state and future tensors bit-for-bit.

## 10. Replay and stop rule

Training is run once with seed 17 and repeated in a fresh process. Checkpoint
tensors, selected epoch, validation outputs, and all aggregate metrics must be
exact.

Failure yields:

```text
STOP_LEARNED_STATE_RESTORER_BEFORE_MODEL_VALIDATION
```

No threshold, loss-weight, source-index, architecture, checkpoint, or gate sweep
is permitted after formal Gate 2 validation is observed.

A pass yields only:

```text
AUTHORIZE_ONE_FROZEN_MODEL_VALIDATION_STATE_RESTORER_AUDIT
```

It does not authorize calibration, final holdout, DAVIS, Kinetics, or a paper
claim.

## 11. Interface gate completed — 2026-07-19

The source-8 primary/replay interface audit passed.

```text
trainable parameters:                    19,685
re-extracted track-feature max error:    2.98e-8
re-extracted support max error:          2.98e-8
minimum re-extraction cosine:            0.99999994
maximum float16 cache error:             1.2204e-4
minimum float16 cache cosine:            0.99999988
zero-action state parity:                exact
independent replay:                      exact
```

Formal decision:

```text
ALLOW_GATE2_TEACHER_CACHE_BUILD
```

This authorizes only the sealed `8–31` training and `32–47` fit-internal
validation caches. Source indices `48–63` and all later data remain locked.

See:

```text
docs/ROUTED_COUNTERFACTUAL_STATE_RESTORER_INTERFACE_RESULT_2026-07-19.md
docs/generated/ROUTED_COUNTERFACTUAL_STATE_RESTORER_INTERFACE_SUMMARY_2026-07-19.json
```

## 12. Pre-cache schema correction — 2026-07-19

Before any Gate 2 teacher cache was built, an internal dimensional inconsistency
was found in the frozen configuration. The enumerated trajectory input contains
nine values per frame:

```text
normalized xy:                       2
normalized framewise delta:          2
visibility and confidence:           2
normalized original query xy:        2
distance from original query:        1
                                      -
total:                                9
```

The configuration omitted an explicit trajectory dimension while the module
default was mistakenly `6`. The interface smoke used synthetic placeholder
trajectory tensors and therefore did not exercise this field; no learned result,
cache, checkpoint, or validation metric existed when the inconsistency was
discovered.

The protocol now explicitly records `model.input.trajectory_dim: 9`, the module
default is corrected to `9`, and the exact feature builder is part of the tested
interface. All other architecture, data partitions,
losses, thresholds, and gates remain unchanged. The interface primary/replay must
be rerun under the corrected config before cache construction.


### Corrected interface replay result

The interface was rerun after the nine-dimensional schema correction.

```text
corrected config SHA256:                 31a62db62d609e28acb9a7ac8cc5740866134d92ee3564e99a0bd7f847a33f6b
trainable parameters:                    19,685
maximum teacher re-extraction error:     2.98e-8
maximum float16 round-trip error:        1.2204e-4
zero-action parity:                      exact
independent replay:                      exact
canonical summary SHA256:                84287c8ad739df418d8a2158ff36a0e2e69a4c87b9545b062c2e6eda4dfa9bfb
```

The decision remains `ALLOW_GATE2_TEACHER_CACHE_BUILD`.


## 13. Pre-cache exact-state storage clarification — 2026-07-19

Before any teacher cache was built, the cache dtype contract was clarified to
preserve both training efficiency and exact causal replay.

```text
model inputs and teacher views:          float16
complete native rollout state:           float32
video frames:                            uint8
```

The float16 view is used by CSRR training and remains subject to the frozen
`1e-3 / 0.99999` quantization gates. A separate float32 copy of the complete
native commit state is required for exact native continuation, zero-action
parity, and C/P/F future-rollout comparisons. Quantizing the only native state
copy would weaken the already frozen exact-no-op gate.

No cache, checkpoint, training epoch, or validation metric existed when this
clarification was made. Model architecture, source partitions, losses, thresholds,
and scientific gates are unchanged. The interface primary/replay must be rebuilt
under the clarified config hash before cache construction.
