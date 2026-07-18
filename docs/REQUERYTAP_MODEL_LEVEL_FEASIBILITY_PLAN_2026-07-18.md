# ReQueryTAP Model-Level Feasibility Plan — 2026-07-18

## Status

New project after formal closure of Safe Re-detection v0 and BridgeTrack frozen
Query-State repair.  This branch may not reuse post-hoc state injection as its
method.

## Hypothesis

Long-occlusion failure should be represented as a **track-token lifecycle**
problem rather than candidate routing or hidden-state repair:

```text
persistent point-identity token
+ ephemeral recurrent track token
+ learned retire / global rebind / respawn operation
```

The identity token does not directly output coordinates.  When the active track
token becomes unreliable after an occlusion, the model globally binds the
identity token to current image tokens and instantiates a fresh track token at
the recovered location.  The old transient state is retired rather than
repaired or blended.

The eventual model must be trained end to end.  A frozen tracker may be used
only for causal oracle/precheck experiments.

## Gate 0 — PyTorch training feasibility

Run a real four-frame PointOdyssey fit clip through the official TAPNext++ torch
model in training mode.  Trainable parameters are the last recurrent block,
encoder norm, coordinate head, visibility head, and point-query token.

Required:

1. forward, backward, and one AdamW step complete on the available 24 GB GPU;
2. finite loss and gradients;
3. nonzero gradient in the recurrent block and coordinate/visibility heads;
4. at least one trainable parameter changes;
5. peak allocated GPU memory below 22 GB.

This is an engineering gate only and creates no performance claim.

## Gate A — causal fresh-query oracle

Use every qualified PointOdyssey fit scene from the frozen BridgeTrack metadata,
with start frame 0 and the existing 8 visible / 32 deterministic corruption /
16 rollout protocol.  For each scene select the fixed 90th motion-quantile point
among at least 64 full-interval valid interior points.

At the first post-gap frame compare:

```text
native_student
fresh_native_coordinate_query
fresh_teacher_coordinate_query
fresh_GT_coordinate_query
```

All fresh branches create a new TAPNext++ recurrent state at the re-entry frame
and process only that frame and future frames.  They do not inject or copy old
hidden state.

Interpretation:

- fresh native coordinate isolates the value of state reset;
- fresh teacher coordinate tests re-localization without exact GT coordinates;
- fresh GT coordinate is the query-respawn upper bound.

Frozen primary gates for fresh GT over all qualified fit scenes:

1. every scene completes;
2. positive future-error gain in at least 80% of scenes;
3. median gain at least 2.0 px at the 256 raster;
4. scene-bootstrap 95% CI lower bound positive;
5. median improved-frame fraction at least 75%;
6. no scene regression worse than 1.0 px.

Passing Gate 0 and Gate A permits only architecture implementation and fit-only
training.  PointOdyssey model-validation, internal holdout, test, DAVIS method
evaluation, and Kinetics 1,144 remain unavailable.

## Proposed model after feasibility passes

### Persistent Identity Token

A query-conditioned token optimized for identity consistency, not coordinates.
It receives selective writes only when local identity evidence is reliable.

### Ephemeral Track Token

A recurrent token optimized for current coordinate, visibility, and local motion.
It may be retired after sustained uncertainty.

### End-to-end Rebinding and Respawn

The identity token performs global cross-attention to image tokens.  A rebind
controller predicts a location and initializes a fresh ephemeral track token.
The operation is part of model recurrence, not an output replacement.

### Counterfactual paired-stream training

The clean and synthetically occluded versions of one sequence share physical
point identity but have different transient histories.  Training uses:

```text
identity consistency across streams
+ rebind location distribution loss
+ fresh-token future rollout coordinate/visibility loss
+ same-object nearby-point hard negatives
+ lifecycle sparsity and false-respawn penalties
```

A zero-respawn ablation must reproduce the original recurrent path, but exact
native equivalence is not the central contribution because the final method is
a newly trained model.

### Gate A scoring clarification frozen before execution

The fresh query is initialized on the first post-gap frame.  That frame is
recorded diagnostically but excluded from the primary metric because a GT query
would otherwise receive a trivial query-frame advantage.  All frozen Gate A
statistics use only the following 15 future frames.

The scene population is the complete set of 17 PointOdyssey fit scenes already
qualified by the pre-model metadata scan.  No scene may be omitted after model
execution.

## Gate B — lifecycle architecture integrity

Before any fit-only performance training, implement a strict single-query
ReQueryTAP wrapper whose TAPNext++ backbone remains a trainable submodule.
Single-query execution is intentional and preserves TAP-Vid query independence.

Required architecture behavior:

1. identity memory is sampled at the original query frame and remains separate
   from the active recurrent track state;
2. a global differentiable locator maps identity memory and current image tokens
   to a rebind coordinate and respawn logit;
3. native and fresh track-token paths are both exposed during training;
4. forced respawn creates a completely new TAPNext state at the predicted or
   supplied coordinate; it never copies, blends, or injects old hidden state;
5. respawn-disabled output is exactly the native TAPNext++ output;
6. the fresh path is differentiable through the predicted coordinate.

Frozen Gate B tests on a real fit clip:

```text
native-off coordinate/logit max difference = 0
forced-GT fresh path vs standalone fresh TAPNext++ max difference <= 1e-6
finite nonzero locator gradients through fresh-token rollout
peak allocated GPU memory < 22 GiB
single-query contract enforced
```

Passing Gate B permits fit-only training of the locator and respawn controller.
It still does not permit PointOdyssey model-validation or locked data.

## Gate C1 — fit-only exact-point locator training

The 17 qualified fit scenes are split before cache construction by ascending
`sha256("17018|requerytap-locator-v0|scene")`: 12 train scenes and 5 dev scenes.
No scene may move after results are observed.

Features are frozen TAPNext++ patch embeddings from `lin_proj + image_pos_emb`.
The query identity comes from frame 0.  Target images are frames 40--55.  Each
scene contributes a deterministic maximum of 256 point identities that are
visible at query time and visible in at least one target frame.

Train only the 263k-parameter global locator for 12 epochs with 4096 sampled
pairs per epoch.  The raw unprojected cosine patch match is the baseline.

Frozen C1 gates over all five dev scenes:

1. every dev scene completes;
2. aggregate median coordinate error <= 12 px;
3. aggregate hit@16 >= 60%;
4. learned median error improves over raw cosine in 5/5 scenes;
5. scene-bootstrap 95% CI lower bound for error reduction is positive;
6. no scene error reduction is worse than -2 px.

C1 passing does not establish a tracking improvement.  It only allows Gate C2,
which will re-instantiate fresh query tokens using predicted coordinates and
score the following future frames.
