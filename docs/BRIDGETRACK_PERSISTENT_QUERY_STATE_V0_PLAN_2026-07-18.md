# BridgeTrack Persistent Query-State v0 — Feasibility Plan

Date: 2026-07-18  
Status: implementation started; heavy TAPNext++ oracle execution blocked only by the still-running exact Route-D replay.  
Target: determine whether explicit persistent point-identity state can repair recurrent TAP degradation after long occlusion without becoming another output-level threshold router.

## 1. Research correction

The previous Safe Re-detection v0 is formally negative:

```text
PointOdyssey model-validation: 160 events
native AJ_RD:                 0.211934
learned AJ_RD:                0.211934
direct gain:                  0
best epoch:                   -1
candidate joint oracle gain: +0.179774
decision: STOP_SAFE_REDETECTION_BEFORE_LOCKED_DATA
```

Its failure is structural, not an invitation to tune more thresholds:

- dense proposal BCE reduces the zero-step candidate oracle;
- multiple independent hard gates converge to permanent abstention;
- two-frame confirmation incorrectly treats dynamic top-K rank as identity;
- frame-local losses do not optimize the deployed temporal action;
- CoTracker3 does not maintain an evolving query latent state after initialization, so “latent state reconstruction” on that architecture would mostly reduce to coordinate/visibility history writeback.

Therefore the old `proposal + comparator + abstention + bounded writeback` route is no longer the main method.

## 2. Fixed-patch constellation audit: early rejection

A deliberately small non-learned audit was implemented before building a tracklet model:

```text
query-frame reference:
  centre feature
  fixed ±4 / ±8 / ±16 pixel local samples
candidate evidence:
  centre correlation
  corresponding-offset cosine
  local self-similarity graph
```

The first 2x2 smoke appeared positive versus centre correlation, but both validation events had perfect native utility. The audit was corrected to measure only rows where a non-native candidate is truly better than native.

A duration-bucket-only smoke was then found to overfit one scene. After correcting sampling to rotate across both duration bucket and scene, the 10-fit / 10-model-validation smoke gave:

```text
fit scenes:                          ani13_new_f, ani18_new_f
model-validation scenes:             ani10_new_f, ani16_new_
selected fit-only weights:            direct=0.5, structure=0
recoverable rows (model-validation): 29
recoverable utility gain vs centre:  +0.010345
beneficial non-native recall gain:   -0.034483
paired event CI:                     [-0.017143, +0.048214]
harmful intervention rate:           0.230769
long bucket d16_63 gain:             -0.054545
full feasibility gate:               fail
```

The self-similarity term is consistently harmful, and fixed image-coordinate offsets do not preserve exact point identity across view and deformation changes.

Decision:

```text
STOP_FIXED_PATCH_CONSTELLATION
```

Do not rescue it by changing radii, adding weights, or training another patch reranker.

## 3. Historical branches that must not be repeated

Repository evidence already excludes several superficially similar routes:

- escort-neighbour geometry reranking failed to improve the candidate selector;
- learned neighbour-deviation reliability was unstable as a main method, although trajectory-neighbour inconsistency is a strong error-detection signal;
- sampled single-state temporal policies failed despite past-oracle state headroom;
- shared-state beam, history beam, and independent-state branching did not produce a deployable path;
- simple CoTracker3 coordinate/visibility state writeback propagated but did not materially beat output-level correction at full scale.

The new method must therefore be neither “more neighbours”, “another beam”, nor “more writeback fields”.

## 4. New scientific hypothesis

TAPNext/TAPNext++ are recurrent point trackers whose per-layer state contains a recurrent cache for every image token and point-query token. Long sequences can degrade the information stored about the original point identity.

Hypothesis:

> A point query needs an explicit persistent identity state that is separated from transient image state. Restoring only the query-token recurrent cache after an occlusion can recover future point predictions while preserving the current image-token state and all unrelated queries.

This changes the problem from post-hoc candidate routing to **causal point-memory repair inside a recurrent tracker**.

## 5. Persistent Query-State Bridge

For every recurrent layer `l`, TAPNext++ exposes:

```text
RG-LRU state:  [B * (P + Q), E]
Conv1D state:  [B * (P + Q), W, E]
```

where `P` is the number of image patch tokens and `Q` the number of point queries. Query tokens are the trailing `Q` tokens for every batch element.

The v0 interface implements:

```text
extract_persistent_query_state(state)
inject_persistent_query_state(current, persistent, query_mask, blend)
query_state_distance(left, right)
```

Hard invariants:

1. image-token recurrent state is untouched;
2. unselected queries are untouched;
3. current causal `step` is untouched;
4. current query metadata is untouched;
5. `blend=0` is exactly equal to the current recurrent cache;
6. `blend=1` replaces only selected point-query state;
7. no GT quantity is an inference input.

Initial unit result:

```text
9 tests passed
```

The tests cover token-layout validation, exact zero blend, selective-query replacement, preservation of image and unrelated-query tokens, partial blending, and state-distance shape.

## 6. Gate A — static corruption state oracle

Use one PointOdyssey **fit** image as a repeated static sequence:

```text
8 visible frames
32 deterministic mean-RGB blackout frames
16 visible rollout frames
```

The teacher observes the original image throughout. The student observes the blackout interval. At the end of the gap, branch from the same causal step:

```text
native_student
pre_gap_query
teacher_query_oracle
teacher_full_oracle
```

Definitions:

- `pre_gap_query`: restore only the query-token state saved immediately before corruption;
- `teacher_query_oracle`: copy only the teacher query-token state at the same causal step;
- `teacher_full_oracle`: copy all teacher recurrent tokens; this is an oracle upper bound.

Primary gates:

```text
teacher-query mean-error improvement >= 2.0 pixels at 256 raster
pre-gap query snapshot improves over native student
query-only state retains >= 50% of full-state oracle improvement
```

Interpretation:

- If teacher full state helps but query-only state does not, identity information is too distributed across image tokens for a query-only bridge.
- If neither helps, recurrent state is not the actionable bottleneck under the controlled corruption.
- If pre-gap state helps, a non-oracle persistent memory route is immediately plausible.
- If only same-time teacher state helps, a learned state reconstruction problem remains plausible but must pass Gate B.

No DAVIS, PointOdyssey holdout/test, or Kinetics is read.

## 7. Gate B — synthetic visible-interval masking on fit scenes

Only after Gate A passes:

1. select naturally visible fit-scene intervals;
2. run the teacher on the original video;
3. corrupt the target region or frame stream for the student;
4. compare native, pre-gap state, teacher query state, and full state;
5. score the future 16/32-frame rollout, not only the first recovered frame.

Required evidence:

```text
teacher-query oracle improves future rollout on >= 75% of fit scenes
sequence-level paired CI lower > 0
query-only state retains >= 50% of full-state oracle improvement
benefit persists for gap lengths 16, 64, 256
```

If this fails, close the Persistent Query-State route before training.

## 8. Gate C — learned state reconstruction

Only after Gate B passes, learn a small bridge:

```text
inputs:
  persistent pre-gap query state
  current degraded query state
  post-reappearance image/query token evidence
  causal state-distance summaries

outputs:
  bounded low-rank residual for query RG-LRU state
  bounded low-rank residual for query Conv1D state
  reconstruction confidence
```

The model must not predict coordinates directly. The frozen TAPNext++ prediction heads and future recurrent rollout determine whether the repaired identity state is useful.

Training objective:

```text
query-state distillation to same-time uncorrupted teacher
+ future coordinate/visibility rollout loss
+ identity-preservation loss on uncorrupted sequences
+ norm-bounded residual penalty
```

Safety contract:

```text
zero residual = exact native tracker
low confidence = no state repair
only selected query tokens may change
current frame cannot be retroactively modified
```

## 9. Novelty boundary

Not novel by itself:

- long-sequence training;
- global image matching;
- post-hoc re-detection;
- confidence thresholding;
- generic tracklet ReID;
- copying a recurrent state.

Potential method contribution:

1. explicitly factor persistent point identity from transient image state in a recurrent TAP model;
2. diagnose long-occlusion failure through query-token state corruption rather than output error alone;
3. learn bounded query-state repair using counterfactual uncorrupted teachers and future rollout supervision;
4. retain strict per-query isolation and exact native equivalence when repair is disabled.

This is materially different from ReTracker’s global image-matching decoder and TAPNext++’s long-sequence/roll-augmentation training: it targets **test-time point-memory repair** within a frozen recurrent tracker.

## 10. Locked-data and reporting rules

```text
PointOdyssey fit:             allowed for feasibility and training
PointOdyssey model-validation: allowed only after fit-only design freeze
PointOdyssey internal holdout: locked
PointOdyssey test:             locked
DAVIS:                         historical exposure; no method tuning
Kinetics 1,144:                permanently frozen; never rerun
```

Oracle state replacement must always be labelled oracle. It is not a learned result and cannot define the paper headline.

## 13. Gate A result and frozen dynamic redesign

The preregistered static Gate A produced a full-state oracle improvement of only
`+0.7742 px`, so the original `>=2 px` threshold was unreachable in that
controlled sequence. This is recorded as `STATIC_GATE_CEILING_TOO_SMALL`, not a
pass. The threshold is not relaxed.

Before running a dynamic redesign, freeze the following fit-only protocol:

- real 56-frame moving PointOdyssey train clip;
- 8 visible frames, 32 deterministic full-frame corruption frames, 16 visible rollout frames;
- candidate points must be visible, valid, finite, and interior for all 56 frames;
- select the 90th-percentile point by displacement across the synthetic gap;
- compare native, pre-gap query state, same-time teacher query state, and full teacher state;
- GT is used only for fit stress selection and scoring;
- no holdout, test, DAVIS method evaluation, or Kinetics data may be read.

Frozen dynamic gates:

1. teacher query-state mean-error improvement `>=2 px` at 256 raster;
2. pre-gap persistent query state improves over native;
3. query-only state retains at least 50% of full-state oracle improvement;
4. teacher query-state improves at least 75% of the 16 future frames.

Only a pass allows expansion to multiple fit scenes. It still does not allow
training or locked-data access.

## 14. Frozen query-state factorization oracle

Dynamic Gate A2 shows that same-time teacher query state is causal and nearly
matches the full teacher state, while stale pre-gap state is harmful. Therefore
no persistence-only method is allowed. Before designing a learned reconstructor,
run a fit-only causal factorization audit on the same frozen clip and point.

Variants are frozen before execution:

- all RG-LRU state only;
- all Conv1D state only;
- all state in layers 0–3, 4–7, or 8–11;
- all state in layers 0–5 or 6–11;
- RG-LRU only in layers 6–11;
- Conv1D only in layers 6–11.

Factorization gates:

1. a variant replacing at most 50% of query-state elements retains at least 80%
   of the full teacher-query improvement;
2. one single cache component retains at least 70% of that improvement;
3. the best compact variant improves at least 75% of future frames.

A pass allows only multi-fit-scene factorization verification. It does not allow
training or locked-data access.

## 15. Frozen multi-fit-scene mid-layer verification

The original component-factorization gate remains failed because neither
RG-LRU-only nor Conv1D-only met the frozen single-component requirement.  The
redesigned hypothesis is narrower: layers 4--7 form a joint recurrent repair
unit and must not be decomposed into a single cache component.

Before any model execution, freeze the verification manifest:

- source population: the 24 existing PointOdyssey fit scenes only;
- discovery scene `ani13_new_f` excluded from the primary aggregate;
- clip start 0, protocol 8 visible / 32 deterministic corruption / 16 rollout;
- margin 64 px for all 56 frames;
- at least 64 eligible, finite, visible points per scene;
- within each scene select the 90th motion-quantile point;
- rank qualified scenes by ascending
  `sha256("17018|bridgetrack-mid4-multiscene-v0|scene")`;
- use the first six scenes after excluding the discovery scene;
- model is loaded once; all scene results, including failures, must be retained.

Frozen primary comparison:

```text
native student
same-time teacher layers 4--7, RG-LRU + Conv1D
same-time teacher full query state
same-time teacher full recurrent state
```

Frozen gates over the six non-discovery fit scenes:

1. mid-layer joint state improves future mean error in at least 5/6 scenes;
2. median mid-layer gain over native is at least 2.0 px at 256 raster;
3. scene-bootstrap 95% CI lower bound for mean gain is positive;
4. median retained full-query gain is at least 80%;
5. median improved-frame fraction is at least 75%;
6. no scene has a harmful regression worse than 1.0 px.

Passing permits only the design of a fit-only mid-layer state reconstructor.  It
does not permit model-validation, holdout, test, DAVIS method evaluation, or
Kinetics access.  Failing closes the Query-State Bridge route.

## 16. Frozen low-rank Query-State repair-subspace oracle

The six-scene audit closes fixed layers 4--7, but same-time full query-state
replacement is positive in all six scenes with a median gain of 3.4759 px.
Therefore the next hypothesis is not another fixed layer subset.  It is that
long-occlusion state corruption lies in a compact cross-layer repair subspace.

Data separation is frozen before execution:

- the discovery scene and all six scenes already inspected by BridgeTrack form
  the basis-development set;
- every remaining qualified fit scene forms the internal verification set;
- no scene may move from verification to basis after results are observed;
- each of seven basis scenes contributes the first three eligible starts from
  `{0,64,128,192,256}`;
- each of ten verification scenes contributes start frame 0;
- all clips use 8 visible / 32 deterministic corruption / 16 rollout frames,
  64 px margin, at least 64 eligible points, and the 90th motion quantile.

State representation is frozen:

1. flatten teacher-minus-student query-state into 24 blocks: RG-LRU and Conv1D
   for each of 12 layers;
2. divide each block by its scalar RMS estimated only from basis clips;
3. subtract the basis mean and compute SVD/PCA;
4. evaluate oracle projections at ranks `{0,1,2,4,8}`;
5. rank 8 is the sole primary variant; verification data cannot select rank;
6. oracle coefficients use the hidden teacher delta and test compressibility
   only, not deployable predictability.

Frozen rank-8 verification gates over all ten unseen fit scenes:

1. all ten scenes complete;
2. full-query Oracle is positive in at least 8/10 scenes;
3. rank-8 projection is positive in at least 8/10 scenes;
4. rank-8 median future-error gain is at least 2.0 px;
5. scene-bootstrap 95% CI lower bound is positive;
6. median retained full-query gain is at least 75%;
7. median improved-frame fraction is at least 75%;
8. no scene regresses by more than 1.0 px.

Passing permits only a fit-only coefficient-prediction feasibility model.  It
does not permit PointOdyssey model-validation, holdout, test, DAVIS method
evaluation, or Kinetics access.  Failing closes BridgeTrack Query-State repair.

## 17. Final closure

The frozen low-rank verification failed at every reported rank. Rank-8 was
positive in only 1/10 unseen fit scenes and had a median gain of -6.7984 px.
Same-time full query-state was also harmful in two scenes. Therefore the
Query-State Bridge route is closed in full.

No further changes to rank, normalization, layer subsets, basis construction,
or state blending are allowed under this branch. No coefficient predictor may
be trained. PointOdyssey model-validation and all locked data remain unread.

A future project may study end-to-end identity/transient-state factorization
inside a newly trained recurrent tracker, but it must be treated as a new model
and a new preregistered branch rather than a continuation of frozen-state repair.
