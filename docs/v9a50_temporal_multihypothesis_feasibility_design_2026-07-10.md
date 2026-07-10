# V9-A5.0 Sampled-Causal Temporal State Feasibility Audit Design

Date: 2026-07-10

## 1. Motivation

V9-A4.5 closed the fixed-topK single-frame reranking-adaptation route. Its conservative residual student was correctly trained and fully audited, but failed the predeclared heldout synthetic gate:

```text
teacher mean error 14.7458
student mean error 15.0387
better / worse / equal = 35 / 70 / 1837
safe16 1563 -> 1544
```

V9-A3.0 nevertheless proved that the C1 top-K set contains substantial coordinate oracle headroom. V9-A5.0 therefore asks a new question:

```text
Does causal temporal state contain useful evidence for selecting among the frozen C1 candidates?
```

This is a deterministic feasibility audit. It is not training, not threshold tuning, and not a DAVIS experiment.

## 2. Scope and claim boundary

```text
Dataset: PointOdyssey canonical balanced sampled pool
Rows: 4,878
Tracks: 279 (clip_id, query_idx) sampled trajectories
Clips: 9
Sequences: ani / animal3 / r4_new_f
Training: none
DAVIS read: none
Tuned weights: none
```

The pool contains selected hard/easy observations rather than every online frame. Consequently:

```text
The self-state policies are sampled-causal, not yet full-stream deployable policies.
```

A positive result can only justify a subsequent deterministic full-stream audit.

## 3. Reproducibility and inputs

Input manifest:

```text
docs/v9a50_input_manifest_2026-07-10.json
```

The audit is fixed to:

```text
branch: v9a45-conservative-residual-20260710
HEAD: 8fe78db05fa23a229efb92329e8fe96a247b7989
```

The manifest hashes:

```text
PointOdyssey hypothesis pool
PointOdyssey C1 pre/post-fusion latent cache
TrackOn2 checkpoint
three PointOdyssey annotations
nine clip-start RGB frames used for exact image dimensions
```

Tracked worktree changes are forbidden at execution time.

## 4. Candidate/latent integrity gate

Before simulation, reconstruct the frozen TrackOn2 C1 scores directly from:

```text
c1_topk_latent
checkpoint reranking_head.score_layer
```

Compare against the pool's `exact_rerank_s` for all 4,878 x 16 candidates.

Mandatory gate:

```text
row keys identical
candidate count = 16
score top1 match rate = 1.0
score max_abs <= 0.005
all coordinates, errors, scores, and latents finite
prefusion latent norm > 1e-8
```

The independently run preflight audit gave:

```text
top1 match rate = 1.0
mismatch rows = 0
score max_abs = 0.001544
```

The small numeric difference is caused by the fp16 latent cache and does not change candidate ordering.

## 5. GT reconstruction

Reconstruct deterministic PointOdyssey query identities from the original annotations using the same stable seed and 32-query selection as the pool/latent exporters.

Mandatory gate:

```text
recomputed 34-action error_px max_abs versus frozen pool < 1e-4
```

## 6. Temporal observation model

Track key:

```text
(clip_id, query_idx)
```

Rows are processed in increasing `frame_tau`.

Temporal evidence is enabled only when the gap from the previous sampled observation is at most eight frames:

```text
gap <= 8: temporal evidence allowed
gap > 8: fall back to teacher and reset all temporal histories
```

The explicit reset prevents a stale pre-gap state from entering the next two-point velocity estimate.

## 7. Candidate state

For each of the 16 frozen C1 candidates:

```text
coordinate: normalized yx
teacher emission: exact_rerank_s
identity descriptor: normalized 512D c1_prefusion_latent
GT error: frozen/reproduced C1 error_px
```

Teacher rank is the stable descending ordinal rank of `exact_rerank_s`.

## 8. Temporal evidence

### 8.1 Motion

With one previous state:

```text
predicted coordinate = previous selected coordinate
```

With two previous states:

```text
velocity = (last_coord - previous_coord) / (last_t - previous_t)
predicted_coord = last_coord + velocity * (current_t - last_t)
```

Candidates receive ascending ordinal ranks by distance to the predicted coordinate.

### 8.2 Identity continuity

Candidates receive descending ordinal ranks by cosine similarity to the previous selected candidate's normalized 512D pre-fusion latent.

### 8.3 Borda fusion

All component ranks are integers in `[0,15]`, lower is better:

```text
teacher + motion
teacher + latent
teacher + motion + latent
```

No learned or manually tuned weights are used.

Tie order:

```text
total Borda rank
teacher rank
candidate index
```

## 9. Predeclared policies

### 9.1 Sampled-causal self-state policies

```text
teacher
causal_teacher_motion
causal_teacher_latent
causal_teacher_motion_latent
```

Each policy updates its own state using its previous sampled selections. These policies use no GT.

### 9.2 Causal diagnostic upper bounds

```text
past_oracle_motion
past_oracle_latent
past_oracle_motion_latent
past_gt_motion
past_gt_motion_plus_teacher
frame_oracle
```

`past_oracle_*` uses the oracle candidate only at earlier sampled observations, never the current or future label.

`past_gt_motion` uses only earlier GT coordinates. It is diagnostic and not deployable.

## 10. Metrics

Report globally, by sequence, hard/easy subset, gap bin, and temporal-eligible subset:

```text
mean / median error
safe4 / safe8 / safe16
oracle regret
exact-best rate
within-1px-best rate
better / worse / equal versus teacher
mean / median difference versus teacher
```

Gap bins:

```text
1
2-4
5-8
>8
```

## 11. Paired uncertainty

Use two predeclared cluster bootstraps, each with 100,000 resamples:

```text
query-track bootstrap: cluster = (clip_id, query_idx), seed 20260710
clip-block bootstrap: cluster = clip_id, seed 20260711
```

The track bootstrap is diagnostic. The more conservative nine-clip block interval is used for the sampled-causal pass gate.

## 12. Sampled-causal pass gate

At least one self-state policy must satisfy all:

```text
mean error lower than teacher on ani
mean error lower than teacher on animal3
mean error lower than teacher on r4_new_f
safe16 not lower on every sequence
global better rows > worse rows
clip-block mean-difference 95% CI upper bound < 0
```

If a policy passes:

```text
Proceed only to a full-stream deterministic causal audit that updates state on every frame.
Do not train a learned belief head yet.
Do not evaluate DAVIS yet.
```

## 13. Oracle-state headroom gate

If sampled-causal policies fail, temporal headroom is retained only when at least one past-oracle or past-GT diagnostic satisfies:

```text
mean error improvement >= 0.25 px on each of the three sequences
global better rows > worse rows
```

If this gate passes:

```text
State-error propagation is the likely blocker.
Proceed only to a full-stream multi-hypothesis/beam-state audit.
Do not train another single-state reranker.
```

If neither gate passes:

```text
Stop temporal reranking and move to candidate-generation/correlation supervision.
```

## 14. Outputs

```text
scripts/v9a50_temporal_multihypothesis_feasibility.py
outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility.json
outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility_selections.npz
docs/v9a50_temporal_multihypothesis_feasibility_result_2026-07-10.md
```
