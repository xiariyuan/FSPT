# V9-A5.1b Candidate-Conditioned Downstream Refinement Audit Design

Date: 2026-07-10

## 1. Motivation

V9-A5.1a established a valid negative baseline for a fixed-K16 shared-state beam that directly outputs raw C1 patch centers.

That baseline mixes three distinct questions:

```text
candidate availability
candidate-conditioned downstream refinement
history transition / pruning / readout
```

V9-A5C.0 subsequently showed large fused K16-to-K64 recall headroom on hard rows. Before constructing another beam, V9-A5.1b isolates the second question:

```text
Can one frozen C1 candidate hypothesis drive a useful frozen C2 + prediction-head refined final coordinate?
```

No training and no DAVIS data are allowed.

## 2. Repository isolation

Run only in:

```text
worktree: /gemini/code/FSPT_v9a51b_clean
branch: v9a51b-candidate-conditioned-refinement-20260710
base/freeze commit: d8e5ca9
```

The exact full HEAD is recorded in the input manifest at execution time.

TrackOn2 Python code and config must be imported from this clean worktree. Immutable checkpoint, DINOv3 weights, PointOdyssey annotations and RGB frames are read from `/gemini/code/FSPT` after SHA256 verification.

## 3. Dataset and full stream

```text
PointOdyssey sequences: ani / animal3 / r4_new_f
clip starts: 0 / 256 / 512
clip length: 96 frames
queries per clip: 32 deterministic identities initialized at frame 0
query-frame observations: 9 * 96 * 32 = 27,648
metric rows: frame_tau > 0
```

Official TrackOn2 memory is updated on every frame using the original official `q_new`. Candidate-conditioned branches are diagnostic readouts and never alter official query or memory state.

## 4. Official and extended candidate path

For every query and frame:

```text
q_pre = official pre-rerank query
c1 = official fused C1 correlation map
fused top64 = torch.topk(c1, K=64)
```

Run the frozen reranking local decoder and fusion layer for all 64 candidates:

```text
z_i = fusion_layer([local_decoder(q_pre, candidate_i), q_pre])
score_i = score_layer(z_i)
certainty_i = certainty_layer(z_i)
```

`z_i` is the frozen 256D post-fusion candidate descriptor.

## 5. Candidate-conditioned downstream refinement

For each candidate descriptor `z_i` independently:

```text
q_base_i = q_pre
q_conditioned_i = reranking_head.fusion(q_base_i, z_i, z_i)
q_conditioned_i = final_projection_layer([q_conditioned_i, q_base_i])
q2_i = projection2(q_conditioned_i)
c2_i = multiscale_correlation(q2_i, frame features)
p_patch_i = argmax(c2_i)
offset_i = prediction_head(q2_i, frame features, p_patch_i)
refined_position_i = p_patch_i + final offset_i
```

The `N * K` candidates are vectorized as independent query slots. All modules are frozen and kept in evaluation mode.

This is a diagnostic singleton-conditioning path. The original model was trained with fusion over all K candidates, so V9-A5.1b does not claim architectural equivalence to the official forward.

## 6. Candidate budgets and native risk

Predeclared candidate budgets:

```text
K16
K64
native dynamic K16/K64
```

Native risk uses only official final-head outputs:

```text
visibility_conf = sigmoid(v_logit)
uncertainty_conf = sigmoid(u_logit)

risk_visibility = visibility_conf < 0.8
risk_uncertainty = uncertainty_conf >= 0.5
risk = risk_visibility OR risk_uncertainty
```

The visibility threshold `0.8` is the clean TrackOn2 config value. The uncertainty threshold `0.5` is the natural binary-logit threshold. No percentile, fitted calibration or result-dependent threshold is allowed.

Dynamic budget:

```text
non-risk: K16
risk: K64
```

The audit computes K64 on every row to compare all readouts. Conceptual deployment compute is reported separately as `16 + 48 * risk_rate` candidates per row.

## 7. Predeclared readouts

### 7.1 Official baseline

```text
official_final
```

### 7.2 Raw C1 diagnostics

```text
raw_score_top1_K16
raw_score_top1_K64
raw_oracle_K16
raw_oracle_K64
raw_oracle_dynamicK
```

### 7.3 Candidate-conditioned refined diagnostics

```text
refined_score_top1_K16
refined_score_top1_K64
refined_oracle_K16
refined_oracle_K64
refined_oracle_dynamicK
```

### 7.4 System-level hybrid readouts

Deployable deterministic diagnostic:

```text
hybrid_refined_score_top1 =
    official_final on non-risk rows
    refined score-top1 using K64 on risk rows
```

GT-only refinement viability upper bound:

```text
hybrid_refined_oracle =
    official_final on non-risk rows
    min(official_final, refined_oracle_K64) on risk rows
```

GT is used only after candidate generation/refinement to compute oracle readouts. It never changes risk, candidate budget, candidate descriptors, q2, C2, offsets or future state.

The primary V9-A5.1b gate uses `hybrid_refined_oracle` relative to `official_final`. The deterministic hybrid is reported but is not allowed to replace the predeclared gate.

## 8. Evaluation subsets

Metrics exclude frame 0 and use GT-visible and valid rows unless stated otherwise.

Report:

```text
all visible
risk visible
non-risk visible
GT-invisible activation counts
hard official rows: official_final error > 4 px
refinement opportunity:
  official_final error > 4 px and refined K64 oracle improves official_final by > 1 px
```

Re-entry uses the existing project definition:

```text
visible after at least one consecutive GT-invisible/invalid frame
```

Report:

```text
first re-entry frame
early4 visible-valid frames from each re-entry
early8 visible-valid frames from each re-entry
occlusion-length bins: 1 / 2-4 / 5-8 / >8
```

Report globally, per sequence and per clip.

## 9. Metrics

For every coordinate readout:

```text
mean / median error
safe4 / safe8 / safe16
better / worse / equal versus official_final
mean / median paired difference versus official_final
```

For candidate refinement:

```text
raw-to-refined displacement mean / median / p95 / max
refined C2 argmax unique count
refined spatial cluster count within 4 px
raw/refined K16 and K64 oracle gap
refined score-top1 versus refined oracle gap
out-of-range coordinate rate before final evaluation clamp: reported, not silently clamped
```

Risk audit:

```text
visibility-trigger rate
uncertainty-trigger rate
trigger overlap
risk rate on visible / invisible / re-entry / opportunity rows
conceptual mean candidate count
```

## 10. Paired uncertainty

Use 100,000-resample clip-block paired bootstrap:

```text
cluster = clip_id
seed = 20260713
```

The conservative nine-clip interval is used for gates.

## 11. Integrity gates

Mandatory:

```text
all input hashes pass
clean branch/HEAD and tracked status pass
TrackOn2 code imported from clean worktree
official p/v/q parity max_abs <= 1e-6
all 27,648 query-frame rows processed exactly once
K64 first-16 candidate set equals official K16 set
score/certainty parity after one-to-one coordinate matching <= 6e-5
preflight basis: all 27,648 query-frame rows showed exact coordinate-set parity and zero official-top1 mismatch; score max 5.245e-5 and certainty max 4.864e-5
all raw/refined coordinates, scores, logits and descriptors finite
candidate-conditioned path receives no GT or error input
official memory update remains the original unconditional q_new path
saved NPZ independently reproduces reported primary means
```

## 11.1 Numerical parity preflight artifact

```text
outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_k16_k64_full_parity_preflight.json
```

This preflight covers all nine clips and all 27,648 query-frame rows without using GT metrics. It fixes the final K16/K64 numerical parity tolerance before the formal refinement result is run.

## 12. Primary refinement viability gate

`hybrid_refined_oracle` passes only if all are true relative to `official_final`:

```text
mean visible error lower on ani
mean visible error lower on animal3
mean visible error lower on r4_new_f
safe16 not lower on every sequence
global visible better rows > worse rows
nine-clip mean-difference 95% CI upper bound < 0
first-reentry mean error lower
early8 mean error lower
for each sequence with >=10 first-reentry rows, first-reentry mean does not increase
```

## 13. Decision

```text
Gate pass:
  Candidate-conditioned refinement is viable.
  Proceed to V9-A5.1c history-preserving risk-gated beam.

Gate fail:
  Do not rebuild the beam.
  The frozen proposal/refinement path is insufficient.
```

Passing V9-A5.1b does not authorize DAVIS or training.

## 14. Outputs

```text
scripts/v9a51b_candidate_conditioned_refinement_audit.py
docs/v9a51b_candidate_conditioned_refinement_result_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement.json
outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz
```
