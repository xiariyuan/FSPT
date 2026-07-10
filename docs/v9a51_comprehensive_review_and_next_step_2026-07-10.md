# V9-A5.1 Comprehensive Review and Next Step

Date: 2026-07-10

## 1. Executive conclusion

Three pieces of evidence are now simultaneously true:

```text
V9-A5.0:
  past-correct temporal state has very large headroom,
  but sampled self-state propagation fails.

V9-A5C.0:
  fused top64 adds large, sequence-consistent candidate recall over top16,
  concentrated on hard rows.

Independent V9-A5.1 fixed-K16 full-stream beam:
  deterministic top1 and same-capacity beam-oracle both fail.
```

The correct interpretation is not “all temporal multi-hypothesis methods fail.”

The defensible interpretation is:

```text
The tested fixed-K16, shared-proposal-state, raw-C1-coordinate beam with
current-candidate deduplication and indefinitely accumulated ordinal cost fails.
```

The next experiment must first test whether individual C1 hypotheses can be converted into candidate-conditioned refined final positions. A larger beam should not be run before this downstream-refinement gate.

## 2. Repository and artifact state

### Canonical committed branch

```text
worktree: /gemini/code/FSPT_v9a45_clean
branch: v9a45-conservative-residual-20260710
HEAD: 448c095543bf3904898170660ac9944c76963615
```

Committed evidence:

```text
8fe78db  V9-A4.5 residual reranking route closure
7c24367  V9-A5.0 sampled-causal temporal-state audit
448c095  V9-A5C.0 fused correlation candidate-recall audit
```

### Independent fixed-K16 V9-A5.1 worktree

```text
worktree: /gemini/code/FSPT_v9a51_clean
branch: v9a51-fullstream-beam-20260710
base HEAD: 7c24367cf8aeb1467946d2ced3b294299032538e
```

Complete uncommitted artifacts exist there:

```text
scripts/v9a51_fullstream_beam_reachability.py
docs/v9a51_fullstream_beam_reachability_design_2026-07-10.md
docs/v9a51_fullstream_beam_reachability_result_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a51_fullstream_beam/v9a51_fullstream_beam_reachability.json
outputs/paper_discovery_2026-07-05/v9a51_fullstream_beam/v9a51_fullstream_beam_reachability.npz
```

### Corrupted experimental draft

A custom patch operation overwrote the untracked dynamic-K draft in the canonical worktree with a 680-byte patch fragment. No committed file or canonical result was affected.

The fragment was quarantined at:

```text
/gemini/code/FSPT_v9a45_untracked_archive_20260710/recovery/
v9a51_fullstream_dynamic_k_beam_audit.corrupt_patch_fragment.py
```

The corrupted path was removed from the worktree to prevent accidental execution.

## 3. Independent fixed-K16 result integrity

The independent audit has strong engineering integrity:

```text
27,648 query-frame rows processed exactly once
20,218 GT-visible rows
864 RGB frames hash verified
official p/v/q parity max_abs = 0
sampled-pool candidate set parity <= 1e-6
sampled teacher score max_abs = 0
sampled teacher top1 match = 1.0
all outputs finite
saved NPZ reproduces reported primary means exactly
beam update signature contains no GT/error argument
```

The script hash in the JSON matches the source:

```text
4c49df7c743ba14229f30786d13e2776348f9bb66f14aa40b0b6215c94bc5ee8
```

Therefore the fixed-K16 negative result is not explained by data corruption, missing rows, GT leakage, official-forward drift, or report-generation error.

## 4. Fixed-K16 result after independent recomputation

### 4.1 Frame 0 does not explain the failure

The original report includes frame 0. Frame 0 has 288 visible query rows and all major top1 policies equal the teacher there.

After excluding frame 0:

```text
teacher top1 mean:                     9.4394 px
beam4 motion+latent top1:              9.8841 px
mean difference:                      +0.4447 px
better / worse / equal:               2879 / 5158 / 11893

teacher top4 oracle:                   6.3934 px
beam4 motion+latent oracle:            7.5775 px
mean difference:                      +1.1841 px
better / worse / equal:               1369 / 2293 / 16268
```

The negative direction is slightly stronger after excluding frame 0.

### 4.2 The result is sequence-wide

Frame>0 primary top1 mean differences versus teacher:

```text
ani:       +0.2798 px
animal3:   +0.0704 px
r4_new_f:  +0.9840 px
```

Eight of nine clips are worse for the primary deterministic beam. The only nearly neutral clip is `ani:512`.

### 4.3 Re-entry is also worse

```text
first re-entry:
  teacher top1                  18.5112 px
  beam4 motion+latent top1      19.8078 px

re-entry early8:
  teacher top1                  16.4022 px
  beam4 motion+latent top1      17.4539 px

re-entry after occlusion >=4:
  teacher top1                  23.2471 px
  beam4 motion+latent top1      24.5195 px
```

The same-capacity beam oracle is also worse than teacher top4 on all these subsets.

### 4.4 Width 8 does not create temporal reachability beyond raw top8

The width-8 beam oracle is better than teacher top4 only because it retains more candidates. The fair same-capacity comparison is:

```text
teacher top8 oracle:            5.3604 px
beam8 motion+latent oracle:      6.2353 px
mean difference:               +0.8749 px
better / worse / equal:         831 / 1388 / 17711
```

Thus the temporal transition/pruning removes useful candidates relative to simply retaining the current teacher top8.

### 4.5 Beam membership changes but is not improved

On visible frame>0 rows:

```text
beam4 set equals teacher top4 set: about 48.7%
beam4 mean overlap with teacher top4: about 82.5%

beam8 set equals teacher top8 set: about 27.7%
beam8 mean overlap with teacher top8: about 85.8%
```

The temporal beam is not merely reproducing the teacher set; it actively replaces candidates. The replacements are, on average, harmful.

## 5. What the fixed-K16 audit validly rules out

Do not repeat the following exact configuration:

```text
frozen C1 top16 only
raw C1 patch-center coordinates as final beam output
beam initialized from teacher-ranked C1 candidates
indefinitely accumulated ordinal path cost
current-candidate-index deduplication
teacher + constant-velocity + previous-latent Borda
shared official TrackOn2 proposal/query/memory state
primary comparison only against C1 teacher top1
```

Increasing the same beam width from 4 to 8 is not justified. The fair top8 comparison is still negative.

## 6. What it does not rule out

### 6.1 K64 risk-gated proposal expansion

The fixed-K16 worktree predates V9-A5C.0. It never sees the large hard-row recall headroom:

```text
hard fused recall@4:
K16 0.5420 -> K64 0.8561
```

Therefore it cannot answer whether a risk-gated K64 beam can preserve a correct hypothesis that does not exist in K16.

### 6.2 Distinct histories at the same current coordinate

The existing pruning retains only one path per current candidate index.

Two paths may land on the same current coordinate while having different previous coordinates and velocities. Merging them deletes exactly the multi-history information needed for the next-frame motion prediction.

A valid history beam must preserve distinct incoming histories or explicitly deduplicate on a fuller state signature, not only current candidate index.

### 6.3 Candidate-conditioned downstream refinement

The existing beam outputs raw C1 patch centers.

Official TrackOn2 produces its final coordinate through:

```text
C1 local candidate decoding
-> candidate fusion descriptors
-> fusion across candidates
-> projection2
-> C2 correlation
-> prediction-head offsets
```

Raw C1 patch centers are coarser than the official final output. Before using a C1 hypothesis as a system output, the experiment must determine whether that hypothesis can drive a candidate-conditioned C2/offset refinement.

### 6.4 System-level official-final hybrid

The existing primary compares against C1 teacher top1, not the official final tracker coordinate.

A deployable risk-gated system should retain official final output on non-risk frames and use a beam/refined hypothesis only on risk frames. Its primary gate must compare against official final.

### 6.5 True model-state multi-hypothesis tracking

All existing beams share one official TrackOn2 query and memory stream. The beam changes only coordinate/descriptor readout history.

This is a shared-proposal-state beam, not an independent per-hypothesis query/memory tracker. If the official memory state drifts, every beam path receives candidates generated from the same drifted state.

A failure of shared-state beam readout does not prove that independent model-state hypotheses are impossible.

## 7. Additional methodological omissions

### 7.1 Infinite cumulative path cost

The existing beam sums ordinal costs from frame 0 through frame 95 without forgetting.

Early score differences can dominate later evidence and prevent recovery after occlusion. This is especially problematic for a re-entry method.

A future beam should use a predeclared finite history, such as the last eight step costs, or another written trust/forgetting rule. This must be fixed before running, not selected from DAVIS or heldout results.

### 7.2 Risk-trigger audit

The model-native risk rule should be audited as separate components:

```text
visibility trigger: sigmoid(v_logit) < 0.8
uncertainty trigger: sigmoid(u_logit) >= 0.5
union and overlap
```

The uncertainty target in TrackOn2 training is positive when error exceeds 12 px or GT is invisible, so high uncertainty probability has the intended semantic direction.

Report activation on:

```text
visible rows
invisible rows
first re-entry rows
hard opportunity rows
```

### 7.3 Actual versus conceptual compute

A diagnostic may compute K64 on every row to compare policies, but a deployable dynamic-K method is two-stage:

```text
run official K16
compute native risk
run K64 extension only on triggered rows
```

Both actual audit compute and conceptual deployment compute must be reported separately.

### 7.4 Initialization

The fixed-K16 beam initializes from teacher-ranked patch candidates, although the exact input query coordinate is known at frame 0.

This did not cause the observed negative aggregate result, but a cleaner state initialization is:

```text
coordinate = exact input query
candidate descriptor = descriptor of nearest current candidate
path cost = 0
```

The query-to-nearest-candidate initialization distance must be reported.

## 8. Reflection

The current research sequence attempted a full beam before validating a necessary intermediate operation: whether one candidate hypothesis can be transformed into a refined downstream final coordinate.

That made the experiment harder to interpret because three failures were mixed together:

```text
candidate availability
candidate-conditioned refinement
history transition/pruning/readout
```

The next phase should separate these causes with strict gates.

A second process lesson is that untracked experimental files are still vulnerable to accidental tool overwrite. The canonical commits were unaffected, but future complex scripts should be created in a dedicated worktree and checkpointed with a local commit immediately after syntax/integrity smoke, before further edits.

## 9. Next experiment: V9-A5.1b candidate-conditioned downstream refinement audit

Do not immediately rerun a larger beam.

### 9.1 Goal

Determine whether each fused C1 top64 hypothesis can be converted into a meaningful candidate-conditioned final coordinate using frozen TrackOn2 downstream modules.

### 9.2 Candidate-conditioned branch

For each C1 candidate descriptor `z_i`:

```text
q_pre = official pre-rerank query
z_i = candidate local-decoder + fusion-layer descriptor

q_conditioned_i = reranking_head.fusion(q_pre, z_i, z_i)
q_conditioned_i = final_projection_layer([q_conditioned_i, q_pre])
q2_i = projection2(q_conditioned_i)
c2_i = multiscale_correlation(q2_i, frame features)
p_patch_i = argmax(c2_i)
offset_i = prediction_head(q2_i, frame features, p_patch_i)
refined_position_i = p_patch_i + final offset_i
```

This reuses frozen modules and conditions the downstream branch on exactly one C1 hypothesis. It is diagnostic, not a claim that the original model was trained with singleton fusion.

### 9.3 Predeclared proposal modes

```text
official fused K16
fused K64
native-risk dynamic K16/K64
```

Risk remains fixed:

```text
sigmoid(v_logit) < 0.8 OR sigmoid(u_logit) >= 0.5
```

### 9.4 Required outputs

For raw C1 and candidate-conditioned refined coordinates report:

```text
teacher-score top1
K16 oracle
K64 oracle
risk-gated hybrid oracle with official final fallback
mean / median / safe4 / safe8 / safe16
per sequence and per clip
re-entry first / early4 / early8
hard opportunity rows
clip-block paired 95% CI versus official final
risk activation, opportunity recall and conceptual mean candidate count
```

### 9.5 Integrity gates

```text
official p/v/q parity <= 1e-6
K64 first-16 candidate set equals official K16 set
score/certainty parity after coordinate matching
candidate-conditioned outputs finite
no GT enters candidate generation or refinement
official final output is bit-identical to the normal forward
all 9 clips x 96 frames x 32 queries processed exactly once
```

### 9.6 Pass gate

The risk-gated refined candidate oracle passes only if:

```text
mean error lower than official final on ani / animal3 / r4_new_f
safe16 not lower on every sequence
global better > worse
9-clip paired CI upper bound < 0
first-reentry and early8 mean error lower than official final
```

If this gate fails, do not rebuild the beam. The proposal/refinement path is insufficient.

## 10. Conditional next experiment: V9-A5.1c history-preserving hybrid beam

Run only if V9-A5.1b passes.

Required corrections relative to the fixed-K16 baseline:

```text
risk-gated K16/K64 candidates
candidate-conditioned refined coordinates
exact input-query initialization
finite eight-frame cost history
preserve distinct incoming histories even at the same current candidate
report coordinate and velocity-state diversity
non-risk output = official final
risk output = deterministic beam top1
GT-only hybrid oracle = min(official final, surviving beam states)
primary comparison = official final
```

Keep the independent fixed-K16 result as `V9-A5.1a` negative baseline. Do not overwrite or reinterpret it as the final V9-A5.1 route closure.

## 11. Immediate execution order

```text
1. Preserve the independent fixed-K16 worktree read-only.
2. Reclassify its artifacts as V9-A5.1a fixed-K16 shared-state beam baseline.
3. Create a new clean worktree from committed HEAD 448c095.
4. Implement only V9-A5.1b candidate-conditioned refinement.
5. Run one-clip / selected-frame integrity smoke.
6. Run the full nine-clip synthetic audit only if smoke parity passes.
7. Build V9-A5.1c beam only if the refined risk-gated oracle passes.
8. Do not read DAVIS and do not train during these audits.
```
