# V9-A5.1c History-Preserving Risk-Gated Beam Design

Date: 2026-07-10

## 1. Motivation

V9-A5.1a showed that the tested fixed-K16 shared-state beam fails.

V9-A5.1b then separated refinement reachability from deterministic selection:

```text
GT-only hybrid refined oracle:
  official 9.1418 -> 7.8185 px
  clip 95% CI [-2.1037,-0.7501]
  all three sequences and all nine clips improve

frozen-score hybrid top1:
  official 9.1418 -> 9.1023 px
  CI [-0.1990,+0.0713]
  2085 better / 2210 worse
```

V9-A5.1c asks whether full-stream temporal history can improve selection among the candidate-conditioned alternatives while correcting the known V9-A5.1a design flaws.

No training and no DAVIS data are allowed.

## 2. Repository isolation

```text
worktree: /gemini/code/FSPT_v9a51c_clean
branch: v9a51c-history-preserving-beam-20260710
base HEAD: 29512f0d95d8b4069b73b9130e60b75d1a892b98
```

TrackOn2 Python code/config must come from this worktree. Immutable data and weights are hash verified.

The committed V9-A5.1b JSON/NPZ/script are frozen inputs and are used for row-level replay parity.

## 3. Full stream

```text
PointOdyssey: ani / animal3 / r4_new_f
clip starts: 0 / 256 / 512
96 frames per clip
32 deterministic query identities per clip
27,648 query-frame observations
frame 0 excluded from metrics
```

Official TrackOn2 memory remains on the original unconditional `q_new` update path. Beam state is an external shared-proposal-state diagnostic.

## 4. Candidate generation and refinement

Reuse the exact frozen V9-A5.1b path:

```text
official q_pre and fused C1
fused top64 C1 candidates
frozen local decoder + fusion-layer descriptor
frozen score/certainty heads
candidate-conditioned singleton fusion
projection2 -> C2 -> prediction-head offset
```

For each candidate retain:

```text
absolute fused-C1 grid index
raw C1 coordinate
256D post-fusion descriptor
frozen rerank score
candidate-conditioned refined coordinate
```

## 5. Native risk and candidate budget

```text
risk_visibility = sigmoid(v_logit) < 0.8
risk_uncertainty = sigmoid(u_logit) >= 0.5
risk = risk_visibility OR risk_uncertainty

non-risk candidate budget: K16
risk candidate budget: K64
```

Thresholds are frozen from V9-A5.1b. No calibration or sweep is allowed.

The audit may compute K64 for every row, but conceptual deployment compute is reported separately.

## 6. Beam state

Each path stores:

```text
current raw C1 coordinate
previous raw C1 coordinate, when available
current refined readout coordinate
current post-fusion descriptor
current absolute C1 grid index
last three absolute C1 grid indices
last eight normalized step costs
current window cost = sum(last eight step costs)
age
```

Raw C1 coordinate and descriptor define the hypothesis state. Refined coordinate is an attached readout and is not used to deduplicate hypotheses.

## 7. Initialization

At frame 0:

```text
raw coordinate = exact input query coordinate
refined readout coordinate = exact input query coordinate
descriptor = descriptor of the nearest fused top64 raw candidate
grid history = nearest candidate absolute C1 grid index
cost history = empty
window cost = 0
beam size = 1
```

Report query-to-nearest-candidate distance.

## 8. Transition evidence

For each parent path and current candidate:

### Emission

Descending ordinal rank of frozen candidate rerank score.

### Motion

One previous raw state:

```text
prediction = last raw coordinate
```

Two previous raw states:

```text
prediction = last raw coordinate + (last raw coordinate - previous raw coordinate)
```

Current candidates receive ascending rank by raw-coordinate distance to the prediction.

### Identity

Descending ordinal rank of cosine similarity between current post-fusion descriptor and parent descriptor.

### Normalization

For budget K:

```text
rank_norm = ordinal_rank / (K - 1)
step_cost = emission_rank_norm + motion_rank_norm + identity_rank_norm
```

No learned or manually tuned component weights are used.

## 9. Finite cost history

```text
new_cost_history = (parent_cost_history + [step_cost])[-8:]
window_cost = sum(new_cost_history)
```

The fixed eight-frame window prevents early evidence from permanently blocking re-entry recovery. Eight is declared before execution and matches the existing early8 re-entry evaluation window.

## 10. History-preserving pruning

A path signature is:

```text
(current absolute C1 grid index,
 previous absolute C1 grid index,
 second-previous absolute C1 grid index)
```

For identical signatures retain only the lowest-cost transition.

Do not deduplicate only by current candidate index or refined coordinate.

Sort remaining transitions by:

```text
window cost
current step cost
emission rank
current absolute grid index
parent slot
```

Keep the first B paths.

## 11. Predeclared policies

Primary:

```text
native_dynamic_B4
```

Diagnostics:

```text
native_dynamic_B1
native_dynamic_B8
fixed16_B4
fixed64_B4
```

No policy may replace the primary after results are observed.

## 12. Readouts

### Deterministic system readout

```text
non-risk row: official final
risk row: refined coordinate of the lowest-window-cost beam path
```

### GT-only beam reachability readout

```text
non-risk row: official final
risk row: min(error(official final), errors of surviving refined beam states)
```

### Raw-state beam oracle diagnostic

Minimum GT error among surviving raw C1 coordinates. This is diagnostic only and tests whether pruning retains a spatially valid raw hypothesis even when candidate-conditioned refinement collapses.

### Full candidate upper bound

Reuse/reproduce V9-A5.1b `hybrid_refined_oracle` for comparison with beam-pruned reachability.

GT never changes risk, transition cost, state membership, pruning or future state.

## 13. Evaluation subsets

```text
all visible
risk visible
non-risk visible
hard official: official error > 4 px
V9-A5.1b opportunity rows
first re-entry
early4
early8
occlusion length 1 / 2-4 / 5-8 / >8
```

Report globally, per sequence and per clip.

## 14. Beam diagnostics

For every policy report:

```text
refined beam survival within 1/2/4/8 px
raw-state beam survival within 1/2/4/8 px
mean unique current raw grid indices
mean unique last-three-grid signatures
mean raw-coordinate 4px clusters
mean refined-coordinate 4px clusters
beam top1/oracle gap
beam oracle versus full V9-A5.1b candidate oracle gap
fraction of full candidate-oracle gains retained after pruning
```

## 15. Replay parity against V9-A5.1b

Mandatory all-row parity:

```text
row keys exact
official final max_abs <= 1e-6
risk flags exact
candidate score max_abs <= 6e-5
raw candidate error max_abs <= 1e-4
refined candidate error max_abs <= 1e-4
V9-A5.1b hybrid oracle max_abs <= 1e-6
```

## 15.1 Beam implementation integrity

Mandatory in addition to V9-A5.1b replay parity:

```text
every policy has exact declared beam width after frame 0
all last-three-grid signatures are unique within every retained beam
all raw/refined state errors and window costs are finite
system top1 equals official final on non-risk rows
system top1 equals lowest-cost refined beam state on risk rows
system oracle equals min(official final, surviving refined states) only after pruning
raw-state beam minimum is included in summaries/bootstrap as a diagnostic
```

Partial-frame smoke replay is allowed only with one clip, because the frozen V9-A5.1b NPZ is stored in complete clip-major row order.

## 16. Statistical uncertainty

Use 100,000-resample clip-block paired bootstrap:

```text
cluster = clip_id
seed = 20260714
```

## 17. Deterministic gate

`native_dynamic_B4` deterministic system readout passes only if all are true relative to official final:

```text
mean error lower on ani / animal3 / r4_new_f
safe16 not lower on each sequence
global better > worse
clip-block 95% CI upper bound < 0
first-reentry mean lower
early8 mean lower
for each sequence with >=10 first-reentry rows, first-reentry mean does not increase
```

## 18. Beam reachability gate

If deterministic top1 fails, the GT-only B4 beam reachability readout passes only if all are true relative to official final:

```text
mean error lower on every sequence
safe16 not lower on every sequence
global better > worse
clip-block 95% CI upper bound < 0
first-reentry and early8 means lower
all nine clip mean differences <= 0
```

Interpretation:

```text
deterministic pass:
  frozen history-preserving beam is viable.

only reachability pass:
  beam retains useful alternatives but path scoring is inadequate;
  next step may be a sequence-heldout learned beam readout, not DAVIS.

reachability fail while full candidate oracle passes:
  transition/pruning destroys V9-A5.1b headroom; redesign state before training.
```

## 19. Outputs

```text
scripts/v9a51c_history_preserving_beam_audit.py
docs/v9a51c_history_preserving_beam_result_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam.json
outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz
```
