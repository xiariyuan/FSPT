# V9-A4.5 Route Closure and V9-A5 Next Step

Date: 2026-07-10

## 1. Scope

This document freezes the decision after the corrected V9-A4.5 conservative residual end-to-end smoke.

The experiment was intentionally restricted to synthetic sequence-heldout validation:

```text
train sequences: animal3 + r4_new_f
heldout sequence: ani
train rows: 2936
heldout rows: 1942
epochs: 1
DAVIS labels: not used
DAVIS evaluation: not run
```

The predeclared rule was that a materially harmful heldout result stops the route before the remaining folds and before DAVIS.

## 2. Integrity status

All implementation and provenance gates passed:

```text
clean branch: v9a45-conservative-residual-20260710
base HEAD: 4f3c01d15a5d3ed14639971d79bffef7748fc96e
tracked worktree status before run: clean
TrackOn2 Python code root: clean worktree
large immutable assets: source data root, hash verified
844 used RGB frames: individually hash verified
PointOdyssey GT reconstruction max_abs: 4.58e-05
pool/prefusion row alignment: exact
initial student/teacher score max_abs: 0
teacher/student candidate max_abs: 0
frozen base change: 0
frozen student change: 0
teacher change: 0
```

The two-step gradient audit also passed:

```text
step 1: delta head receives non-zero gradient
step 2: local decoder, fusion layer, delta head, and alpha all receive non-zero gradients
```

Therefore the result is not explained by a broken training path, missing gradients, candidate misalignment, data leakage, or frozen-parameter mutation.

## 3. Model and trust region

The student used:

```text
student_score = teacher_score + alpha * delta_score
alpha in (0, 0.25)
initial alpha = 0.05
initial delta head = 0
```

The initial student output exactly matched the teacher. TrackOn2 decoder dropout was disabled during adaptation by keeping trainable modules in evaluation mode while retaining gradients.

After one epoch:

```text
alpha: 0.0500 -> 0.0732
score-change max_abs p95: 0.2939
local-decoder parameter L2 change: 0.1219
fusion-layer parameter L2 change: 0.5689
delta-head parameter L2 change: 0.1535
```

The residual path was active and learned a non-trivial ranking change.

## 4. Heldout synthetic result

Overall heldout `ani`:

```text
teacher mean error: 14.7458 px
student mean error: 15.0387 px
mean student-minus-teacher: +0.2929 px
better / worse / equal: 35 / 70 / 1837
safe16: 1563 -> 1544
oracle regret: 10.2049 -> 10.4979
teacher-good retention: 0.9768
improvement-opportunity success: 0.0324
```

Hard/easy split:

```text
hard mean: 28.1539 -> 28.5244
hard better/worse/equal: 32/57/862
hard teacher-good retention: 0.8966

easy mean: 1.8788 -> 2.0973
easy better/worse/equal: 3/13/975
easy teacher-good retention: 0.9893
```

Per-clip mean error:

```text
ani:0   11.5395 -> 12.0649
ani:256 21.8962 -> 21.8803
ani:512  6.1312 ->  6.3560
```

The negative result is not produced by one metric only. Mean error, safe16, oracle regret, better/worse structure, hard rows, easy rows, and two of three heldout clips all point in the same direction.

## 5. Synthetic gate

```text
mean error improved: false
better > worse: false
safe16 not decreased: false
teacher-good retention >= 0.95: true
oracle regret improved: false

pass_all: false
```

Only one of the five gates passed.

## 6. Small-alpha control

A previous `alpha_logit=-8` draft kept the effective initial residual scale near `8.4e-05`.

Its heldout result was exactly equal to the teacher:

```text
better / worse / equal: 0 / 0 / 1942
```

The corrected alpha=0.05 run was independently reproduced by an earlier no-dropout runner and the final canonical script; their heldout metrics are identical.

Together these controls show:

```text
near-zero residual scale -> no learning
alpha=0.05 residual adaptation -> harmful heldout transfer
```

This does not justify an intermediate-alpha sweep. Such a sweep would be post-hoc tuning against the heldout sequence and would violate the predeclared smoke protocol.

## 7. Decision

```text
V9-A4.5 fails.
Do not run the remaining animal3/r4_new_f heldout folds.
Do not run DAVIS.
Do not increase epochs.
Do not sweep alpha, learning rate, loss weights, or residual caps on ani.
```

Combined with V9-A3.1 through V9-A4.4, this closes the current fixed-topK reranking adaptation route:

```text
post-hoc exported-summary ranking: failed
raw latent ranking: failed
pre-fusion small-head adaptation: failed
unconstrained local-decoder adaptation: failed
conservative residual local-decoder adaptation: failed synthetic heldout gate
```

The repeated pattern is training-domain improvement or active optimization without transferable within-source ranking.

## 8. Next allowed direction: V9-A5 candidate-generation / correlation supervision

The next experiment must alter which candidates enter top-K, not only rescore a frozen top-K set.

TrackOn2 currently builds the candidate map as:

```text
normalized query-feature correlation at four scales
-> bilinear upsampling to stride-4 grid
-> learned 1x1 ms_corr_proj
-> top-K spatial selection
-> local decoder / reranker
```

V9-A5 should begin with a written oracle and supervision audit around the correlation map.

### V9-A5.0: candidate-recall oracle audit

Measure, per PointOdyssey heldout sequence and DAVIS diagnostic rows:

```text
teacher top-K recall within 1/2/4/8 px of GT
recall as K varies: 1, 4, 8, 16, 32, 64
per-scale c4/c8/c16/c32 recall
union-of-scales recall
rank of nearest-GT grid location
correlation margin between nearest-GT and current top1
```

Stop immediately if top-K candidate recall is already saturated and the remaining gap is purely ordering; that would mean the current branch has reached its limit.

### V9-A5.1: conservative residual correlation adapter

Only if V9-A5.0 shows candidate-recall headroom:

```text
freeze backbone/query/memory
freeze original correlation map
learn a bounded residual over the four-scale correlation stack before ms_corr_proj/top-K
initialize residual output to zero
select all hyperparameters on sequence-heldout synthetic validation
require candidate-recall improvement on every heldout sequence
```

No DAVIS model selection is allowed.

### V9-A5 synthetic gate

Every heldout sequence must satisfy:

```text
top16 recall@4px improves
nearest-GT candidate rank improves
teacher top1 coordinate error does not materially worsen
candidate diversity does not collapse
```

Only after all synthetic folds pass may the new candidate generator be connected to a reranking head and evaluated zero-shot on DAVIS.

## 9. Canonical artifacts

```text
scripts/v9a45_conservative_residual_end_to_end.py
docs/v9a45_conservative_residual_end_to_end_design_2026-07-10.md
docs/v9a45_input_manifest_2026-07-10.json
docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json
docs/v9a45_smoke_holdout_ani_e1_seed20260710_result_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a45_conservative_residual/v9a45_smoke_holdout_ani_e1_seed20260710.json
```
