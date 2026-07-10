# V9-A4.5 Conservative Residual End-to-End Rank Adaptation Design

Date: 2026-07-10

## 1. Motivation

Corrected V9-A4.4 showed that unconstrained synthetic fine-tuning is active but transfers catastrophically:

```text
rank_plus_distill C1 mean error: 6.025 px
original dynamic C1 mean error: 4.427 px
best learned AJ_RD_256 delta: +0.022851
frozen V9-A2 AJ_RD_256 delta: +0.029379
paired 95% CI: [-0.007535,-0.002366]
```

The next experiment tests whether the original TrackOn2 ranking can be improved without abruptly replacing it.

## 2. Repository isolation

Run only in:

```text
/gemini/code/FSPT_v9a45_clean
branch: v9a45-conservative-residual-20260710
base HEAD: 4f3c01d15a5d3ed14639971d79bffef7748fc96e
```

Input hashes are recorded in:

```text
docs/v9a45_input_manifest_2026-07-10.json
docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json
```

TrackOn2 Python code and configuration are loaded from the clean worktree. Large immutable datasets, checkpoints, DINOv3 weights, frozen caches, and frozen result files are read from `/gemini/code/FSPT` only after hash verification.

The original dirty worktree must not receive code or result writes from this experiment.

## 3. Data and split

PointOdyssey canonical pool:

```text
ani      1942 rows: 951 hard / 991 easy, 278 unique frames
animal3  1308 rows: 654 hard / 654 easy, 282 unique frames
r4_new_f 1628 rows: 814 hard / 814 easy, 284 unique frames
```

Three predeclared sequence-heldout folds:

```text
fold 0: hold out ani
fold 1: hold out animal3
fold 2: hold out r4_new_f
```

No random row split and no DAVIS labels for model selection.

## 4. Smoke protocol

The first corrected run is fixed as:

```text
holdout sequence: ani
training rows: all 2936 rows from animal3 + r4_new_f
epochs: 1
seed: 20260710
DAVIS evaluation: disabled
```

An earlier draft smoke used `alpha_logit=-8`, giving `alpha≈8.4e-5`, only two non-zero gradient tensors, total gradient norm about `4.1e-4`, and exactly unchanged predictions on all 1942 heldout rows. It is preserved as an invalid learning-rate/gradient diagnostic and is not a V9-A4.5 result.

## 5. Model

Keep a fully frozen TrackOn2 teacher reranking path.

The student starts functionally identical to the teacher and trains only:

```text
reranking_head.local_decoder
reranking_head.fusion_layer
new residual delta-score head
one bounded residual scale parameter
```

All backbone, FPN, query decoder, temporal memory, correlation, top-K generation, original score layer, uncertainty layer, and final query-update path remain frozen.

Score:

```text
student_score = teacher_score + alpha * delta_score
alpha = 0.25 * sigmoid(alpha_logit)
```

Initialization:

```text
delta-score weights and bias = 0
initial alpha = 0.05
alpha_logit = logit(0.05 / 0.25) = -1.386294...
```

The initial student output is exactly equal to the teacher because the delta head is zero, while the first-step delta-head gradient remains measurable.

All trainable reranking modules remain in evaluation mode during adaptation. Gradients are enabled, but TrackOn2's 10% decoder dropout is disabled so teacher/student trust-region comparisons are deterministic.

## 6. Fixed optimization settings

```text
local decoder learning rate: 1e-6
fusion layer learning rate: 5e-6
delta-score head learning rate: 5e-5
alpha-logit learning rate: 1e-3
weight decay: 1e-4
gradient clipping: 1.0
candidate temperature: 1.5 px
tie margin: 0.5 px
improvement margin trigger: 1.0 px
score margin: 0.5
```

No learning-rate or loss-weight sweep is allowed in the smoke.

## 7. Objective

For dynamically regenerated C1 top-16 candidates and reconstructed PointOdyssey GT:

```text
L = L_tie_rank
  + 2.0 * L_KL_teacher
  + 1.0 * L_teacher_good_retention
  + 0.5 * L_improvement_margin
  + 0.05 * L_delta_magnitude
  + 1e-4 * L_parameter_anchor
```

Definitions:

```text
L_tie_rank:
  uniform target over candidates within 0.5 px of the row oracle.

L_KL_teacher:
  KL from the frozen teacher ranking distribution to the student distribution.

L_teacher_good_retention:
  on rows where teacher top1 is within 1 px of oracle, preserve teacher top1.

L_improvement_margin:
  only when an alternative beats teacher top1 by more than 1 px,
  require a best candidate to exceed teacher top1 score by margin 0.5.
```

## 8. Integrity checks

Mandatory assertions:

```text
PointOdyssey GT reconstruction reproduces frozen error_px within 1e-4.
Initial student scores equal teacher scores within 1e-7.
Student and teacher candidate coordinates are identical.
Frozen parameters receive no gradients and do not change.
Only declared trainable modules move.
All losses, logits, gradients, and coordinates are finite.
No train/heldout sequence overlap.
TrackOn2 code is imported from the clean worktree.
The clean branch and HEAD match the manifest.
No tracked modifications exist before execution.
All 844 RGB frames actually used by the experiment pass the frame manifest.
The three PointOdyssey annotations, DINOv3 weights, TrackOn2 checkpoint, config, pools, and frozen results pass SHA256 verification.
```

## 9. Heldout synthetic metrics

Report teacher and student:

```text
mean / median candidate error
safe4 / safe8 / safe16
oracle regret
exact-best and within-1px-best rate
better / worse / equal rows
teacher-good retention
improvement-opportunity success
teacher/student top1 match
score residual magnitude
```

## 10. Synthetic pass gate

Every final heldout fold must satisfy all:

```text
student mean error < teacher mean error
better rows > worse rows
student safe16 >= teacher safe16
teacher-good retention >= 0.95
student oracle regret < teacher oracle regret
```

The one-fold smoke additionally requires all integrity assertions. A metric-gate miss after one epoch is recorded rather than hidden.

## 11. Decision after smoke

```text
If integrity fails: fix implementation; do not train.
If integrity passes and heldout metrics improve or remain close: run all three folds.
If integrity passes but heldout ranking collapses materially: stop before DAVIS and inspect objective/regularization.
DAVIS is evaluated only after all three sequence-heldout folds pass.
```

## 12. Recovery note

The initial 913-line source file was accidentally replaced by a malformed patch fragment after it had already compiled. The damaged fragment is preserved under `scripts/recovery/`. The compiled pre-damage implementation is preserved by SHA256 and executed only through a transparent source wrapper that overrides the audited defects. No result is accepted unless the wrapper records its own hash, the recovered-bytecode hash, clean code paths, and all integrity checks.
