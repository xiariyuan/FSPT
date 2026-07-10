# V9-A4 Comprehensive Review and Next Step

Date: 2026-07-10

## 1. Repository state

```text
branch: mainline-pivot-foundation-20260626
HEAD: 4f3c01d15a5d3ed14639971d79bffef7748fc96e
active experiment processes: none
tracked modified files: 15
untracked V9-A3/V9-A4 files: about 80
```

Multiple sessions have written overlapping V9-A4 artifacts into the same worktree. Before the next training run, create a clean branch/worktree and explicitly select the canonical scripts/results. Do not bulk-commit the current worktree.

## 2. Canonical evidence chain

### V9-A2.4

Frozen dynamic-horizon prototype:

```text
all_logreg + event_max + fixed0.05
AJ_RD_256 delta +0.029379
paired versus W16 mean +0.000665
95% CI [-0.000051,+0.001681]
```

Positive prototype, not statistically established final method.

### V9-A3.0

TrackOn2 C1/C2 top-K oracle shows real action-space headroom:

```text
union coordinate oracle AJ_RD_256 delta +0.036312
full action oracle about +0.0374
C1 supplies most oracle choices
```

The opportunity is candidate ranking, not another selector threshold.

### V9-A3.1–V9-A3.9

All post-hoc models over exported summaries fail:

```text
tree rankers
pairwise/listwise heads
source factorization
full-touched/full-frame supervision
RGB/PointOdyssey pretraining
candidate DINOv3 features
pretrained set-rank head
```

### V9-A4.1–V9-A4.3

Raw post-fusion latents and cached pre-fusion head adaptation also fail video-heldout transfer.

The high-capacity raw-latent scorer can fit DAVIS in-sample almost perfectly but worsens video-heldout ranking. This is a generalization failure, not a capacity or label bug.

## 3. Corrected V9-A4.4 audit

Artifacts:

```text
scripts/v9a44_end_to_end_rerank_pilot_corrected.py
outputs/paper_discovery_2026-07-05/v9a44_end_to_end_rerank/v9a44_end_to_end_rerank_pilot_corrected.json
outputs/paper_discovery_2026-07-05/v9a44_end_to_end_rerank/v9a44_end_to_end_rerank_pilot_corrected_scores.npz
docs/v9a44_end_to_end_rerank_pilot_corrected_result_2026-07-10.md
```

The original V9-A4.4 draft had several invalidating issues:

```text
1. DAVIS frames were not resized to the same 256x256 preprocessing used by the exporter.
2. fp16 q_pre changed top-K ordering on some rows, but historical error labels were still indexed by the old order.
3. apply-back selected historical coordinates instead of the dynamically recomputed coordinates.
4. frozen backbone/query modules were put into train mode.
5. C1 coordinate metrics included GT-invisible rows.
6. the primary combined the new C1 head with an already failed C2 raw-latent ranker.
7. distillation used historical logits in the old candidate order.
```

The corrected pilot:

```text
reconstructs PointOdyssey GT with error reproduction max_abs < 1e-4
recomputes errors on dynamic candidates
uses dynamic coordinates for apply-back
keeps all frozen modules in eval mode
uses only 490 GT-visible DAVIS extension rows for C1 ranking
uses original C2 top1 in the primary
uses a frozen dynamic teacher for distillation
```

Corrected result:

```text
original dynamic C1 score:
  mean error 4.427 px
  oracle regret 2.691 px

rank_only:
  mean error 7.674 px
  oracle regret 5.938 px

rank_plus_distill:
  mean error 6.025 px
  oracle regret 4.289 px

best fully learned trajectory:
  frozen learned-source + C2-top1 + rank_plus_distill
  AJ_RD_256 delta +0.022851
  frozen V9-A2 +0.029379
  paired mean -0.004803
  95% CI [-0.007535,-0.002366]
  sign-flip p=0.000427
```

Training was active rather than broken:

```text
rank_only loss 2.624 -> 2.284
rank_plus_distill loss 2.677 -> 2.376
parameter L2 movement about 1.2
candidate coordinates unchanged by the trainable modules
```

The failure is catastrophic transfer/forgetting: the student departs from the original ranking without learning a transferable alternative. In the distillation variant, teacher/student top1 agreement during training is only about 16.2%.

## 4. What is now ruled out

Do not continue with:

```text
more selector threshold tuning
more post-hoc tree/MLP/listwise models on exported summaries
more post-fusion latent decoders
small cached fusion/score adaptations
unconstrained local-decoder fine-tuning on 1,024 hard synthetic rows
simply increasing epochs for the current V9-A4.4 objective
```

The current two-epoch pilot is not proof that full end-to-end training is impossible, but more of the same objective is not justified because it already moves sharply in the wrong direction.

## 5. Remaining causal structure

The factorized diagnostics remain important:

```text
learned source + oracle within-source rank reaches about +0.0328 AJ_RD_256
full coordinate oracle reaches about +0.0354 under frozen activation
```

Therefore the source/gate path is not the dominant remaining blocker. The blocker is transferable within-source ranking.

## 6. Next experiment: V9-A4.5 conservative residual end-to-end rank adaptation

Do not immediately run a larger version of V9-A4.4. First require synthetic sequence-heldout evidence.

### 6.1 Data split

Use all 4,878 PointOdyssey rows, balanced hard/easy, with three sequence-heldout folds:

```text
train on two sequences
validate on the third sequence
rotate across ani / animal3 / r4_new_f
```

No DAVIS labels are used for model selection.

### 6.2 Model

Keep a fully frozen TrackOn2 teacher reranking path.

Student path trains:

```text
local_decoder
fusion_layer
residual delta-score head
```

Use a trust-region score:

```text
student_score = teacher_score + alpha * delta_score
alpha constrained to [0, 0.25] and initialized at 0
```

This prevents the immediate catastrophic ranking replacement observed in V9-A4.4.

### 6.3 Objective

Use a conservative objective:

```text
1. tie-aware best-candidate ranking loss
2. KL trust-region to teacher ranking
3. explicit retention loss on rows where teacher is already within 1 px of oracle
4. improvement-margin loss only when an alternative beats teacher top1 by >1 px
```

Do not tune against DAVIS trajectory metrics.

### 6.4 Synthetic pass gate

Every heldout sequence must satisfy:

```text
mean candidate error < teacher
better rows > worse rows
safe16 does not decrease
teacher-good retention >= 95%
oracle regret decreases
```

If any heldout sequence fails, stop the A-level cached/frozen-topK route without running DAVIS.

### 6.5 DAVIS evaluation gate

Only after the synthetic gate passes:

```text
zero-shot DAVIS evaluation
frozen V9-A2 fixed0.05 activation
existing heldout source prediction
C2 original top1
paired video-level AJ_RD_256 versus frozen V9-A2
```

### 6.6 Decision

```text
If V9-A4.5 passes synthetic sequence-heldout validation and improves DAVIS, scale to full sequence-level training.
If it passes synthetic but fails DAVIS, the dominant issue is domain transfer; add broader synthetic domains rather than DAVIS threshold tuning.
If it fails synthetic, stop this method branch. The next change must alter candidate generation/correlation supervision, not reranking adaptation.
```

## 7. Immediate execution order

```text
1. Create a clean worktree from HEAD 4f3c01d.
2. Copy only the canonical V9-A3.0, V9-A4.0 latent, corrected V9-A4.4 scripts/results needed for provenance.
3. Add a manifest with file hashes and row-alignment assertions.
4. Implement V9-A4.5 with sequence-heldout synthetic validation.
5. Run a 1-fold/1-epoch smoke.
6. Run the three predeclared sequence folds.
7. Run DAVIS only if all synthetic gates pass.
```
