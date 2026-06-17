# GPT Pro Next-Stage Brief — 2026-03-09

## Executive summary

We completed the full internal evidence chain for the current mainline:

- `local`
- `localglobal + top1 rematch + no commit`
- `localglobal + top1 rematch + deferred commit`

The conclusion is now much clearer than before:

1. the current `deferred commit` mechanism is **alive** (pending / confirm / commit is no longer all-zero),
2. but this version does **not** justify becoming the paper mainline,
3. and the overnight autopilot therefore correctly decided **not** to run full deferred training.

The current best interpretation is:

> retrieval helps a little, but the current single-candidate deferred-commit design is not the right state-transition mechanism.

We now want GPT Pro to review the updated code + experiments and tell us what the next architecture should be.

---

## What was tested

### Dev chain (matched recipe)

These are the main dev configs:

- `projects/mmp_tracker/configs/local_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_nocommit_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_dev.yaml`

These were run under the same basic dev recipe and compared on the same validation protocol.

### Full run

Because dev evidence was not strong enough, the overnight autopilot only ran:

- `projects/mmp_tracker/configs/local_night_full.yaml`

and explicitly **did not** run:

- `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_night_full.yaml`

That decision was made automatically from the completed dev evidence.

---

## Main code path to review

### Core code

- `projects/mmp_tracker/mmp_tracker/model.py`
- `projects/mmp_tracker/mmp_tracker/config.py`
- `projects/mmp_tracker/mmp_tracker/losses.py`
- `projects/mmp_tracker/mmp_tracker/global_relocator.py`
- `projects/mmp_tracker/mmp_tracker/local_matcher.py`
- `projects/mmp_tracker/mmp_tracker/memory_bank.py`
- `projects/mmp_tracker/train_mmp.py`

### Queue / automation

- `projects/mmp_tracker/run_controlled_commit_matrix.sh`
- `projects/mmp_tracker/run_overnight_autopilot.sh`

### Most important configs

- `projects/mmp_tracker/configs/local_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_nocommit_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_dev.yaml`
- `projects/mmp_tracker/configs/local_night_full.yaml`
- `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_night_full.yaml`

---

## What the experiments showed

The exact decision file is:

- `projects/mmp_tracker/pro_review_artifacts/20260309/overnight_autopilot_decision.json`

### Seed 42 best dev checkpoints

- `local`
  - best epoch: `0`
  - `AJ = 0.1125`
  - `AJ_longocc20 = 0.0859`
  - `AJ_longocc30 = 0.0487`
  - `reapp20 = 58.61`
  - `reapp30 = 54.04`

- `nocommit`
  - best epoch: `0`
  - `AJ = 0.1076`
  - `AJ_longocc20 = 0.0854`
  - `AJ_longocc30 = 0.0656`
  - `reapp20 = 68.41`
  - `reapp30 = 67.84`

- `deferred`
  - best epoch: `2`
  - `AJ = 0.0914`
  - `AJ_longocc20 = 0.0624`
  - `AJ_longocc30 = 0.0596`
  - `reapp20 = 61.55`
  - `reapp30 = 57.28`

### Seed 43 best dev checkpoints

- `local`
  - best epoch: `5`
  - `AJ = 0.0981`
  - `AJ_longocc20 = 0.0662`
  - `AJ_longocc30 = 0.0633`
  - `reapp20 = 63.43`
  - `reapp30 = 65.38`

- `nocommit`
  - best epoch: `3`
  - `AJ = 0.1018`
  - `AJ_longocc20 = 0.0662`
  - `AJ_longocc30 = 0.0634`
  - `reapp20 = 65.64`
  - `reapp30 = 54.55`

- `deferred`
  - best epoch: `4`
  - `AJ = 0.0920`
  - `AJ_longocc20 = 0.0662`
  - `AJ_longocc30 = 0.0540`
  - `reapp20 = 68.21`
  - `reapp30 = 59.93`

### Mean long-occ score used for autopilot decision

- `local_mean_score = 0.01135`
- `nocommit_mean_score = 0.01211`
- `deferred_mean_score = -0.00235`

Autopilot conclusion:

- `run_deferred_full = false`

### Full local baseline

Artifacts:

- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_local_night_full_20260308_232033.epoch_metrics.jsonl`

Best AJ checkpoint (epoch 3):

- `AJ = 0.1009`
- `AJ_longocc20 = 0.0717`
- `AJ_longocc30 = 0.0593`

---

## What worked

### 1. The mechanism is no longer dead

The current deferred route does stage and confirm candidates:

- `pending_stage_rate > 0`
- `pending_confirm_rate > 0`
- `commit_rate > 0`

So the old zero-commit pathology is fixed.

### 2. The evaluation / automation chain is now reliable

We verified that:

- dev runs complete correctly,
- seed42 -> seed43 -> decision -> full baseline automation works,
- and the full baseline can run unattended.

### 3. The current route is informative enough to reject

This is important: the experiments were not useless.
They give us a strong negative result:

> current top1-rematch + deferred-commit is not the right mainline.

---

## What failed

### 1. Deferred commit lost to simpler baselines

Across the two-seed summary, deferred was worse than both:

- the matched local baseline, and
- the no-commit variant.

This means the current commit mechanism is not improving the retrieval branch enough to justify its complexity.

### 2. Single-candidate state transition appears too brittle

The current design still assumes:

- one global candidate,
- one rematch around that candidate,
- one commit decision.

This may be the wrong abstraction. The retrieval side may need multiple explicit candidates, not one top1 track plus a fragile state-update rule.

### 3. Full training did not magically close the gap

Even the overnight full local baseline stayed around AJ ~0.10, which is still very far from public strong baselines.
So this is not just a "train longer" issue.

---

## Current best next-step hypothesis

Our current working hypothesis is:

> the next route should not be another version of single-candidate deferred commit.
> instead, it should likely become:
>
> `global top-k candidates -> per-candidate rematch -> candidate scoring / abstain`

The goal would be to turn the problem from:

- "should I commit this one global point?"

into:

- "which verified candidate should I trust, or should I abstain entirely?"

This is the direction we want GPT Pro to assess.

---

## What we want GPT Pro to decide

We want a hard judgment on the following:

1. Is the current `deferred commit` route decisively negative enough that we should stop it entirely?
2. Is `top-k rematch + candidate scoring / abstain` the highest-ROI next mainline?
3. If not, what architecture should replace it?
4. Given the current evidence, what is the strongest paper story still available?
5. Which experiments are absolutely required before spending more full-training budget?

---

## Important meta-point

Please do **not** answer this as if the question were still "is local/global retrieval a good idea?"

That question is too old and too broad.

The actual next decision point is:

> after a completed negative result for `top1 rematch + deferred commit`, what should the next mainline architecture be?

