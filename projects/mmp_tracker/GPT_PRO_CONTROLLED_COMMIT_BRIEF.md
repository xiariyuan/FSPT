# GPT Pro Review Brief: Controlled Commit Route After Top1 Rematch

## Goal

We have now narrowed the MMP tracker paper route to a much tighter question:

> local matching + visible-memory global relocalization + top1-centered rematch + controlled commit

The old broad question of whether "local/global matching" is worth pursuing is mostly answered.
The current question is now much sharper:

> how should controlled commit be designed so that long-occlusion recovery improves reliably
> without the instability of aggressive state updates?

We want GPT Pro to review the current code and the newest evidence, and tell us what the next mainline design should be.

## Current mainline files

- trainer: `projects/mmp_tracker/train_mmp.py`
- config dataclasses: `projects/mmp_tracker/mmp_tracker/config.py`
- core model: `projects/mmp_tracker/mmp_tracker/model.py`
- losses: `projects/mmp_tracker/mmp_tracker/losses.py`
- global retrieval: `projects/mmp_tracker/mmp_tracker/global_relocator.py`
- local matcher: `projects/mmp_tracker/mmp_tracker/local_matcher.py`
- memory bank: `projects/mmp_tracker/mmp_tracker/memory_bank.py`

## Current model story

The current route is no longer the earlier residual-refiner or posterior branch.

The model story is now:

1. local matcher tracks short-range motion accurately
2. visible-memory global retrieval proposes long-occlusion candidates
3. top1-centered rematch verifies the global candidate in the current frame
4. selector decides whether to output local or global for the current frame
5. commit decides whether that global result should become recurrent state

This separation between output selection and recurrent state update is now the core of the paper idea.

## What changed in the latest iteration

### 1. Top1 rematch branch added and validated

We explicitly changed rematch center selection from expected-point rematch toward top1-centered rematch.

- config knob: `global_refine_center`
- implementation: `projects/mmp_tracker/mmp_tracker/model.py`

This matters because the global heatmap can be multi-modal, and rematching around the expectation can land between peaks.

### 2. Commit was separated from simple branch selection

The code now distinguishes between:

- selecting a global output for the current frame
- committing that global result into recurrent state

This is the key conceptual shift.

Relevant implementation:

- `projects/mmp_tracker/mmp_tracker/model.py`
- `projects/mmp_tracker/mmp_tracker/losses.py`
- `projects/mmp_tracker/train_mmp.py`

### 3. Added explicit commit diagnostics

The evaluator now reports not just AJ/OA but also:

- `selected_global_rate`
- `commit_rate`
- `commit_given_selected`
- `global_better_rate`
- `selected_when_global_better`
- `commit_when_global_better`
- `selected_precision`
- `commit_precision`
- `blocked_by_quality_rate`
- `blocked_by_consistency_rate`

These diagnostics are intended to tell us whether the failure mode is retrieval, routing, or commit.

### 4. Added commit supervision path

We added a temporary learned commit path in code and tested it, but the empirical result was poor.

That branch appears to confirm that commit matters, but the learned commit design used here is not yet the final answer.

## Latest experiment evidence

### Stable internal baseline

`local_dev` on seed 42:

- AJ = 0.1096
- AJ_longocc20 = 0.0664
- AJ_longocc30 = 0.0553
- reapp_error_longocc20_px = 64.78
- reapp_error_longocc30_px = 57.46

This is the main internal comparator.

### Top1 rematch result

`localglobal_top1rematch_dev` on seed 42 produced the first real positive signal.

Best AJ checkpoint:

- AJ = 0.1076
- AJ_longocc20 = 0.0690
- AJ_longocc30 = 0.0485

Best long-occlusion checkpoint:

- AJ = 0.1046
- AJ_longocc20 = 0.0721
- AJ_longocc30 = 0.0520
- reapp_error_longocc20_px = 57.11
- reapp_error_longocc30_px = 52.85

Interpretation:

- top1 rematch clearly helps long-occ20 and reappearance error
- but overall AJ still trails the local baseline slightly
- and long-occ30 is not stably above baseline yet under the pure top1-rematch variant

### Aggressive relaxed commit result

`localglobal_top1_relaxedcommit_dev` on seed 42 showed stronger long-occlusion upside.

Representative strong checkpoints:

- epoch 2:
  - AJ = 0.0993
  - AJ_longocc20 = 0.0762
  - AJ_longocc30 = 0.0598
  - reapp_error_longocc20_px = 58.21
  - reapp_error_longocc30_px = 54.83
- epoch 3:
  - AJ = 0.0960
  - AJ_longocc20 = 0.0769
  - AJ_longocc30 = 0.0586
  - reapp_error_longocc20_px = 56.74
  - reapp_error_longocc30_px = 59.25

Interpretation:

- this is the first time both long-occ20 and long-occ30 exceeded the seed-42 local baseline
- but the branch was not stable over training and later collapsed back down

### Seed 43 instability test

We then ran a seed-43 paired comparison and stopped it early to save cost once the instability became clear.

`local_dev` seed 43 remained healthy:

- epoch 5: AJ = 0.1138
- epoch 6: AJ = 0.1122

`top1_relaxedcommit` seed 43 was unstable and clearly worse:

- epoch 0: AJ = 0.0810, avg_error_px = 53.46
- epoch 1: AJ = 0.0858, avg_error_px = 49.21
- epoch 2: AJ = 0.0677, avg_error_px = 59.70
- epoch 3: AJ = 0.0721, avg_error_px = 58.26
- epoch 4: AJ = 0.0840, avg_error_px = 55.33

Its routing behavior was also far more aggressive:

- `selected_global_rate` around 0.18 to 0.21
- `commit_rate` matched selected rate because commit mode was effectively always-selected once chosen

Interpretation:

- aggressive commit can unlock long-occlusion recovery on some seeds
- but the current always-selected version is too unstable to be the final mainline

### Learned commit head test

We also tested a commit-head variant and it was not useful.

Observed behavior:

- commit rate stayed effectively 0
- branch did not outperform the simpler top1-rematch result

Interpretation:

- commit matters, but the tested commit-head design is not yet the right mechanism

## Current technical conclusion

At this point we believe the following are already established:

1. `top1 rematch` is a real improvement over expected-point rematch.
2. commit is structurally important.
3. purely aggressive `always_selected` commit is too unstable across seeds.
4. the next mainline should likely be a controlled commit mechanism rather than either:
   - heuristic-overconservative commit that almost never fires
   - or always-selected commit that overfires and destabilizes tracking

## What we want GPT Pro to answer now

### Main design question

Given the above evidence, what is the most defensible next design for controlled commit?

We especially want judgment on whether the next mainline should be:

1. thresholded controlled commit over top1-rematch
2. learned commit with a different target / abstention formulation
3. candidate-based commit / abstain instead of direct binary commit
4. selector redesign first before commit redesign
5. or some simpler alternative we are missing

### Specific questions

1. Is the paper story now legitimately centered on `top1 rematch + controlled commit`?
2. What is the highest-ROI controlled-commit redesign that is still publication-oriented rather than over-engineered?
3. Should commit be supervised as:
   - local-vs-global instantaneous correctness,
   - state-update safety,
   - abstention / defer decision,
   - or a candidate-ranking problem?
4. Is selector still the main bottleneck, or has the evidence shifted the bottleneck clearly to commit?
5. What are the minimum next experiments needed to convert this into a publishable evidence chain?

## Requested answer format

Please answer in this structure:

1. `Executive judgment`
2. `What is now established`
3. `What remains structurally wrong`
4. `Best next controlled-commit design`
5. `Minimal next experiment set`
6. `Paper-story recommendation`

## Important constraint

Please do not give generic advice like "train longer".

We want a blunt structural review of controlled commit specifically, now that:

- top1 rematch is already showing value
- aggressive commit has both upside and instability
- the question has narrowed from broad architecture search to commit design
