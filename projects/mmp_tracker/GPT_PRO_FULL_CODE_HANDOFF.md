# GPT Pro Full-Code Handoff for Controlled-Commit Review

## Why this package exists

This package is intended for a deeper GPT Pro review when we are no longer asking a narrow code question,
but need a broader architectural judgment over the current research direction.

The user explicitly prefers that when the next decision point is uncertain, we should:

1. package the full codebase relevant to the current route,
2. include a plain summary of what has already been tried,
3. include the future experiment plan,
4. and ask GPT Pro to reason over the bigger picture rather than only one small patch.

This file is meant to carry that context into the review bundle.

## Current research route

We are not using the old residual-refiner / frequency-mainline route anymore.

The current route is:

> local matching + visible-memory global retrieval + top1 rematch + controlled commit

The paper hypothesis is no longer just "global retrieval helps".

The sharper hypothesis is:

> long-occlusion tracking improves when global recovery is explicitly verified in-frame
> and then committed into recurrent state in a controlled way.

## What has already been tried

### 1. Matched local baseline

Config:

- `projects/mmp_tracker/configs/local_dev.yaml`

Role:

- This is the current internal baseline for fair comparison.

### 2. Early localglobal mainline

Configs:

- `projects/mmp_tracker/configs/localglobal_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_nocommit_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_norematch_dev.yaml`

Observed conclusion:

- the original localglobal route did not beat local baseline overall
- selector / commit activity was too low
- rematch was likely important

### 3. Top1 rematch branch

Config:

- `projects/mmp_tracker/configs/localglobal_top1rematch_dev.yaml`

Observed conclusion:

- this was the first variant that produced a real positive long-occlusion signal
- compared with expected-point rematch, top1-centered rematch was better aligned with the retrieval behavior
- long-occ20 and reappearance error improved meaningfully on seed 42

### 4. Learned commit-head branch

Config:

- `projects/mmp_tracker/configs/localglobal_commithead_dev.yaml`

Observed conclusion:

- commit almost never fired
- branch did not outperform top1 rematch
- this suggested commit matters, but this particular learned commit design was not effective

### 5. Aggressive relaxed-commit branch

Config:

- `projects/mmp_tracker/configs/localglobal_top1_relaxedcommit_dev.yaml`

Observed conclusion on seed 42:

- this variant was the first one to push both `AJ_longocc20` and `AJ_longocc30` above the local baseline at some checkpoints
- this strongly suggests commit is a real bottleneck

Observed conclusion on seed 43:

- the same aggressive route became unstable
- global selection / commit became too frequent
- AJ and long-occ metrics became clearly worse than the seed-43 local baseline

Interpretation:

- aggressive commit is useful as a diagnostic upper-bound experiment
- but it is not stable enough to be the final mainline

## What is currently believed to be established

1. `top1 rematch` is a real improvement over expected-point rematch.
2. commit is a real bottleneck.
3. over-conservative commit wastes global recovery opportunities.
4. always-selected style commit is too unstable to serve as the final paper design.
5. the next route should likely be a controlled-commit design between those two extremes.

## What is still uncertain

These are the questions we specifically want GPT Pro to help resolve:

1. What is the best publication-oriented controlled-commit design?
2. Should controlled commit remain threshold-based, or become an abstention / defer mechanism?
3. Is selector still the main bottleneck, or has commit now clearly become the main bottleneck?
4. What is the minimum experiment set needed next to make the evidence chain paper-ready?
5. Does this route still have a top-tier upgrade path, or should we aim for a narrower / mid-tier story?

## Future plan if the route continues

The current intended next step is not to add more unrelated modules.

The intended next step is:

1. retire `always_selected` as a final design candidate,
2. design a more controlled commit variant,
3. run only the minimum next experiments needed to validate that controlled-commit mechanism,
4. then build the formal result tables and paper-story assets.

In other words, the project is now in a convergence phase, not a wide-open exploration phase.

## Recommended reading order for GPT Pro

1. `projects/mmp_tracker/GPT_PRO_CONTROLLED_COMMIT_BRIEF.md`
2. `projects/mmp_tracker/GPT_PRO_FULL_CODE_HANDOFF.md`
3. `projects/mmp_tracker/PAPER_EVIDENCE.md`
4. `projects/mmp_tracker/EXPERIMENT_MATRIX.md`
5. implementation files under `projects/mmp_tracker/`

## Bundle scope

This full-code review package should include:

- the main repository code needed to understand the current route
- the baselines and utilities if relevant
- the current MMP tracker project files
- configs and dataset code

It should exclude large non-code artifacts such as:

- weights
- checkpoints
- cached datasets
- logs / outputs
- previous review bundles

## Bottom line for the reviewer

Please do not answer with generic "train longer" advice.

We already know the route is now about controlled commit.
The real question is how to turn the current diagnostic evidence into a stable, defensible mainline.
