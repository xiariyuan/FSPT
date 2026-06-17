# Deferred Commit Update — 2026-03-08

## What changed

- Added a real `deferred` commit mode in `projects/mmp_tracker/mmp_tracker/model.py`.
- Introduced one-step pending state:
  - stage a global candidate without requiring `selected_global`
  - confirm on the next frame before writing recurrent state
- Split next-frame confirmation into two valid paths:
  - local verify path: pending-centered rematch passes quality and motion-consistency
  - global re-confirm path: current global candidate re-matches the pending point with sufficient quality
- Prevented unsafe local/template/memory writes while a pending candidate is unresolved.
- Added clean pending diagnostics to model info + eval metrics:
  - `pending_input_rate`
  - `pending_stage_rate`
  - `pending_confirm_rate`
  - `confirm_given_pending`
  - `pending_stage_when_global_better`
  - `pending_confirm_when_global_better`
  - `pending_stage_precision`
  - `pending_confirm_precision`

## Config changes

- Added `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_dev.yaml`
- Fixed `projects/mmp_tracker/configs/localglobal_nocommit_dev.yaml` into a clean top1-rematch no-commit ablation:
  - `global_refine_center: top1`
  - `commit_mode: none`

## Current evidence goal

This matrix is meant to answer a narrower causal question:

`local` vs `top1 + no commit` vs `top1 + deferred commit`

across paired seeds, under the same recipe.

## Why this is the right next step

- `top1 rematch` already showed the first real long-occlusion signal.
- `always_selected` commit showed upside but was too unstable.
- The remaining bottleneck is now state transition, not posterior fusion and not frequency modules.

## Runs in progress

Target matrix currently being run on the remote server:

- `local_dev`, seed 42
- `local_dev`, seed 43
- `localglobal_nocommit_dev`, seed 42
- `localglobal_nocommit_dev`, seed 43
- `localglobal_top1_deferredcommit_dev`, seed 42
- `localglobal_top1_deferredcommit_dev`, seed 43

## Decision rules after the matrix finishes

1. If `deferred commit` does not beat `nocommit` on long-occ / reappearance metrics across both seeds, do not promote it.
2. If `deferred commit` beats `nocommit` but not `local`, revisit commit thresholds and confirmation geometry before scaling training.
3. If `deferred commit` beats both `nocommit` and `local` repeatably, re-run a matched `top1rematch` heuristic baseline and complete the causal ladder:
   - `local`
   - `top1 + no commit`
   - `top1 + heuristic commit`
   - `top1 + deferred commit`

## Likely next redesign if this still fails

If the new deferred design still underperforms, the next architecture step should be:

- top-k global candidates
- per-candidate rematch / rescoring
- learned abstain-or-commit policy

not a return to posterior weighted fusion.
