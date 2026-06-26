# Current Mainline Status (2026-06-26)

This is the short current-status entry point for the repository.

## Current state

The repository is not in an active training state and does not currently claim a reproducible FSPT SOTA method.

The active work is:

1. Route A: external stronger teacher feasibility.
2. Route B: engineering foundation cleanup.
3. Route C: analysis / benchmark paper planning if Route A does not produce a stronger teacher.

## Closed routes

The following routes are closed and must not be restarted without a new gate document:

- Original frequency-semantic FSPT method claim as a current SOTA method.
- CoTracker3 + FSPT refinement as the current recommended route.
- DINOv2 local feature pseudo-label rollout.
- CT-offline local refiner.
- CT-offline local grid verifier.
- P4 student training on the old pseudo-label route.

## Current labels

- `P0 = BLOCKED_METRIC_RECONCILIATION`
- `P1 = WEAK_DIAGNOSTIC_ONLY`
- `P2 = PARTIAL_LOCAL_FEATURE_SMOKE`
- `P3A = SMOKE_STOP`
- `P4 = DO_NOT_START`

## Canonical metric

- Paper-facing canonical metric: `true_AJ_RD_256`.
- Original-resolution `true_AJ_RD` is internal diagnostic only.

## Allowed work

- Path/package cleanup.
- External stronger-teacher checkpoint acquisition and evaluation.
- Benchmark / analysis paper planning.
- New route exploration only after written gates.

## Read next

- `CURRENT_MAINLINE.md`
- `docs/current_redetection_route_closure_2026-06-26.md`
- `docs/fspt_package_path_helper_manifest_2026-06-26.md`
