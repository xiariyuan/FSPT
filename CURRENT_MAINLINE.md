# CURRENT_MAINLINE

## Status

Current re-entry / re-detection work has pivoted from the failed frozen-DINO/local-refiner routes into a metric-clean, teacher-oracle diagnostic phase.

The active branch is `mainline-pivot-foundation-20260626`.  The current paper-level metric is `true_AJ_RD_256`.

### Closed routes

- DINOv2 whole-frame retrieval.
- DINOv2 multi-anchor retrieval.
- DINOv2 local feature pseudo-label rollout.
- CT-offline-centered local refiner.
- CT-offline-centered grid verifier.
- Learned offset regression on frozen DINOv2 features.
- Any continued training on the above routes.

### Canonical metric

- Paper canonical: `true_AJ_RD_256`.
- Internal diagnostic only: original-resolution `true_AJ_RD`.
- `P0 = RESOLVED_METRIC_RECONCILIATION`.
- The canonical go/no-go delta is same-space: `oracle_teacher_selection_by_max_ajrd_256 - fixed_best_256`.
- Supporting commits:
  - `48cf9d9`: added `summary_key` support and canonical selection priority.
  - `96dadc7`: fixed the cross-space subtraction bug and made `summary_key` strong-priority.
  - `2e09e1d`: fixed the teacher-audit CLI summary-print key typo.

### Current diagnostic result

Three existing teachers were audited under the unified strided+original DAVIS protocol:

- `cotracker3_online`
- `cotracker3_offline`
- `trackon2`

Result:

- Fixed best teacher: `cotracker3_offline`.
- Fixed best `true_AJ_RD_256`: `0.5546`.
- Per-event oracle teacher selection `true_AJ_RD_256`: `0.6509`.
- Oracle upper-bound delta: `+0.0963` / `+9.6pp`.
- Decision: `STRONG_DIAGNOSTIC_HEADROOM`.

Important caveat: this is an oracle upper bound from per-event teacher selection.  It is not a deployable model gain.  It means there is real teacher complementarity and a selector / ensemble / distillation route is worth testing before any new training is authorized.

### Current direction

Preserve the re-entry / re-detection problem statement and use audits before training.

#### Route A: teacher complementarity and selector feasibility

- Immediate next baseline: evaluate a uniform ensemble of the three existing teacher caches.
- If uniform ensemble improves over `fixed_best_256`, start with ensemble distillation.
- If uniform ensemble does not improve but oracle remains strong, prioritize an event-level teacher selector / confidence gate.
- Add external teachers only after the current three-teacher baseline is understood.
- Track-On-R / TAPNext++ / AllTracker can be added to the teacher pool, but their result must be reported as either single-teacher gain, ensemble gain, or oracle upper bound.

#### Route B: engineering foundation

- `fspt/` package scaffolding exists.
- `pyproject.toml` uses `setuptools.build_meta`.
- Path helpers and environment-variable roots exist.
- Coordinate conversions and re-entry metrics have package wrappers.
- `python -m fspt.smoke` performs real checks.
- Remaining non-blocking cleanup:
  - rewrite `verify_project.py` so it no longer validates deprecated model paths;
  - add coordinate unit tests;
  - move or hard-guard stale monolithic routes such as `cotracker_refiner.py`;
  - remove remaining hard-coded paths and stale SOTA claims from historical scripts/docs.

#### Route C: analysis / benchmark paper

- Keep as fallback if Route A fails to produce a deployable selector / ensemble / teacher-distillation gain.
- Emphasize canonical AJ_RD, per-occ-length bins, oracle headroom, protocol bias, and the null-result ladder.

## Working rules

- Do not train the current DINO/local refiner/verifier stack.
- Do not restart pseudo-label rollout.
- Do not re-open P4.
- Use audits before training.
- Any new idea must introduce genuinely new information.
- Report oracle upper bounds separately from deployable gains.

## This week's tasks

- Document the three-teacher oracle result and its oracle-vs-deployable caveat.
- Run the three-teacher uniform ensemble baseline from existing caches.
- Decide whether the next Route A step is simple ensemble distillation or an event-level teacher selector.
- Fetch/evaluate external teacher checkpoints only after the current three-teacher baseline is understood, or in parallel if the environment is ready.
- Keep this file updated as the single source of truth for active mainline work.
