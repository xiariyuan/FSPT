# CURRENT_MAINLINE

## Status

Current re-entry / re-detection work is in a pivot state.

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
- `P0 = BLOCKED_METRIC_RECONCILIATION` is cleared only after the canonical metric is fixed to 256-space in documentation and downstream reports.

### Current direction

Preserve the re-entry / re-detection problem statement, but search for a new concrete idea.

#### Route A: external stronger teacher

- Fetch TAPNext++, Track-On-R, or AllTracker checkpoints.
- Evaluate under the unified strided+original protocol.
- If any teacher clearly beats raw CoTracker3 offline on re-entry, restart the teacher-centered route.
- If none do, do not force this path.

#### Route B: engineering foundation

- Introduce a real `fspt/` Python package.
- Add `setup.py` or `pyproject.toml`.
- Remove hard-coded paths.
- Centralize coordinate conversions.
- Split large monolithic scripts into modules.
- Add minimal smoke tests.
- Keep `CURRENT_MAINLINE.md` as the single entry point for active work.

#### Route C: analysis / benchmark paper

- Write the failure analysis cleanly if Route A does not produce a stronger teacher.
- Emphasize canonical AJ_RD, per-occ-length bins, oracle headroom, and the null-result ladder.

## Working rules

- Do not train the current DINO/local refiner/verifier stack.
- Do not restart pseudo-label rollout.
- Do not re-open P4.
- Use audits before training.
- Any new idea must introduce genuinely new information.

## This week's tasks

- Get external teacher checkpoints by 2026-06-28.
- Add package scaffolding by 2026-06-28.
- Keep this file updated as the single source of truth for active mainline work.
