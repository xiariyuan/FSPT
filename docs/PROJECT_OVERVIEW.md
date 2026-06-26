# Project Overview Status

> Status: historical overview, superseded by `CURRENT_MAINLINE.md`.

This file used to describe FSPT as a completed frequency-semantic point tracker. That no longer reflects the current project state.

## Current authoritative entry points

- [`../CURRENT_MAINLINE.md`](../CURRENT_MAINLINE.md)
- [`current_redetection_route_closure_2026-06-26.md`](current_redetection_route_closure_2026-06-26.md)
- [`route_b_engineering_foundation_plan_2026-06-26.md`](route_b_engineering_foundation_plan_2026-06-26.md)
- [`fspt_package_path_helper_manifest_2026-06-26.md`](fspt_package_path_helper_manifest_2026-06-26.md)

## Current status

The active project is not a validated SOTA FSPT tracker. It is in a re-entry / re-detection pivot state.

Closed routes:

- DINOv2 whole-frame / multi-anchor retrieval.
- DINOv2 local feature pseudo-label rollout.
- CT-offline-centered local refiner.
- CT-offline-centered grid verifier.
- Learned offset regression on frozen DINOv2 features.
- Any continued P4 training on the above route.

Active allowed directions:

- Route A: external stronger teacher evaluation.
- Route B: engineering foundation cleanup.
- Route C: analysis / benchmark paper if Route A does not produce a stronger teacher.

## What not to cite

Do not cite this repository as currently having:

- a reproducible FSPT SOTA model;
- a validated 67%+ TAP-Vid-DAVIS AJ result;
- a validated frequency-semantic tracker that beats CoTracker3 / TAPNext;
- a live DINO/local-refiner/verifier training route.

## Engineering note

The repository still contains legacy modules, draft experiments, and closed branches. New work should start from `CURRENT_MAINLINE.md` and the `fspt/` package scaffolding rather than from old paper-draft documents.
