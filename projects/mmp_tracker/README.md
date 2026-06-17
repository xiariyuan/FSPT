# MMP-Tracker

MMP-Tracker is a fresh research branch for point tracking built around four ideas:

- local explicit matching instead of direct coordinate regression
- visible-memory global re-localization for long occlusion recovery
- posterior fusion over prior, local evidence, and global evidence
- a clean ablation ladder that can support a paper narrative

This branch is intentionally isolated from the legacy FSPT and refiner routes.

## Model ladder

- `LocalMatch`: local matching only
- `LocalMatch + GlobalReloc`: adds visible-memory re-localization
- `LocalMatch + GlobalReloc + PosteriorFusion`: adds hypothesis fusion
- `Full`: reserves space for frequency-guided evidence reweighting later

## Current scope

This initial commit provides:

- a standalone Python package under `projects/mmp_tracker/mmp_tracker`
- a first-pass implementation of the core modules
- a paper evidence plan in `projects/mmp_tracker/PAPER_EVIDENCE.md`
- a shape sanity test in `tests/test_mmp_tracker_shapes.py`

## Short-term goal

The first engineering milestone is not benchmark SOTA. It is to establish a clean,
trainable, matching-first baseline with a defensible ablation path.

