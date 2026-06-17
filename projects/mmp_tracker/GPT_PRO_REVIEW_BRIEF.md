# GPT Pro Review Brief: MMP Tracker Restart Route

## Goal

We pivoted away from the legacy FSPT residual-refiner route and rebuilt a new branch under `projects/mmp_tracker/`.

The new paper direction is:

> explicit local matching + visible-memory global relocalization + optional posterior fusion
> for long-occlusion point tracking.

We need a deep architectural review focused on whether this route is structurally sound, what should be simplified, and what should be changed next before committing more training time.

## Current branch layout

- main trainer: `projects/mmp_tracker/train_mmp.py`
- core config dataclasses: `projects/mmp_tracker/mmp_tracker/config.py`
- core model: `projects/mmp_tracker/mmp_tracker/model.py`
- local matching: `projects/mmp_tracker/mmp_tracker/local_matcher.py`
- memory/global relocalization: `projects/mmp_tracker/mmp_tracker/global_relocator.py`
- posterior fusion: `projects/mmp_tracker/mmp_tracker/posterior_fusion.py`
- memory bank: `projects/mmp_tracker/mmp_tracker/memory_bank.py`
- losses: `projects/mmp_tracker/mmp_tracker/losses.py`
- first-frame query adapter: `projects/mmp_tracker/mmp_tracker/data.py`

## What already changed in this iteration

### Latest protocol / supervision fixes in the newest package

These were added after the previous review round and should be treated as the current mainline state:

1. Validation protocol is no longer silently capped to 128 points.
   - `projects/mmp_tracker/train_mmp.py`
   - `resolve_dataset(...)` now only applies `max_points` if explicitly configured.

2. Evaluation no longer hard-codes `query_mode="first"`.
   - `projects/mmp_tracker/train_mmp.py`
   - eval now resolves query mode from the batch / dataset and should better match official TAP-Vid protocol semantics.

3. Training loader now shuffles when appropriate.
   - `projects/mmp_tracker/train_mmp.py`

4. The local/global selector now has a straight-through gate.
   - `projects/mmp_tracker/mmp_tracker/model.py`
   - forward remains a hard routing decision, but gradients can flow through selector probability during training.

5. Added explicit global coordinate supervision.
   - `projects/mmp_tracker/mmp_tracker/losses.py`
   - in addition to global heatmap CE, the global top-1 coordinate now receives direct regression-style supervision on frames where global relocalization is actually relevant.

6. Selector supervision is now focused on global-needed frames.
   - `projects/mmp_tracker/mmp_tracker/losses.py`
   - selector BCE is no longer applied broadly to all visible points.

7. `no_harm_prior` is now closer to true no-harm behavior.
   - `projects/mmp_tracker/mmp_tracker/losses.py`
   - it compares against the better of temporal prior and local branch rather than accidentally behaving like `no_harm_local` only.

8. `global_coordinate` is no longer a dead supervision term.
   - `projects/mmp_tracker/mmp_tracker/global_relocator.py`
   - `projects/mmp_tracker/mmp_tracker/losses.py`
   - the global branch now exposes differentiable `expected_points` from the full-frame heatmap, and `global_coordinate` supervises that path instead of non-differentiable top-k cell centers.

9. Template / memory writes now use a safer policy.
   - `projects/mmp_tracker/mmp_tracker/model.py`
   - selected global jumps no longer automatically poison the tracker state; writes are gated by local-vs-global agreement in pixel space.

10. Long-occlusion subset metrics are now point-level micro averages rather than video-level macro averages.
    - `projects/mmp_tracker/train_mmp.py`
    - this makes `AJ_longocc*`, `OA_longocc*`, and reappearance-error diagnostics better aligned with actual point counts.

11. Dataset pixel-to-normalized coordinate conversion is now aligned with `align_corners=True` semantics.
    - `datasets/coord_utils.py`
    - `datasets/tapvid_davis.py`
    - `datasets/tapvid_kinetics.py`
    - `datasets/tapvid_kubric.py`
    - `datasets/tapvid_kinetics_sharded.py`
    - `datasets/tapvid_kubric_sharded.py`
    - corner points now map consistently to normalized `1.0` rather than being biased inward by `/H` and `/W` normalization.

12. Dev training on sharded Kubric no longer reuses a fixed shard/sample prefix every epoch by default.
    - `datasets/tapvid_kubric_sharded.py`
    - shard order and within-shard sample order now reshuffle across iterations for train split, which matters when using small `subset` limits like `256`.

13. Evaluation now defaults to the actual model input raster rather than original pre-resize video size.
    - `projects/mmp_tracker/train_mmp.py`
    - this keeps mainline reporting aligned with resized eval settings such as DAVIS `256x256`.

14. Training now saves a long-occlusion-aware best checkpoint in addition to overall AJ best.
    - `projects/mmp_tracker/train_mmp.py`
    - `best_longocc.pth` is selected by a simple long-occ composite rather than overall AJ only.

15. The `localglobal` mainline now uses coarse-global retrieval followed by same-frame local rematch and a confirmed commit rule.
    - `projects/mmp_tracker/mmp_tracker/model.py`
    - `global_points` now refer to the rematched global candidate rather than raw coarse top-1 cells.
    - recurrent state updates are no longer tied directly to every selected global jump; commit is gated by global quality and coarse-to-refine consistency.

16. A recipe-matched `local_dev` baseline config now exists for fair `Local` vs `LocalGlobal` comparison.
    - `projects/mmp_tracker/configs/local_dev.yaml`

17. Eval now exposes routing diagnostics for mechanism audits.
    - `projects/mmp_tracker/train_mmp.py`
    - current logs can report `selected_global_rate`, `commit_rate`, and `commit_given_selected` in addition to AJ / OA / long-occ metrics.

### Structural fixes already applied

1. Template is no longer fully static.
   - We now keep an `anchor_feat` and a momentum-updated `template_feat`.
   - See `projects/mmp_tracker/mmp_tracker/model.py`.

2. Global relocalization no longer averages all memory descriptors before retrieval.
   - It now performs slot-wise matching and aggregates logits with visibility/recency weighting.
   - See `projects/mmp_tracker/mmp_tracker/global_relocator.py`.

3. Visibility is partly decoupled from raw matching confidence.
   - Added a separate `visibility_predictor` in the model.
   - See `projects/mmp_tracker/mmp_tracker/model.py`.

4. Several configured losses that were previously inactive are now wired up.
   - local heatmap supervision
   - long-occlusion focus mask
   - no-harm prior
   - gradient accumulation
   - See `projects/mmp_tracker/mmp_tracker/losses.py` and `projects/mmp_tracker/train_mmp.py`.

5. Coordinate loss now uses input-resolution-scaled error rather than raw normalized-space error.
   - See `projects/mmp_tracker/mmp_tracker/losses.py`.

## Current configuration status

### Main branch to train now

- `projects/mmp_tracker/configs/localglobal_dev.yaml`

### Branch currently unstable and not recommended for long run

- `projects/mmp_tracker/configs/posterior_dev.yaml`

`posterior_dev` still collapses badly in smoke checks. We need GPT Pro to judge whether the posterior branch should be redesigned, delayed, or removed from the main paper path.

## Latest smoke / env-check observations

### Before these structural fixes

- `localmatch` was roughly AJ ~ 0.01 and OA ~ 0.26.
- `localglobal` was near AJ ~ 0.00 and OA ~ 0.17 on tiny env-checks.

### After these structural fixes

#### Local-only env-check

- AJ ~ 0.0811
- OA ~ 0.7424
- avg_error_px ~ 32.74

#### LocalGlobal env-check

- AJ ~ 0.1135
- OA ~ 0.8276
- avg_error_px ~ 18.89

#### Posterior env-check

- still unstable / collapsed
- AJ = 0.0
- OA ~ 0.17
- avg_error_px extremely high

Interpretation:

- the local + global branch appears much healthier than before
- the posterior branch is still not trustworthy

## Data context

Server now has:

- existing `Kubric` preprocessed training data
- newly uploaded `MegaDepth-1500` + `scene_info`
- `YouTube-VOS 2019 train/valid`
- `RoboTAP`
- `TAP-Vid DAVIS`
- `TAP-Vid RGB-Stacking`

Immediate training does not yet consume all of these new datasets, but the data budget is now available for the next stages.

## Existing paper plan docs

- `projects/mmp_tracker/PAPER_EVIDENCE.md`
- `projects/mmp_tracker/EXPERIMENT_MATRIX.md`
- `projects/mmp_tracker/TRAINING_STAGES.md`
- `projects/mmp_tracker/DOWNLOAD_MANIFEST.md`

## What we want GPT Pro to answer

1. Is the decomposition `local matching -> memory relocalization -> posterior fusion` the right architecture ladder?
2. Should posterior fusion be redesigned as:
   - gated residual over `prior/local`,
   - candidate classifier,
   - energy minimization,
   - or removed from the main path for now?
3. Is the current local matcher supervision sufficient, or should it be changed to a stronger patch-classification / contrastive objective?
4. Should the memory bank remain per-point visible memory, or become a learned key-value bank with better slot selection?
5. Is the paper’s main contribution better framed around:
   - long-occlusion recovery,
   - matching-first tracking decomposition,
   - or a narrower first-frame tracking setting first?
6. What is the next highest-ROI change before spending many epochs on training?

## Recommended immediate execution path

If no structural objection is found, the current recommended mainline is:

1. train `localglobal_dev`
2. validate whether long-occ metrics become meaningfully positive
3. keep posterior as a separate redesign branch
4. only add frequency or extra modules after the local/global story is solid

## Important caveat

The current branch still assumes first-frame queries / forward-only tracking in the new scaffold.
That is a deliberate simplification for engineering progress, but it may need to be generalized later depending on the final benchmark target.
