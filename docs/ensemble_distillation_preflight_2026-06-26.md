# Ensemble distillation preflight -- 2026-06-26

## Purpose

This preflight gates the masked-median ensemble distillation run.  The goal is to prove that the exported ensemble cache, base_tracks injection, and distillation config are consistent before running a 20-epoch feasibility smoke.

## Current status

- Fixed best teacher: `cotracker3_offline`, `true_AJ_RD_256 = 0.5546`.
- Best deployable ensemble: `visibility_masked_median`, `true_AJ_RD_256 = 0.5871`.
- Oracle upper bound: `true_AJ_RD_256 = 0.6509`.
- Ensemble closes `33.8%` of the oracle gap.

## Required gates before 20-epoch training

1. Export consistency check passes:
   - exported file count equals cache record count;
   - every exported `base_tracks` equals source cache `pred_tracks` within tolerance;
   - every exported `base_visibility` equals source cache `pred_visibility`;
   - `query_points` match source cache.
2. Dataloader injection smoke passes with `base_tracks_strict=true`:
   - `base_tracks` and `base_visibility` appear in batch;
   - shapes are `(B,N,T,2)` and `(B,N,T)` after collation;
   - all base coordinates are finite.
3. Config is pure ensemble-distillation smoke:
   - DAVIS GT position/occlusion losses disabled (`weight=0.0`);
   - `base_track_consistency.mask=base_visible` so re-entry frames supervised by ensemble visibility are not skipped when the student predicts invisible.

## Non-claim

The DAVIS distillation config is not a held-out model-gain claim.  It is an in-domain feasibility smoke that answers whether the student can absorb the validated deployable ensemble behavior.
