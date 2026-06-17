# Top-k Candidate Mainline Update (2026-03-09)

## Decision

- Kill `top1 rematch + deferred commit` as the mainline.
- Keep `local` as the locked internal baseline.
- Promote `top-k retrieval + per-candidate rematch + scorer/local-abstain` as the new mainline.
- Keep recurrent commit out of v1 until same-frame candidate reasoning proves real gain.

## What changed in code

- `projects/mmp_tracker/mmp_tracker/model.py`
  - Added `localglobal_topk` variant.
  - Added vectorized per-candidate rematch for all global top-k proposals.
  - Added multi-candidate scorer over `{local + K refined global candidates}`.
  - Output selection now uses scorer + local abstain.
  - Recurrent state remains local-only in this first top-k stage.

- `projects/mmp_tracker/mmp_tracker/losses.py`
  - Upgraded selector supervision from binary BCE to multi-class candidate CE when candidate logits are present.
  - Preserved the old binary fallback for legacy branches.

- `projects/mmp_tracker/train_mmp.py`
  - Added oracle coarse and oracle rematch evaluation from candidate sets.
  - Added candidate-level routing metrics such as `candidate_global_better_rate` and `candidate_selected_oracle_rate`.

## New configs

- Dev: `projects/mmp_tracker/configs/localglobal_topk_abstain_dev.yaml`
- Full/night: `projects/mmp_tracker/configs/localglobal_topk_abstain_night_full.yaml`

## New matrix entrypoint

- `bash projects/mmp_tracker/run_topk_candidate_matrix.sh 42 43`

This runs the next decisive comparison ladder:

1. `local_dev`
2. `localglobal_nocommit_dev`
3. `localglobal_topk_abstain_dev`

## Immediate evidence targets

We care about:

- `AJ_longocc20`
- `AJ_longocc30`
- `reapp_error_longocc20_px`
- `reapp_error_longocc30_px`
- `oracle_coarse_AJ_longocc20/30`
- `oracle_rematch_AJ_longocc20/30`
- `candidate_global_better_rate`
- `candidate_selected_oracle_rate`

## Stop rules

- If `oracle_rematch` shows weak headroom over `local`, kill the current retrieval-mechanism story on this backbone.
- If oracle headroom exists but `topk_abstain` still cannot beat `local` and `nocommit`, kill the current scorer route on this backbone.
- Only after scorer/no-commit works do we revisit any state-update or commit design.
