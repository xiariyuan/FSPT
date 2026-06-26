# Uniform teacher ensemble baseline result — 2026-06-26

## Purpose

This note records the first deployable three-teacher ensemble baseline after the three-teacher oracle audit.

Unlike the oracle teacher selection result, these ensemble caches do not use ground-truth-derived per-event teacher choice at inference time.  They are deployable teacher-combination baselines built from the existing unified caches.

## Inputs

Teacher caches:

- `outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt`
- `outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt`
- `caches/trackon2_strided_original.pt`

Reference points:

- Fixed best teacher: `cotracker3_offline`
- Fixed best `true_AJ_RD_256`: `0.5546`
- Oracle teacher selection `true_AJ_RD_256`: `0.6509`
- Oracle gap: `0.0963`

## Script correction before running

`scripts/build_uniform_teacher_ensemble_cache.py` now supports explicit track aggregation modes:

- `mean`: legacy unmasked arithmetic mean.
- `nanmean`: unmasked finite-coordinate mean.
- `visibility_masked_mean`: mean over visible+finite teachers, with finite-coordinate fallback.
- `median`: unmasked finite-coordinate median.
- `visibility_masked_median`: median over visible+finite teachers, with finite-coordinate fallback.

The default is now `visibility_masked_mean`, because unmasked averaging can let stale or undefined coordinates from invisible teachers affect re-entry-frame coordinates.

## Commands

```bash
python scripts/build_uniform_teacher_ensemble_cache.py \
  --teacher-caches \
    cotracker3_online=outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt \
    cotracker3_offline=outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt \
    trackon2=caches/trackon2_strided_original.pt \
  --output-cache outputs/uniform_three_teacher_ensemble_masked_mean_2026-06-26.pt \
  --model-name uniform_three_teacher_masked_mean \
  --visibility-strategy majority \
  --track-aggregation visibility_masked_mean

python scripts/eval_aj_rd_from_cache.py \
  --cache-path outputs/uniform_three_teacher_ensemble_masked_mean_2026-06-26.pt \
  --output-json outputs/uniform_three_teacher_ensemble_masked_mean_2026-06-26_ajrd.json
```

The same command was repeated with `--track-aggregation visibility_masked_median` and output paths ending in `_masked_median_2026-06-26`.

## Result

| Method | Track aggregation | true_AJ_RD | true_AJ_RD_256 | Delta vs fixed 256 | Oracle gap used | Median px | <4px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed best teacher | n/a | 0.3870 | 0.5546 | 0.0000 | 0.0% | 3.71 | n/a |
| Legacy ensemble | unmasked mean | 0.4053 | 0.5817 | +0.0271 | 28.1% | 3.77 | 52.85% |
| Masked ensemble | visibility masked mean | 0.4103 | 0.5860 | +0.0314 | 32.6% | 3.77 | 53.14% |
| Masked robust ensemble | visibility masked median | 0.4107 | 0.5871 | +0.0325 | 33.8% | 3.74 | 53.50% |
| Oracle teacher selection | per-event max AJ_RD_256 | 0.4586 | 0.6509 | +0.0963 | 100.0% | n/a | n/a |

Per-d_min `true_AJ_RD_256` for the best deployable ensemble (`visibility_masked_median`):

```text
d_min=1:  0.5847
d_min=4:  0.5648
d_min=16: 0.4994
```

DAVIS coverage note:

```text
n_valid_samples_by_dmin_256: {1: 1385, 4: 1076, 16: 414}
n_eligible_events_by_dmin:   {1: 1863, 4: 1214, 16: 421, 64: 0, 256: 0}
```

The canonical summary still uses `true_AJ_RD_256`; `d_min=64` and `d_min=256` have no eligible DAVIS events in this current audit output.

## Interpretation

The corrected ensemble baseline is positive:

- It improves over the fixed best teacher by `+3.25pp` on canonical `true_AJ_RD_256`.
- It uses about `33.8%` of the oracle gap.
- The robust masked median is slightly better than masked mean, which suggests that some per-event teacher outliers exist but the gap is not explained by outliers alone.

Decision:

- Do not treat the oracle `+9.63pp` as a deployable gain.
- Treat the masked ensemble `+3.25pp` as the first deployable Route A gain.
- Next primary route: ensemble distillation from the corrected ensemble cache.
- Next parallel diagnostic: event-level teacher selector / confidence gate, because about two thirds of the oracle gap remains unclosed.
