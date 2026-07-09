# Uniform teacher ensemble result -- 2026-06-26

## Status

This is the reproducibility record for the first deployable three-teacher ensemble baseline on the canonical Route A metric.

Canonical metric:

```text
true_AJ_RD_256
```

Important caveat:

- Oracle teacher selection is an upper bound because it uses ground-truth-derived per-event teacher choice.
- Uniform / masked / median teacher ensembles are deployable teacher-combination baselines because they do not use ground truth at inference time.
- Distillation from DAVIS exported base tracks is a feasibility smoke unless evaluated on a held-out split or external dataset.

## Inputs

Teacher caches:

```text
outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt
outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt
caches/trackon2_strided_original.pt
```

Teacher pool:

```text
cotracker3_online
cotracker3_offline
trackon2
```

## Oracle reference

From `outputs/oracle_teacher_selection_2026-06-26.json`:

| Quantity | Value |
|---|---:|
| n_events | 1385 |
| fixed_best_teacher | cotracker3_offline |
| fixed_best true_AJ_RD | 0.3870 |
| fixed_best true_AJ_RD_256 | 0.5546 |
| oracle teacher selection true_AJ_RD | 0.4586 |
| oracle teacher selection true_AJ_RD_256 | 0.6509 |
| delta oracle vs fixed, true_AJ_RD | +0.0716 / +7.2pp |
| delta oracle vs fixed, true_AJ_RD_256 | +0.0963 / +9.6pp |

Oracle teacher usage is balanced enough to indicate real complementarity rather than a single dominant teacher:

```text
cotracker3_online:  419 / 1385 = 30.3%
cotracker3_offline: 512 / 1385 = 37.0%
trackon2:            454 / 1385 = 32.8%
```

## Ensemble construction commands

Legacy unmasked mean can be reproduced with:

```bash
python scripts/build_uniform_teacher_ensemble_cache.py \
  --teacher-caches \
    cotracker3_online=outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt \
    cotracker3_offline=outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt \
    trackon2=caches/trackon2_strided_original.pt \
  --output-cache outputs/uniform_three_teacher_ensemble_2026-06-26.pt \
  --model-name uniform_three_teacher \
  --visibility-strategy majority \
  --track-aggregation mean
```

Corrected masked median baseline:

```bash
python scripts/build_uniform_teacher_ensemble_cache.py \
  --teacher-caches \
    cotracker3_online=outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt \
    cotracker3_offline=outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt \
    trackon2=caches/trackon2_strided_original.pt \
  --output-cache outputs/uniform_three_teacher_ensemble_masked_median_2026-06-26.pt \
  --model-name uniform_three_teacher_masked_median \
  --visibility-strategy majority \
  --track-aggregation visibility_masked_median
```

Evaluation command:

```bash
python scripts/eval_aj_rd_from_cache.py \
  --cache-path outputs/uniform_three_teacher_ensemble_masked_median_2026-06-26.pt \
  --output-json outputs/uniform_three_teacher_ensemble_masked_median_2026-06-26_ajrd.json
```

## Results

| Method | Track aggregation | true_AJ_RD | true_AJ_RD_256 | Delta vs fixed 256 | Oracle gap used | Median px | <4px |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed best teacher | n/a | 0.3870 | 0.5546 | 0.0000 | 0.0% | 3.71 | n/a |
| legacy ensemble | unmasked mean | 0.4053 | 0.5817 | +0.0271 | 28.1% | 3.77 | 52.85% |
| masked ensemble | visibility_masked_mean | 0.4103 | 0.5860 | +0.0314 | 32.6% | 3.77 | 53.14% |
| masked robust ensemble | visibility_masked_median | 0.4107 | 0.5871 | +0.0325 | 33.8% | 3.74 | 53.50% |
| oracle teacher selection | per-event max AJ_RD_256 | 0.4586 | 0.6509 | +0.0963 | 100.0% | n/a | n/a |

Best deployable ensemble so far:

```text
visibility_masked_median true_AJ_RD_256 = 0.5871
```

## Per-d_min canonical breakdown

For the best deployable ensemble (`visibility_masked_median`):

```text
d_min=1:  0.5847
d_min=4:  0.5648
d_min=16: 0.4994
```

Coverage:

```text
n_valid_samples_by_dmin_256: {1: 1385, 4: 1076, 16: 414}
n_eligible_events_by_dmin:   {1: 1863, 4: 1214, 16: 421, 64: 0, 256: 0}
```

DAVIS has zero eligible events at `d_min=64` and `d_min=256` in this audit output, so paper-facing tables should avoid implying those bins are measured.

## Interpretation

The corrected masked-median ensemble provides the first deployable Route A gain:

```text
0.5546 -> 0.5871 = +0.0325 / +3.25pp
```

This closes about one third of the oracle gap:

```text
(0.5871 - 0.5546) / (0.6509 - 0.5546) = 33.8%
```

About two thirds of the oracle gap remains.  This supports two next steps:

1. Pure ensemble-distillation smoke: test whether a single student can absorb the deployable ensemble behavior.
2. Event-level teacher selector / confidence gate: test whether non-GT features can recover the remaining per-event teacher-routing gap.

## Distillation caution

The current distillation config uses DAVIS exported base tracks.  Any training-and-evaluation result on the same DAVIS videos is a feasibility smoke, not a held-out generalization claim.  A deployable paper claim requires a held-out split or cross-dataset validation.
