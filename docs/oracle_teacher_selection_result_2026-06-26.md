# Oracle teacher selection result — 2026-06-26

## Purpose

This note records the first metric-clean three-teacher oracle selection result under the unified strided+original DAVIS protocol.

The result is a diagnostic upper bound.  It is not a deployable model result.

## Teacher pool

- `cotracker3_online`
- `cotracker3_offline`
- `trackon2`

Inputs used in the local run:

- `outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt`
- `outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt`
- `caches/trackon2_strided_original.pt`

Output JSON:

- `outputs/oracle_teacher_selection_2026-06-26.json`

## Canonical metric

Paper-level metric: `true_AJ_RD_256`.

The canonical oracle delta is same-space:

```text
oracle_teacher_selection_by_max_ajrd_256 - fixed_best_256
```

This was fixed in commit `96dadc7`; commit `2e09e1d` fixed the CLI summary-print typo afterward.

## Result

```text
n_events: 1385
fixed_best_teacher: cotracker3_offline
fixed_best_median_px: 3.71

fixed_best_original: 0.3870
fixed_best_256: 0.5546
oracle_teacher_selection_by_max_ajrd:     0.4586
oracle_teacher_selection_by_max_ajrd_256: 0.6509

delta_max_ajrd_vs_fixed:    +0.0716 / +7.2pp
delta_max_ajrd256_vs_fixed: +0.0963 / +9.6pp
decision: STRONG_DIAGNOSTIC_HEADROOM
selection_metric: true_AJ_RD_256
```

Teacher median errors:

```text
cotracker3_online:  3.86 px
cotracker3_offline: 3.71 px
trackon2:           3.91 px
```

Oracle teacher usage:

```text
cotracker3_online:  419 events / 30.3%
cotracker3_offline: 512 events / 37.0%
trackon2:           454 events / 32.8%
```

## Metric-space and d_min coverage note

The oracle headroom is metric-space dependent.  Original-resolution `true_AJ_RD` shows `+7.2pp`, while canonical 256-space `true_AJ_RD_256` shows `+9.6pp`.  The paper-facing decision uses `true_AJ_RD_256`, but both numbers should be reported when discussing headroom.

Current DAVIS eligible coverage is concentrated in `d_min={1,4,16}`; `d_min=64` and `d_min=256` have zero eligible events in this audit output.

## Interpretation

The three teachers have real complementarity: per-event oracle selection is +9.6pp over the fixed best teacher on `true_AJ_RD_256`.

This does **not** mean a deployable system already gains +9.6pp.  The oracle uses ground-truth-derived per-event teacher choice.  The next question is whether this complementarity can be approximated by a deployable method.

Immediate follow-up baselines:

1. Corrected visibility-masked uniform/robust ensemble of the existing three teacher prediction caches.
2. If uniform ensemble improves: consider ensemble distillation.
3. If uniform ensemble does not improve but oracle remains strong: prioritize a teacher selector / confidence gate.
4. External teachers such as Track-On-R can be added, but their result must be separated into single-teacher, ensemble, and oracle-upper-bound gains.
