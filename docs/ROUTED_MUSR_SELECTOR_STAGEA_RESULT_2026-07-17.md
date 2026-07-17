# Route-D MUSR Stage-A selector result — 2026-07-17

## 1. Decision

```text
FAIL the selector feasibility gate.
STOP before calibration.
STOP before state-write training.
REDESIGN candidate representation; do not tune thresholds or writeback strength.
```

The frozen candidate generator has large oracle headroom, but the current
64-dimensional handcrafted/binned candidate representation cannot identify the
useful candidate often enough to produce a material AJ gain.

## 2. Data and protocol isolation

Completed candidate caches:

| Partition | Videos | Coordinate-oracle AJ gain | Use in Stage A |
|---|---:|---:|---|
| fit | 48 | +6.0778 | gradient updates |
| model validation | 16 | +6.8463 | checkpoint selection |
| calibration | 32 | +6.8410 | **exported but not read** |

The candidate qualification gate had already passed on a separate 15-video
partition. The final synthetic holdout, TAP-Vid-DAVIS, and the corrected
1,144-sample TAP-Vid-Kinetics evaluation were not read.

## 3. Structural corrections before the formal run

An initial end-to-end smoke exposed two implementation mismatches:

1. random scoring heads selected arbitrary non-native candidates at zero step;
2. training used soft candidate mixtures while deployment used hard argmax.

These were corrected structurally, not by threshold or learning-rate tuning:

- utility/risk/selector final layers are zero-initialized;
- deterministic argmax therefore selects native candidate 0 at zero step;
- initial abstention is high and coordinate write strength is conservative;
- zero-step official metrics are exactly equal to native;
- state-write training uses straight-through hard selection;
- non-deterministic memory-efficient attention kernels are disabled.

The formal Stage-A selector then froze the abstention and state-write heads
entirely. It trained only the candidate-set encoder and utility/risk selector.

## 4. Stage-A objective

Candidate ranking was aligned to weighted `1/2/4/8/16 px` hit utility rather
than minimum continuous Euclidean error. Utility ties deterministically fall
back to native candidate 0. A native-vs-global gate penalizes selecting a global
candidate whose true threshold utility is below native.

This stage evaluates the **raw selected coordinate**. No learned write strength
can hide a bad ranking decision.

## 5. Formal seed-17 result

The run requested 10 epochs with patience 4 and stopped after epoch 6. The best
checkpoint was epoch 2.

| Metric | Native | Selected | Gain |
|---|---:|---:|---:|
| AJ | 25.0135 | 25.0308 | **+0.0173 point** |
| Delta average | 37.1739 | 37.4171 | +0.2432 point |
| OA | 84.9784 | 84.9784 | 0 |
| <1 px | 10.8573 | 10.9248 | +0.0676 point |
| <2 px | 19.2643 | 19.4363 | +0.1719 point |
| <4 px | 31.6998 | 32.0007 | +0.3009 point |
| <8 px | 50.0553 | 50.4360 | +0.3807 point |
| <16 px | 73.9929 | 74.2876 | +0.2948 point |

Paired-video AJ gain:

```text
mean:  +0.0209 AJ point
95% CI: [-0.0092, +0.0537]
positive videos: 8 / 16
per-video median: approximately +0.0028 AJ point
```

The 16px severe-error rate improves from `0.25878` to `0.25583`, but this tail
improvement does not translate into a material or statistically established AJ
gain.

## 6. Selector behavior

```text
selected non-native rate:       4.5116%
oracle utility match rate:     61.8296%
mean threshold-utility regret:  0.08930
harmful global selection rate:  0.3395%
```

The high utility-match number is dominated by rows where native is already the
best utility candidate or ties for best. Coverage of useful non-native actions
is too low. Increasing coverage later in training causes more harmful decisions
and validation AJ declines.

## 7. Reproducibility

The complete seed-17 run was repeated independently. Both runs produced exactly
the same:

- best epoch;
- complete validation metrics and per-video rows;
- bootstrap confidence intervals;
- gate decision;
- model-state SHA-256.

```text
model-state SHA-256:
9744e3215111ddfbb7f5c0b8bd9413c3940dc8287422c502aef3b7f359b94179

primary checkpoint SHA-256:
cb650cac24d0a89081e4bb3b67b849d5e430af1d6727fc0362969e5f2be4fe9f

primary metrics SHA-256:
36afe88738457d182276bf88742877cdbcd85ffc023d94f960367af4b0de3f06
```

## 8. Gate result

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Raw selector AJ gain | >= +0.5 | +0.0173 | FAIL |
| Paired AJ CI lower | > 0 | -0.0092 | FAIL |
| Delta gain | > 0 | +0.2432 | PASS |
| Severe 16px rate | not worse | improved | PASS |
| Oracle-utility match | >= 0.25 | 0.6183 | PASS |

Final emitted decision:

```text
STOP_STAGE_B_AND_REDESIGN_CANDIDATE_REPRESENTATION
```

## 9. Scientific interpretation

The failure is not lack of coordinate reachability: model-validation oracle
headroom is `+6.8463 AJ`. It is not primarily uncontrolled state feedback,
because Stage A performs raw coordinate selection with writeback disabled. The
remaining bottleneck is candidate discrimination under the compressed
representation.

This agrees with earlier negative evidence from handcrafted CoTracker summaries
and fixed-topK reranking. Continuing with more epochs, threshold sweeps, gate
margins, or writeback tuning would repeat a closed route.

## 10. Next allowed route

Keep candidate coordinates and their generator unchanged. Change only the
candidate/state representation:

- full 128-D query support feature;
- full 128-D current candidate feature;
- full 128-D feature difference;
- full 128-D elementwise product;
- candidate-centered local correlation patch;
- full 128-D `online_track_feat` state;
- existing structured features retained as auxiliary inputs.

Run the same selector-only Stage-A gate on fit/model-validation. State write,
calibration, final holdout, DAVIS, and Kinetics remain locked.
