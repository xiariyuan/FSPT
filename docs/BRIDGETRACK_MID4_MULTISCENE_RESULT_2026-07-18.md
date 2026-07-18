# BridgeTrack Fixed Mid-Layer Multi-Scene Result — 2026-07-18

## Status

The frozen six-scene PointOdyssey fit-only verification is complete.  It is an
oracle architecture diagnostic, not a learned or deployable result.

## Frozen scene set

```text
r2_new_
character
seminar_h52_ego2
scene_carb_h_tables
r0_new_f_
dancing
```

The discovery scene `ani13_new_f` was excluded.  Every selected scene had at
least 64 points visible, valid, finite, and 64 px inside the image boundary for
all 56 protocol frames.

## Primary result

Replacing only query-token RG-LRU and Conv1D state in layers 4--7 produced:

```text
positive scenes:                  5 / 6
median future-error gain:         +1.4654 px at 256 raster
mean scene-bootstrap 95% CI:      [+0.4911, +31.7955] px
median retained full-query gain:  51.14%
median improved-frame fraction:   75.0%
worst scene gain:                 -0.0004 px
```

The frozen gate failed because:

```text
median gain < +2.0 px
median retained full-query gain < 80%
```

Decision:

```text
CLOSE_FIXED_MID4_QUERY_STATE_REPAIR
NO TRAINING
NO MODEL_VALIDATION
```

## Full-query causal signal

Same-time full teacher query-state replacement remained positive in all six
scenes:

```text
positive scenes:              6 / 6
median gain:                  +3.4759 px
scene-bootstrap 95% CI:       [+1.6482, +45.8074] px
mean excluding largest scene: +2.5430 px
```

This means the Query-State hypothesis itself is not rejected.  What is rejected
is the claim that layers 4--7 are a universal repair target.  The causal repair
state is distributed differently across scenes.

## Allowed next step

The only allowed continuation is a fit-only low-rank repair-subspace oracle:

1. all already exposed fit scenes form the basis-development set;
2. previously unused qualified fit scenes form an internal verification set;
3. fit a block-normalized PCA/SVD basis to teacher-minus-student query-state
   deltas;
4. use oracle projection coefficients only to test whether a compact state
   subspace can retain future rollout gain;
5. no learned coefficient predictor may be trained until that oracle gate
   passes.

PointOdyssey model-validation, internal holdout, test, DAVIS method evaluation,
and Kinetics 1,144 remain unread.
