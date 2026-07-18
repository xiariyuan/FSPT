# Route-D Safe Re-detection v0 Formal Result — 2026-07-18

## Status

The frozen primary and independent seed-17 replay are complete.

```text
source commit:  d7598718763ac24fb6a143f7e34447047198658a
fit events:     240
validation:     160 events
best epoch:     -1 (zero-step native state)
```

## Formal result

```text
native AJ_RD:                 0.21193
selected AJ_RD:               0.21193
direct AJ_RD gain:            0.00000
zero-step joint oracle gain: +0.17977
visible interventions:        0
harmful interventions:        0
```

Training history:

```text
epoch 0 loss 3.33119, direct gain 0, oracle gain +0.08813
epoch 1 loss 1.62307, direct gain 0, oracle gain +0.08541
epoch 2 loss 1.32368, direct gain 0, oracle gain +0.08498
```

The model converged to permanent abstention.  Proposal training also reduced the
candidate oracle relative to the zero-step proposal.

## Exact replay

After removing the expected absolute `checkpoint_path` string, primary and
replay `metrics.json` are exactly equal.  The two checkpoint files are also
byte-identical.

```text
checkpoint SHA256:
3f0b59cc4a3e76c013d1210eeddcf68e9bff290334379d32fb0d11e2cb1d8750

model-state SHA256:
2477dc2ef67303048532e83588daff876a7d7005f170508f2f8ee7e124a9f3d2
```

## Gate decision

```text
candidate-oracle gate:              PASS
direct AJ_RD gain gate:             FAIL
paired CI lower-positive gate:      FAIL
all safety/non-regression gates:     PASS
overall:                             FAIL

STOP_SAFE_REDETECTION_BEFORE_LOCKED_DATA
```

PointOdyssey internal holdout, PointOdyssey test, DAVIS method evaluation, and
the frozen 1,144-video Kinetics experiment were not read or rerun.

This closes Safe Re-detection v0.  Its outputs remain a negative baseline and
motivation for the separate BridgeTrack recurrent query-state investigation.
