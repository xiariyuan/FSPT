# Route-D two-stage causal top-1 Gate 3C1D v1 — 2026-07-20

## Status

```text
PREREGISTERED_NOT_RUN
```

## Parent authorization

Gate 3C1D v0 rejected a single candidate-support probability used simultaneously
as ranking score and action confidence. Gate 3C2 then qualified 512 additional
raw-record-disjoint Kubric videos and authorized a two-stage redesign:

```text
AUTHORIZE_GATE3C1D_V1_TWO_STAGE_TOP1_PREREGISTRATION
```

Gate 3C2 is a data-identity result. No renewed candidate feature, selector score,
or policy metric was read while designing this protocol.

## Mechanism diagnosis

The design pool consists only of the already exposed development caches:

| partition | failure rows |
|---|---:|
| gradient train 64--383 | 3,325 |
| old checkpoint 384--447 | 631 |
| old fit-only audit 448--511 | 753 |
| old original-model validation 48--63 | 192 |
| **total** | **4,901** |

A four-fold outer source-video split with three-fold inner OOF candidate
predictions compared two ranking rules. Candidate AUC/AP were 0.8483/0.4549.
Expected-distance ranking reached 35.6662% raw top-1 12-pixel support and reduced
raw harmful selection from 10.3040% under probability argmax to 8.0800%.

Value and harm gates alone produced zero passing strategies. Adding a separate
12-pixel support condition produced a valid policy region. Under expected-distance
ranking the highest-coverage design-only passing point was:

```text
support probability >= 0.30
predicted value      >= 0 px
predicted harm       <= 0.20
coverage              = 31.7282%
action precision      = 65.6592%
mean error reduction  = 3.9970 px
video CI               = [3.6819, 4.3937] px
all-row harmful rate   = 0.7958%
```

These values motivate factorization and gate ranges only. They are not formal
renewed checkpoint results and are not copied as the final policy.

Machine-readable design evidence:

`docs/generated/ROUTED_TEMPORAL_IDENTITY_TOP1_GATE3C1D_V1_DESIGN_DIAGNOSTIC_2026-07-20.json`

## Frozen model factorization

Four fixed HistGradientBoosting models are trained:

1. **candidate support classifier** — predicts whether a shortlist candidate is
   within 12 pixels at commit;
2. **candidate expected-distance regressor** — predicts clipped commit distance
   and supplies the stable top-1 ranking;
3. **row value regressor** — predicts native minus proposed-candidate error;
4. **row harm classifier** — predicts whether the proposal exceeds native error
   by more than four pixels.

The candidate models use the frozen 102-D causal shortlist representation. The
row models use a 164-D label-free representation containing candidate prediction
summaries plus selected/native/difference causal fields. Teacher distance and
frames 16--23 are absent from every model input.

All model hyperparameters, `early_stopping=False`, random seeds, feature shapes,
and runtime versions are fixed in:

`configs/routeD_temporal_identity_top1_gate3c1d_v1.yaml`

## Strict OOF contract

Row-model training must not consume in-sample candidate predictions.

```text
all 4,901 old exposed development rows
    -> five source-video GroupKFold splits
    -> OOF support and expected-distance predictions
    -> OOF selected candidate and 164-D row features
    -> fit value and harm models
    -> refit final candidate models once on all 4,901 rows
```

The renewed 512-video population is forbidden from every model fit. It is used
only for policy selection and unchanged confirmation.

## Renewed data sequence

```text
checkpoint selection v1: 0--255   (256 videos)
fit-only audit v1:       256--383 (128 videos)
model validation v1:     384--511 (128 videos)
```

Each feature cache requires exact manifest membership, sidecar hashes, frozen
backbone hashes, source-code hashes, and explicit read-state verification.
Audit cache creation is blocked until checkpoint replay authorizes it. Model
validation cache creation is blocked until audit replay authorizes it.

## Frozen policy grid

Checkpoint selection evaluates exactly 480 combinations:

```text
support minimum:
[0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65]

predicted value minimum, pixels:
[0,1,2,3,4,5,6,8]

predicted harm maximum:
[0.03,0.05,0.075,0.10,0.15,0.20]
```

A proposal is committed only when it is non-native and passes all three
conditions. Otherwise output is exact native candidate zero.

Among policies passing every checkpoint gate, selection order is frozen as:

1. highest action coverage;
2. lower harm threshold;
3. higher support threshold;
4. higher value threshold.

No threshold, model, feature, or ordering may change after checkpoint results are
read.

## Checkpoint gates

All conditions are required:

```text
candidate AUC                                  >= 0.78
candidate AP                                   >= 0.30
raw top-1 12 px support                        >= 0.32
action coverage                                >= 0.20
action precision within 12 px                  >= 0.65
mean commit-error reduction                    >= 2.0 px
video-cluster CI lower                         >= 1.0 px
all-row harmful rate                           <= 0.02
action-conditional harmful rate                <= 0.08
nonnegative-video fraction                     >= 0.70
fresh-process exact replay                     required
```

```text
pass -> AUTHORIZE_GATE3C1D_V1_FIT_ONLY_AUDIT
fail -> STOP_GATE3C1D_V1_WITHOUT_AUDIT
```

## Audit and model-validation gates

The frozen bundle and policy must pass unchanged on both later partitions:

```text
candidate AUC                                  >= 0.75
candidate AP                                   >= 0.25
raw top-1 12 px support                        >= 0.30
action coverage                                >= 0.15
action precision within 12 px                  >= 0.60
mean commit-error reduction                    >= 1.5 px
video-cluster CI lower                         >= 0.0 px
all-row harmful rate                           <= 0.025
action-conditional harmful rate                <= 0.10
nonnegative-video fraction                     >= 0.65
fresh-process exact replay                     required
```

```text
checkpoint pass -> audit
checkpoint fail -> stop without audit
audit pass      -> model validation
audit fail      -> stop without model validation
validation pass -> AUTHORIZE_GATE3C1E_CAUSAL_TOP1_FUTURE_ROLLOUT_PREREGISTRATION
validation fail -> STOP_GATE3C1D_V1_AT_MODEL_VALIDATION
```

## Exact replay

Checkpoint replay independently retrains all four models. Because sklearn joblib
serialization contains non-scientific byte variation, exact replay does not
require replay bundle files to be byte-identical. It requires equality of:

- OOF row-feature, support, expected-distance, and selected-slot digests;
- development feature and shortlist digests;
- all target support, expected-distance, selected-slot, row-feature, value, harm,
  and output-slot digests;
- frozen policy and complete policy-grid digest;
- target cache-index hashes, point records, metrics, and scientific payload.

Audit and model validation load only the hash-pinned primary bundle produced by
the checkpoint primary run.

Runtime versions are frozen to Python 3.11.8, NumPy 1.26.4, PyTorch
2.2.2+cu121, scikit-learn 1.4.2, and joblib 1.4.2.

## Claim boundary

Gate 3C1D v1 is a commit-level causal top-1 selector gate. A pass would not yet
establish future tracking improvement. Coordinate-plus-memory future rollout
under the causal policy is reserved for Gate 3C1E.

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked throughout Gate 3C1D v1.
