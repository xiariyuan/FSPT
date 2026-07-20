# Route-D causal top-1 selector Gate 3C1D v0 — 2026-07-20

## Status

```text
COMPLETED_FAIL
```

## Why Gate 3C1D is separate from shortlist retention

Gate 3C1B proved that query closure plus DINO identity retains a much better
native-plus-eight shortlist. Gate 3C1C proved that a correct retained coordinate
becomes useful only when the four CoTracker memory levels are reinstated.
Neither gate solves the deployable decision: choose one retained candidate
without a teacher, or abstain to exact native state.

A direct argmax of the fixed query-closure score is not sufficient. On the
training-only population it reaches only 9.5338% 12-pixel top-1 support, while
the same shortlist oracle reaches 59.3083%. Score margin and top-two gap have
approximately random success discrimination. Gate 3C1D therefore trains an
explicit candidate-success probability and keeps the analytic score as a
shortlist mechanism rather than misusing it as a top-1 confidence.

## Data sequence

```text
gradient train 64--383
  -> fit the fixed model once
checkpoint selection 384--447
  -> choose one threshold from a frozen grid
fit-only audit 448--511
  -> one-shot confirmation, no adjustment
original model validation 48--63
  -> final unchanged confirmation
```

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked. No top-1 metric from checkpoint, audit, or original model validation has
been observed before this protocol is committed.

## Frozen causal feature contract

For every candidate in the frozen native-plus-eight shortlist, construct 102
float32 fields:

1. 14 frozen static/list channels;
2. query-identity and previous-frame-identity values at the original query frame;
3. normalized query-frame index;
4. mean, minimum, maximum, and standard deviation for temporal channels
   0, 1, 2, 3, 6, 7, and 8;
5. velocity path length, acceleration mean/max, and jerk mean;
6. candidate-relative z-scores of all 49 raw fields;
7. frozen query-closure score, its shortlist-relative z-score, shortlist
   position, and native-role indicator.

Teacher distance and frames 16--23 are not accepted by the feature interface.
The shortlist and feature tensors freeze before the 12-pixel labels are read.

## Frozen implementation authority

The config pins SHA256 values for the analytic shortlist implementation, the
102-D causal feature implementation, and the complete Gate 3C1D runner. The
runner verifies these hashes before loading any cache sidecar. Formal artifacts
also record Python, NumPy, PyTorch, scikit-learn, and joblib versions.

## Frozen model

```text
HistGradientBoostingClassifier
max_iter             = 250
learning_rate         = 0.05
max_leaf_nodes        = 31
l2_regularization     = 1.0
min_samples_leaf      = 30
random_state          = 17
class reweighting     = none
label                 = commit distance <= 12 px
```

The model is fit exactly once on all gradient-train candidate rows. There is no
epoch selection, architecture sweep, ensemble choice, or checkpoint-dependent
model change.

A group-held gradient-train design probe motivated this fixed family: candidate
AUC/AP 0.8489/0.4926 and raw top-1 12-pixel support 38.8406%. This is design
evidence only. The probe used a temporary float16 feature copy; formal runs use
the committed float32 feature implementation and reproduce from source
sidecars.

## Native-safe action

The model scores all nine candidates. The candidate with maximum predicted
12-pixel probability is proposed. A non-native coordinate is committed only if:

1. the proposed candidate is non-native; and
2. its probability is at least the frozen threshold.

Otherwise the output is candidate zero and no state field changes.

## Frozen threshold selection

Checkpoint selection tests only:

```text
[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
```

Choose the lowest threshold passing all of:

```text
action coverage                                  >= 0.20
action precision within 12 px                    >= 0.65
mean commit-error reduction vs native            >= 2.0 px
video-cluster error-reduction CI lower           >= 1.0 px
all-row harmful rate (> native + 4 px)            <= 0.02
```

If no threshold passes, Gate 3C1D stops. The threshold is never changed on audit
or original model validation.

## Confirmation gates

Both fit-only audit and original model validation require:

```text
action coverage                                  >= 0.10
action precision within 12 px                    >= 0.60
mean commit-error reduction vs native            >= 2.0 px
video-cluster error-reduction CI lower           >= 0.0 px
all-row harmful rate (> native + 4 px)            <= 0.025
candidate AUC                                    >= 0.75
candidate AP                                     >= 0.25
exact fresh-process replay                        required
```

```text
checkpoint pass -> AUTHORIZE_GATE3C1D_FIT_ONLY_AUDIT
audit pass      -> AUTHORIZE_GATE3C1D_ORIGINAL_MODEL_VALIDATION
validation pass -> AUTHORIZE_GATE3C1E_END_TO_END_ROLLOUT_PREREGISTRATION
```

## Exact replay

Each stage freezes and compares:

- causal feature and shortlist digests;
- all candidate probabilities;
- selected output slots;
- threshold, the complete threshold-grid digest, and complete policy metrics;
- point-record digest;
- scientific payload digest.

Checkpoint replay independently refits the model from the gradient cache. Audit
and model-validation replay load the hash-pinned primary model.

## Claim boundary

Gate 3C1D is a commit-level top-1 gate. A positive result would establish a
causal, native-safe coordinate decision but would not yet establish future
tracking improvement. End-to-end coordinate-plus-memory rollout is reserved for
Gate 3C1E after all three commit confirmations pass.


## Final result

Gate 3C1D v0 completed on checkpoint-selection indices `384--447` and replayed
exactly. No threshold in the frozen grid passed all checkpoint gates. Formal
decision: `STOP_GATE3C1D_TOP1_SELECTOR`. Fit-only top-1 audit and original-model
top-1 confirmation were not run. See
`docs/ROUTED_TEMPORAL_IDENTITY_TOP1_GATE3C1D_V0_RESULT_2026-07-20.md`.
