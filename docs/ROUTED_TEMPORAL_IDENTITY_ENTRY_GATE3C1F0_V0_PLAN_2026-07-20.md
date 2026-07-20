# Route-D full-population causal entry Gate 3C1F0 v0 — 2026-07-20

## Status

```text
PREREGISTERED_NOT_RUN
new raw-record-disjoint population unread
DAVIS/Kinetics unread for this route
```

## Motivation

Gate 3C1E proves that the deployable Gate 3C1D selector plus deterministic
four-level memory improves future trajectories on a scientifically isolated
natural-failure population. That population was defined using future ground
truth and therefore cannot be used as a runtime trigger or paper-table entry
condition.

Gate 3C1F0 freezes a low-cost causal entry classifier that decides which ordinary
query points are allowed to enter the expensive 129-candidate temporal-identity
pipeline. It uses only CoTracker state observed through frame 15.

## Inputs and feature contract

The entry feature vector has exactly `130` float32 values:

- the existing `8 x 9` causal CoTracker trajectory-history tensor;
- frame-15 native visibility and confidence probabilities;
- 16 compact trajectory, visibility, confidence, joint-probability, and query
  distance summaries;
- mean, standard deviation, minimum, maximum, and normalized L2 norm for each of
  four native track-feature levels and four native support-memory levels.

Teacher coordinates, future ground truth, DINO descriptors, candidate maps, and
candidate selector predictions are forbidden. Candidate generation occurs only
after entry acceptance.

## Training and validation data

Only already exposed CSRR caches are used:

```text
train:                 source indices 8--31, 164 failure + 212 clean rows
fit-only validation:   source indices 32--47, 80 failure + 80 clean rows
```

The old CSRR learned action head is not reused. It collapsed under a joint
restoration/action loss. Gate 3C1F0 trains a separate entry-only HGB classifier.

## Frozen model

```text
HistGradientBoostingClassifier
max_iter:             240
learning_rate:        0.04
max_leaf_nodes:       15
l2_regularization:    3.0
min_samples_leaf:     15
random_state:         17
early_stopping:       false
```

Five-fold source-video OOF predictions are mandatory. The final model is fit on
all train rows and evaluated once on fit-only validation.

## Frozen operating point

Entry requires both:

```text
entry probability >= 0.93
native joint visibility-confidence probability <= 0.02
```

The rule is intentionally conservative. On exposed design data it gives:

```text
5-fold OOF: recall 14.63%, clean false apply 0%, precision 100%
fit validation: recall 15.00%, clean false apply 0%, precision 100%
```

## Design-only full-population pilot

On already exposed source indices `48--63`, the frozen entry plus unchanged Gate
3C1D top-1 policy produced 37 final actions across 16 complete videos:

```text
AJ gain:       +0.0858 points, 95% CI [-0.0268,+0.2073]
delta gain:    +0.4707 points, 95% CI [+0.0617,+1.0038]
OA gain:       +0.5242 points, 95% CI [+0.1594,+0.9489]
action future mean reduction:  +16.5102 px
action positive fraction:       95.65%
action harmful >4px fraction:    4.35%
```

This pilot is design evidence only. Its AJ interval crosses zero and these videos
have prior project exposure. It cannot authorize an external or paper-table
claim.

## Preregistered gates

### Five-fold source-video OOF

```text
AUC >= 0.82
AP >= 0.78
failure recall >= 0.12
clean false-apply rate <= 0.01
action precision >= 0.95
```

### Fit-only validation

```text
AUC >= 0.73
AP >= 0.75
failure recall >= 0.12
clean false-apply rate <= 0.01
action precision >= 0.95
```

Fresh-process replay must exactly reproduce feature, label, group, OOF
probability, validation probability, point-record, operating-point, and scientific
payload digests. Joblib byte identity is not the scientific equality criterion;
only the primary bundle may be used downstream and is pinned by file SHA256.

A passing replay issues only:

```text
AUTHORIZE_GATE3C1F0_RAW_DISJOINT_FULL_POPULATION_DATA
```

A failed gate issues:

```text
STOP_GATE3C1F0_CAUSAL_ENTRY
```

## Next authorized step

A pass allows materializing a third Kubric population after excluding all 1,024
raw identities used by Gate 3C0 and Gate 3C2. The complete-video two-stage policy
must then pass a new raw-record-disjoint internal confirmation before any official
DAVIS or Kinetics run.

Final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain locked.
