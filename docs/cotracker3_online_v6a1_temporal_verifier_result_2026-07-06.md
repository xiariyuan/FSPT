# CoTracker3 Online V6-A1 Temporal Verifier Result — 2026-07-06

## Purpose

V6-A1 tests a short-latency online verifier using native CoTracker3 temporal evidence over a local window instead of single-frame event evidence.

Compared to V5-D:

```text
V5-D = single-frame event-level visibility/opening verifier
V6-A1 = short-latency temporal verifier over t-2...t+4
```

No DINO features are used in V6-A1. This isolates whether CoTracker3 native temporal signals alone are enough.

## Dataset

Builder:

```text
scripts/build_cotracker3_online_v6a1_temporal_event_dataset.py
```

Dataset:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a1_temporal_verifier/cotracker3_v6a1_temporal_event_dataset.npz
```

Window:

```text
t-2 ... t+4
```

Statistics:

```text
samples: 1686
videos: 27
feature_dim: 45
future_safe16: 519 / 1686 = 30.78%
future_safe8: 389 / 1686 = 23.07%
future_gt_visible: 630 / 1686 = 37.37%
open_t_safe16: 365 / 1686 = 21.65%
```

Label used:

```text
y_future_safe16 = within t...t+4, there exists a GT-visible frame where native coordinate error <= 16 px
```

Feature leakage audit:

```text
Passed. No GT-derived feature names are used as model input.
```

## Training / validation

Script:

```text
scripts/train_cotracker3_online_v6a1_temporal_verifier.py
```

Summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a1_temporal_verifier/v6a1_temporal_verifier_y_future_safe16_loov_summary.json
```

Validation:

```text
leave-one-video-out
```

Models:

```text
LogisticRegression balanced
ExtraTrees balanced
RandomForest balanced
```

Opening strategies evaluated:

```text
open_t: open current trigger frame
first_recovery: open first frame in t...t+4 where score >= 0.6 or native visible
scoremax: open max-score frame in t...t+4
```

In this run, first_recovery and scoremax often produced the same best rows, indicating that future score peaks often coincide with first recovery in accepted events.

## Classification result

| Model | AP | ROC-AUC |
|---|---:|---:|
| LogisticRegression | 0.5168 | 0.6701 |
| ExtraTrees | 0.5640 | 0.6710 |
| RandomForest | 0.5628 | 0.6635 |

This is stronger than V5-D:

```text
V5-D ExtraTrees AP 0.4136, AUC 0.6682
V6-A1 ExtraTrees AP 0.5640, AUC 0.6710
```

So temporal evidence improves verifier ranking substantially.

## Native reference

```text
AJ      65.2366
OA      90.8186
delta   77.9458
d4px    85.8670
AJ_RD   0.3534
AJ_RD_256 0.5333
```

## Constrained metric review

Constraint used for method viability:

```text
AJ >= native - 0.10
OA >= native - 0.10
```

### LogisticRegression best constrained

```text
strategy first_recovery / scoremax
threshold 0.8406
accept 166
precision 67.5%
recall 21.6%
AJ      -0.0194
OA      +0.0161
AJ_RD   +0.0002
AJ_RD_256 +0.0006
```

### ExtraTrees best constrained

```text
strategy first_recovery / scoremax
threshold 0.6006
accept 298
precision 63.1%
recall 36.2%
AJ      -0.0389
OA      +0.0474
AJ_RD   +0.0003
AJ_RD_256 +0.0012
```

### RandomForest best constrained

```text
strategy first_recovery / scoremax
threshold 0.5512
accept 364
precision 59.3%
recall 41.6%
AJ      -0.0893
OA      -0.0003
AJ_RD   +0.0007
AJ_RD_256 +0.0024
```

## Raw AJ_RD_256-oriented review

Aggressive thresholds improve AJ_RD_256 more but violate AJ/OA constraints.

### ExtraTrees raw best

```text
strategy first_recovery / scoremax
threshold 0.45
accept 705
precision 47.1%
recall 64.0%
AJ      -0.3334
OA      -0.1795
AJ_RD   +0.0015
AJ_RD_256 +0.0049
```

### RandomForest raw best

```text
strategy first_recovery / scoremax
threshold 0.4446
accept 627
precision 48.5%
recall 58.6%
AJ      -0.3491
OA      -0.2217
AJ_RD   +0.0014
AJ_RD_256 +0.0041
```

These reach the AJ_RD_256 target but damage AJ/OA too much.

## Interpretation

V6-A1 is a real improvement in verifier quality:

```text
Temporal verifier AP is much higher than V5-D single-frame verifier.
```

But it is not yet a successful online method under the strict metric trade-off:

```text
Safe thresholds preserve AJ/OA but AJ_RD gain is still too small.
Aggressive thresholds reach AJ_RD_256 gain but damage AJ/OA beyond the allowed range.
```

## Decision

V6-A1 does not yet pass the online-method success gate.

However, unlike V5-C/V5-D, it shows enough signal to justify one more targeted iteration:

```text
V6-A2: temporal verifier with damage-aware objective / thresholding.
```

The next iteration should not add DINO first. It should fix the decision objective:

```text
Train/threshold for utility that penalizes false openings and AJ damage, not only future_safe16 classification.
```

Recommended V6-A2 target:

```text
positive = safe opening with AJ_RD utility
negative = opening would damage AJ/OA or open GT-occluded frame
sample_weight = utility magnitude / damage magnitude
```

If V6-A2 fails, CoTracker3 online should be finalized as appendix/diagnostic unless moving to a learned sequence-level module or internal CoTracker correlation verifier.
