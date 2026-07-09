# CoTracker3 Online V5-D Visibility/Opening Verifier Result — 2026-07-06

## Purpose

After posthoc review showed that V5-B selective-oracle headroom mostly came from visibility opening rather than coordinate replacement, V5-D reframes the problem from candidate-coordinate selection to event-level visibility/opening verification.

## Dataset

Builder:

```text
scripts/build_cotracker3_online_v5d_event_dataset.py
```

Dataset:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v5d_visibility_verifier/cotracker3_v5d_event_visibility_dataset.npz
```

Statistics:

```text
samples: 3021
videos: 27
GT-visible positives: 857 / 3021 = 28.37%
safe16 positives: 653 / 3021 = 21.62%
safe8 positives: 435 / 3021 = 14.40%
```

Label used for training:

```text
y_safe16 = GT visible AND native coordinate error <= 16 px
```

Feature leakage audit:

```text
Passed. GT-derived fields are not used as input features.
```

## Training / validation

Script:

```text
scripts/train_cotracker3_online_v5d_visibility_verifier.py
```

Summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v5d_visibility_verifier/v5d_visibility_verifier_y_safe16_loov_summary.json
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

## Classification result

| Model | AP | ROC-AUC |
|---|---:|---:|
| LogisticRegression | 0.3679 | 0.6609 |
| ExtraTrees | 0.4136 | 0.6682 |
| RandomForest | 0.3792 | 0.6319 |

Compared to V5-C candidate-coordinate verifier:

```text
V5-C AP ~0.04-0.05, near random/base rate.
V5-D AP ~0.37-0.41, clearly better.
```

Therefore the posthoc correction was right:

```text
The learnable signal is mainly event-level visibility/opening, not candidate-coordinate selection.
```

## Metric simulation

Native full30:

```text
AJ      65.2366
OA      90.8186
delta   77.9458
d4px    85.8670
AJ_RD   0.3534
AJ_RD_256 0.5333
```

### Constrained threshold review

The important constraint is not maximizing AJ_RD blindly; opening too many frames hurts AJ/OA.

#### ExtraTrees best constrained-style rows

```text
threshold 0.75
accepted 111 events
precision 69.4%
recall 11.8%
AJ      +0.0077
OA      +0.1469
AJ_RD   -0.0001
AJ_RD_256 +0.0006
```

```text
threshold 0.7938
accepted 31 events
precision 64.5%
recall 3.1%
AJ      -0.0003
OA      +0.0256
AJ_RD   +0.0001
AJ_RD_256 +0.0004
```

#### RandomForest best constrained-style rows

```text
threshold 0.80
accepted 63 events
precision 63.5%
recall 6.1%
AJ      +0.0038
OA      +0.0598
AJ_RD   -0.0002
AJ_RD_256 +0.0003
```

```text
threshold 0.90
accepted 1 event
precision 100%
recall 0.2%
AJ      +0.0015
OA      +0.0017
AJ_RD   +0.0000
AJ_RD_256 +0.0001
```

### Aggressive thresholds

Aggressive thresholds can increase AJ_RD/AJ_RD_256 but hurt AJ/OA substantially.

Example:

```text
RandomForest threshold 0.414
accepted 1199 events
precision 32.3%
AJ      -1.1942
OA      -0.9865
AJ_RD   +0.0026
AJ_RD_256 +0.0084
```

This is not acceptable for a main method.

## Interpretation

V5-D confirms that event-level visibility/opening is learnable better than coordinate candidate selection:

```text
candidate-coordinate verifier: near random
visibility/opening verifier: AP ~0.37-0.41, AUC ~0.63-0.67
```

However, the metric gains are still not strong enough:

```text
safe thresholds preserve AJ/OA but give negligible AJ_RD gain.
aggressive thresholds improve AJ_RD_256 but damage AJ/OA too much.
```

## Final decision on CoTracker3 online branch

Do not continue CoTracker3 online as a mainline under current hand-engineered V3/V4/V5 designs.

What can be safely claimed:

```text
CoTracker3 true-streaming online experiments reveal that event-level visibility/opening is the main source of recoverable signal, while coordinate candidate selection alone has limited utility. A simple event-level verifier is learnable but does not yet yield a useful AJ/AJ_RD trade-off.
```

What cannot be claimed:

```text
CoTracker3 online ReEntry improves true-streaming re-detection.
```

## Recommended next step

Stop CoTracker3 online optimization for the main paper and use this branch as appendix/diagnostic.

Only resume if changing the modeling scope to one of:

```text
1. temporal event verifier using multiple future/past frames under short latency;
2. CoTracker internal feature/correlation verifier;
3. a trained sequence-level re-detection module, not hand-written thresholds;
4. a separate paper focused on online visibility calibration/risk-coverage rather than AJ_RD gain.
```

Current best paper use:

```text
Appendix: true-streaming online extension analysis and limitations.
```
