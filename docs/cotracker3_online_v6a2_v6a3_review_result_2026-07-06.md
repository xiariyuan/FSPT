# CoTracker3 Online V6-A2 / V6-A3 Review Result — 2026-07-06

## Context

V6-A1 showed that temporal evidence over `t-2...t+4` improves event-level verifier ranking, but the classifier trained on `future_safe16` still does not produce enough AJ_RD gain under AJ/OA constraints.

V6-A2 and V6-A3 were run to determine whether the issue is target mismatch:

```text
V6-A2: useful-opening utility target
V6-A3: early re-entry target
```

## V6-A2 utility-label audit

Artifacts:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a2_utility_verifier/v6a2_utility_label_audit.json
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a2_utility_verifier/cotracker3_v6a2_utility_labels_from_v6a1.npz
```

Definition:

```text
useful opening = native prediction is currently invisible AND opening frame is GT-visible AND native coordinate error <= 16 px
```

Strategy statistics:

```text
open_t useful: 365 / 1686 = 21.65%
first_recovery useful: 311 / 1686 = 18.45%
scoremax useful: 311 / 1686 = 18.45%
```

Oracle useful-opening upper bound:

| Variant | opened | dAJ | dOA | dAJ_RD | dAJ_RD_256 |
|---|---:|---:|---:|---:|---:|
| useful open_t | 365 | +0.2890 | +0.8621 | +0.0061 | +0.0163 |
| useful first_recovery | 311 | +0.2027 | +0.6194 | +0.0048 | +0.0128 |
| useful scoremax | 311 | +0.2027 | +0.6194 | +0.0048 | +0.0128 |

This proves the online visibility-opening headroom is real if useful openings can be selected.

## V6-A2 useful-opening verifier

Artifact:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a2_utility_verifier/v6a2_utility_verifier_open_t_loov_summary.json
```

Label:

```text
y_useful_open_t
```

Statistics:

```text
samples: 1686
positives: 365
positive rate: 21.65%
```

LOOV classification:

```text
Logistic AP 0.3860, AUC 0.6448
ExtraTrees AP 0.4030, AUC 0.6506
RandomForest AP 0.3863, AUC 0.6277
```

Best constrained rows:

```text
Logistic threshold 0.95:
accept 22, precision 77.3%, recall 4.7%
AJ +0.0040, OA +0.0237, AJ_RD +0.0001, AJ_RD_256 +0.0003
```

```text
ExtraTrees threshold 0.70:
accept 85, precision 60.0%, recall 14.0%
AJ -0.0062, OA +0.0754, AJ_RD -0.0002, AJ_RD_256 +0.0007
```

```text
RandomForest threshold 0.70:
accept 71, precision 69.0%, recall 13.4%
AJ +0.0167, OA +0.0974, AJ_RD +0.0001, AJ_RD_256 +0.0008
```

Aggressive thresholds recover AJ_RD_256 but violate AJ/OA constraints:

```text
ExtraTrees threshold 0.4651:
accept 622, precision 35.0%, recall 59.7%
AJ -0.5557, OA -0.3496, AJ_RD +0.0015, AJ_RD_256 +0.0055
```

Conclusion:

```text
Utility target improves semantic alignment but still does not recover enough gain safely.
```

## V6-A2 event-offset diagnosis

Useful openings are mostly not early re-entry frames:

```text
open_t useful events: 365
GT re-entry first frame: 21 / 365 = 5.75%
GT re-entry within first 4 frames: 97 / 365 = 26.58%
```

Most useful events are later visible-segment low-score frames. They can improve OA/AJ, but they are not strongly aligned with AJ_RD.

## V6-A3 early re-entry oracle

Artifacts:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/v6a3_reentry_utility_oracle_report.json
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/cotracker3_v6a3_reentry_utility_labels.npz
```

Oracle counts and upper bounds:

| Label | count | dAJ | dOA | dAJ_RD | dAJ_RD_256 |
|---|---:|---:|---:|---:|---:|
| re-entry first | 21 | +0.0281 | +0.0638 | +0.0005 | +0.0016 |
| re-entry early2 | 64 | +0.0720 | +0.1748 | +0.0024 | +0.0066 |
| re-entry early4 | 97 | +0.0976 | +0.2622 | +0.0032 | +0.0088 |
| re-entry early8 | 140 | +0.1286 | +0.3479 | +0.0044 | +0.0120 |
| all useful | 365 | +0.2890 | +0.8621 | +0.0061 | +0.0163 |

This proves early re-entry openings are exactly the AJ_RD-aligned subset we want.

## V6-A3 early re-entry verifier result

### early4 target

Artifact:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/v6a3_reentry_verifier_y_reentry_early4_loov_summary.json
```

Statistics:

```text
samples: 1686
positives: 97
positive rate: 5.75%
```

LOOV result:

```text
Logistic AP 0.0516, AUC 0.3941
ExtraTrees AP 0.0401, AUC 0.3266
RandomForest AP 0.0482, AUC 0.4404
```

All models are near or below base rate; early4 is not learnable from current native temporal features.

### early8 target

Artifact:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/v6a3_reentry_verifier_y_reentry_early8_loov_summary.json
```

The early8 target is less sparse but still not learned well. Representative RandomForest rows show precision around or below the positive base rate and no meaningful AJ_RD gain under constraints.

Example:

```text
RandomForest threshold 0.438:
accept 197, precision 5.58%, recall 7.86%
AJ -0.2301, OA -0.2195, AJ_RD +0.0003, AJ_RD_256 +0.0007
```

Conclusion:

```text
Native temporal score/motion features cannot identify AJ_RD-aligned early re-entry events.
```

## Code inspection for next direction

CoTracker3 online exposes internal features/correlation hooks:

```text
cotracker3_online.py:
- self.fnet
- get_track_feat()
- get_correlation_feat()
- online_track_feat
- online_track_support
- corr_volume / corr_embs inside forward_window()
```

Relevant code locations:

```text
baselines/cotracker/cotracker/models/core/cotracker/cotracker3_online.py
```

This suggests a feasible next branch:

```text
V7: CoTracker internal correlation / support-memory verifier
```

## Decision

Stop iterating on native temporal score-only verifiers.

The current lightweight online chain is now well diagnosed:

```text
V6-A1: temporal window improves general ranking.
V6-A2: useful-opening target has real oracle headroom but mostly captures late visible-segment openings.
V6-A3: AJ_RD-aligned early re-entry oracle is strong, but native temporal features cannot learn it.
```

Next step should not be more threshold/model sweeps.

Proceed only by adding stronger online evidence:

```text
V7-A: export CoTracker internal correlation/support-memory features for event windows.
```

Success criterion for V7-A smoke:

```text
On V6-A3 early4/early8 labels, internal correlation features must improve AP clearly above base rate and above native-temporal features.
```

If V7-A does not improve early-reentry classification, CoTracker3 online should be appendix/diagnostic unless training a full sequence module.
