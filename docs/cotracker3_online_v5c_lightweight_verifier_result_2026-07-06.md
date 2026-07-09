# CoTracker3 Online V5-C Lightweight Verifier Feasibility Result — 2026-07-06

## Purpose

V5-B showed moderate oracle headroom for hard-event candidate pools. V5-C tests whether a lightweight verifier can recover this headroom without GT at inference.

This is a diagnostic experiment, not a method result.

## Dataset

Builder:

```text
scripts/build_cotracker3_online_v5c_candidate_dataset.py
```

Dataset:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v5c_verifier/cotracker3_v5c_local96_top20_candidate_dataset.npz
```

Base:

```text
CoTracker3 true-streaming native first10 hard-event set from V5-B
```

Candidate pool:

```text
local96_s8 top20 candidates per hard event
```

Dataset statistics:

```text
hard events: 40
candidate samples: 800
positive candidate_good: 34
positive rate: 4.25%
videos with hard events: bike-packing, bmx-trees, breakdance, car-roundabout, dance-twirl
```

Positive label:

```text
candidate_err + 2 px < native_err
AND candidate_err <= 8 px
```

Feature leakage audit:

```text
Passed. GT-derived quantities such as candidate_err/native_err/labels are not used as input features.
```

GT is used only for labels and diagnostic evaluation.

## Validation protocol

Script:

```text
scripts/train_cotracker3_online_v5c_lightweight_verifier.py
```

Summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v5c_verifier/v5c_lightweight_verifier_loov_summary.json
```

Validation:

```text
leave-one-video-out
```

Models:

```text
LogisticRegression(class_weight=balanced)
ExtraTreesClassifier(class_weight=balanced)
RandomForestClassifier(class_weight=balanced)
```

## Candidate-level result

Positive rate baseline:

```text
4.25%
```

OOF results:

| Model | AP | ROC-AUC |
|---|---:|---:|
| Logistic regression | 0.0500 | 0.4773 |
| ExtraTrees | 0.0449 | 0.5122 |
| RandomForest | 0.0442 | 0.5129 |

Interpretation:

```text
The verifier does not learn a useful cross-video candidate-quality signal. AP is near the positive base rate and AUC is near random.
```

## Simulated verifier metric result

Best observed metric sweeps were weak and did not recover V5-B selective oracle gains.

Representative best rows:

### Logistic regression

```text
threshold 0.95
accepted candidates: 23
candidate precision: 8.70%
event accept count: 2
AJ delta: -0.0021
OA delta: +0.0000
AJ_RD delta: +0.0000
AJ_RD_256 delta: +0.0000
```

### ExtraTrees

```text
threshold 0.45
accepted candidates: 79
candidate precision: 5.06%
event accept count: 11
AJ delta: -0.0477
OA delta: +0.0611
AJ_RD delta: +0.0000
AJ_RD_256 delta: +0.0003
```

### RandomForest

```text
threshold 0.35
accepted candidates: 9
candidate precision: 0.00%
event accept count: 4
AJ delta: -0.0170
OA delta: +0.0245
AJ_RD delta: +0.0000
AJ_RD_256 delta: +0.0001
```

## Comparison to V5-B oracle

V5-B selective oracle had moderate upper bound:

```text
local96_s8_top20 selective oracle:
AJ +0.0856
OA +0.2038
AJ_RD +0.0050
AJ_RD_256 +0.0106
```

V5-C lightweight verifier recovered essentially none of this:

```text
AJ_RD gain <= +0.0000 / +0.0001 in best useful sweeps
candidate precision far below 70%
accepted positive candidates too few / unreliable
```

## Decision

Do not continue CoTracker3 online V5 with the current candidate features and lightweight verifier.

Safe conclusion:

```text
Although hard-event candidate pools contain moderate oracle headroom, simple GT-free candidate-level features do not allow a lightweight verifier to reliably select better candidates across videos.
```

Unsafe conclusion:

```text
CoTracker3 online V5 verifier improves re-detection.
```

## Next possible directions

Only continue CoTracker3 online if changing the modeling assumption, not by tuning current features:

```text
1. Use temporal candidate consistency, not single-frame candidate scoring.
2. Use CoTracker internal feature/correlation tensors rather than DINO patch CLS alone.
3. Use a sequence-level verifier trained with more events, not first10 only.
4. Use a stronger candidate generator inspired by TAPIR/LocoTrack-style matching/refinement.
```

Recommended paper positioning:

```text
Keep CoTracker3 online V3/V4/V5 as appendix/diagnostic evidence:
- state writeback is structurally possible;
- visibility-only writeback has little headroom;
- hard-event candidate pools have oracle headroom;
- simple verifier cannot recover it.
```
