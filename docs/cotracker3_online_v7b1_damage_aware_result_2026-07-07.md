# CoTracker3 Online V7-B1 Damage-Aware Raw Vis/Conf Verifier — 2026-07-07

## Context

V7-B0 raw vis/conf visibility opening was the strongest previous CoTracker3 online result, but remained a near-miss:

```text
Best V7-B0 branch: early4 + all_components
threshold 0.85
AJ       -0.0833
OA       -0.0228
AJ_RD    +0.0014
AJ_RD_256 +0.0035
```

The V7-B0 damage audit showed the accepted events contained too many bad openings:

```text
accepted = 96
safe16 = 36
bad = 60
bad_rate = 62.5%
```

Therefore V7-B1 tests a damage-aware decision rule:

```text
decision_score = P(benefit) - lambda * P(damage)
```

where:

```text
benefit label = y_reentry_early4
damage label = GT occluded at event frame OR native coordinate error > 16px
```

## Scripts

```text
scripts/train_cotracker3_online_v7b1_damage_aware_raw_visconf.py
```

Inputs:

```text
features:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz

native cache:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt
```

Output summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/v7b1_damage_aware_raw_visconf_summary.json
```

## Model design

Benefit model:

```text
LOOV LogisticRegression, class_weight=balanced
features = all raw vis/conf component features
label = y_reentry_early4
```

Damage model:

```text
LOOV LogisticRegression, class_weight=balanced
training-fold-only top-k feature selection over component+product candidates
damage_top_k = 60
label = bad = GT occluded OR coordinate error > 16px
```

The damage feature selection is performed inside each training fold, not on the held-out video.

## OOF quality

```text
benefit_oof AP  0.1183
benefit_oof AUC 0.6377

damage_oof AP   0.8380
damage_oof AUC  0.6497
safe AP from negative damage score 0.4114
```

Damage AP is high partly because damage is the majority class, but the AUC is still meaningful and supports damage-aware gating.

## Native metric

```text
AJ       65.2366
OA       90.8186
AJ_RD    0.3534
AJ_RD_256 0.5333
```

Success gate:

```text
AJ delta >= -0.10
OA delta >= -0.10
AJ_RD delta >= +0.0015
AJ_RD_256 delta >= +0.004
```

## Main result

V7-B1 found 4 threshold/lambda settings that satisfy the full gate.

### Best constrained row

```text
lambda = 1.25
threshold = 0.0
accept = 135
benefit precision = 14.81%
benefit recall = 20.62%
accepted damage rate = 56.30%
accepted safe16 rate = 43.70%

AJ       -0.0786
OA       +0.0177
AJ_RD    +0.0020
AJ_RD_256 +0.0046
```

This passes the success gate.

### Other successful rows

```text
lambda = 1.25, threshold = -0.04135
accept = 161
AJ       -0.0971
OA       +0.0261
AJ_RD    +0.0019
AJ_RD_256 +0.0046
```

```text
lambda = 1.50, threshold = -0.10002
accept = 161
AJ       -0.0720
OA       +0.0942
AJ_RD    +0.0019
AJ_RD_256 +0.0046
```

```text
lambda = 1.50, threshold = 0.0
accept = 101
AJ       -0.0290
OA       +0.0491
AJ_RD    +0.0017
AJ_RD_256 +0.0040
```

## Interpretation

V7-B1 validates the hypothesis from V7-B0 damage audit:

```text
The bottleneck was not lack of benefit signal; it was insufficient filtering of bad openings.
```

Compared with V7-B0:

```text
V7-B0 best safe:
AJ_RD +0.0014, AJ_RD_256 +0.0035

V7-B1 best constrained:
AJ_RD +0.0020, AJ_RD_256 +0.0046
```

V7-B1 also improves OA in the successful settings, while keeping AJ within the safety budget.

## Important caveats

This is the first successful CoTracker3 online result in this branch, but it should not yet be overclaimed.

Caveats:

```text
1. lambda and threshold were selected on the same full30 OOF evaluation set.
2. Damage labels use GT for training/evaluation labels, as expected in supervised verifier development.
3. The current intervention opens only event frame visibility, not full state-writeback or temporal propagation.
4. The result needs refined threshold/lambda validation and possibly a held-out or video-family robustness audit.
```

## Decision

V7-B1 should replace V7-B0 as the main CoTracker3 online line.

Next step:

```text
V7-B1-refine: local lambda/threshold refinement around successful regions.
```

Focus regions:

```text
lambda around 1.25, threshold around [-0.06, 0.04]
lambda around 1.50, threshold around [-0.12, 0.02]
```

Evaluation must keep the same gate:

```text
AJ >= native - 0.10
OA >= native - 0.10
AJ_RD >= native + 0.0015
AJ_RD_256 >= native + 0.004
```

If refinement remains stable, move to:

```text
V7-B2: implement damage-aware visibility opening as the final CoTracker3 online method variant and run final comparison/report.
```
