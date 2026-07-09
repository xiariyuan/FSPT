# CoTracker3 Online V7-B1 Refine and Robustness Review — 2026-07-07

## Purpose

V7-B1 damage-aware raw vis/conf verifier was the first CoTracker3 online result to pass the predefined full metric gate. This review checks whether the success is a fragile single threshold/lambda point or a stable local region, and whether fixed successful settings are robust across video groups.

## Inputs

```text
features:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz

benefit OOF:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/benefit_oof_early4_all_components.npy

damage OOF:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/damage_oof_bad_topk.npy
```

Native metric:

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

## Local refine grid

Script:

```text
scripts/refine_cotracker3_online_v7b1_damage_aware_grid.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/refine_grid/v7b1_refine_grid_report.json
```

Grid:

```text
lambda = 1.15 ... 1.60, step 0.05
threshold = -0.14 ... 0.04, step 0.01
n_grid = 190
success_count = 122
```

This strongly suggests the success is not an isolated threshold accident.

### Best constrained rows

```text
lambda 1.35, threshold -0.01
accept 128
precision 15.62%
recall 20.62%
damage_rate 52.34%
safe16_rate 47.66%
AJ       -0.0576
OA       +0.0410
AJ_RD    +0.0020
AJ_RD_256 +0.0047
```

```text
lambda 1.30, threshold 0.01
accept 126
precision 15.87%
recall 20.62%
damage_rate 53.17%
safe16_rate 46.83%
AJ       -0.0606
OA       +0.0285
AJ_RD    +0.0020
AJ_RD_256 +0.0047
```

```text
lambda 1.25, threshold 0.03
accept 125
precision 16.00%
recall 20.62%
damage_rate 53.60%
safe16_rate 46.40%
AJ       -0.0628
OA       +0.0234
AJ_RD    +0.0020
AJ_RD_256 +0.0047
```

The previous V7-B1 row remains good:

```text
lambda 1.25, threshold 0.00
accept 135
AJ       -0.0786
OA       +0.0177
AJ_RD    +0.0020
AJ_RD_256 +0.0046
```

## Success region by lambda

Successful thresholds exist across multiple lambda values:

```text
lambda 1.15: 3 successful thresholds, best AJ_RD_256 +0.0046
lambda 1.20: 6 successful thresholds, best AJ_RD_256 +0.0046
lambda 1.25: 9 successful thresholds, best AJ_RD_256 +0.0047
lambda 1.30: 12 successful thresholds, best AJ_RD_256 +0.0047
lambda 1.35: 13 successful thresholds, best AJ_RD_256 +0.0047
lambda 1.40: 7 successful thresholds, best AJ_RD_256 +0.0047
```

This is the strongest evidence so far that the result is a real local basin rather than an isolated point.

## Fixed-setting group robustness

Script:

```text
scripts/eval_cotracker3_online_v7b1_fixed_setting_group_robustness.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/group_robustness/v7b1_fixed_setting_group_robustness_report.json
```

Five greedy event-balanced video folds were used:

```text
fold0 load 338: parkour, india
fold1 load 340: car-roundabout, soapbox, drift-chicane, paragliding-launch, scooter-black, lab-coat
fold2 load 336: dance-twirl, mbike-trick, gold-fish, dogs-jump, loading, camel, cows
fold3 load 337: bmx-trees, bike-packing, pigs, dog, judo, drift-straight
fold4 load 335: shooting, breakdance, car-shadow, motocross-jump, libby, horsejump-high, blackswans, goat, kite-surf
```

### Candidate: lambda 1.35, threshold -0.01

```text
mean AJ       -0.0472
mean OA       +0.0626
mean AJ_RD    +0.0017
mean AJ_RD_256 +0.0040
success_folds 2 / 5
```

Worst fold:

```text
min AJ       -0.1065
min OA       -0.0218
min AJ_RD    -0.0007
min AJ_RD_256 +0.0014
```

### Candidate: lambda 1.30, threshold 0.01

```text
mean AJ       -0.0496
mean OA       +0.0520
mean AJ_RD    +0.0017
mean AJ_RD_256 +0.0040
success_folds 2 / 5
```

### Candidate: lambda 1.25, threshold 0.03

```text
mean AJ       -0.0584
mean OA       +0.0404
mean AJ_RD    +0.0017
mean AJ_RD_256 +0.0040
success_folds 2 / 5
```

### Candidate: lambda 1.50, threshold -0.10

```text
mean AJ       -0.0631
mean OA       +0.1111
mean AJ_RD    +0.0016
mean AJ_RD_256 +0.0040
success_folds 2 / 5
```

## Interpretation

The local refine grid is strong:

```text
122 / 190 local settings pass the full gate.
```

However, group robustness is mixed:

```text
All fixed candidates average around AJ_RD_256 +0.0040 across folds,
but only 2 / 5 individual folds pass the full gate.
```

This means V7-B1 is globally positive on full30 and not a single-threshold accident, but it is not uniformly positive across all video groups.

The weak folds are informative:

```text
fold0: strong OA/AJ but weak or negative AJ_RD
fold1: weak AJ_RD and sometimes AJ slightly below safety
fold4: good AJ_RD_256 but AJ can fall slightly below -0.10
```

Therefore, V7-B1 should not yet be presented as a fully robust universal method. It is a validated positive candidate with uneven per-group gains.

## Decision

V7-B1 remains the main CoTracker3 online line, but before V7-B2 finalization it needs one more targeted improvement:

```text
V7-B1.1: conservative fold-robust operating point / policy selection.
```

Recommended fixed candidate for further refinement:

```text
lambda = 1.35, threshold = -0.01
```

Rationale:

```text
Best local full30 row.
Mean fold AJ_RD_256 = +0.0040.
Mean fold AJ = -0.0472.
Mean fold OA = +0.0626.
```

But the next step must address fold failures by adding a small global safety rule, not by changing the whole model.

Candidate safety gates:

```text
1. minimum benefit probability floor;
2. maximum damage probability ceiling;
3. require decision_score and benefit score to both pass thresholds;
4. per-video accept cap to avoid concentrated bad openings;
5. only open top-k accepted events per video by decision score.
```

The most promising next step is:

```text
V7-B1.1 per-video accept cap / top-k policy.
```

This directly targets folds where a few videos accumulate too many damaging openings.
