# CoTracker3 Online V7-B1.2 Damage-Ceiling Policy Result — 2026-07-07

## Purpose

V7-B1.1 showed that simple per-video top-k caps were not the best operating policy. Damage-probability ceilings were more directly aligned with the real failure mode: bad visibility openings.

V7-B1.2 therefore evaluates fixed damage-ceiling policies on both full30 and 5 event-balanced video groups.

Base decision score remains:

```text
decision = P(benefit early4) - 1.35 * P(damage)
threshold = -0.01
```

Policies evaluated:

```text
base_no_ceiling
damage_ceiling = 0.7
damage_ceiling = 0.6
benefit_floor = 0.2 + damage_ceiling = 0.7
```

## Script

```text
scripts/eval_cotracker3_online_v7b12_damage_ceiling_group_robustness.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b12_damage_ceiling_group_robustness/v7b12_damage_ceiling_group_robustness_report.json
```

## Full30 results

### Base, no damage ceiling

```text
accept = 128
safe16_rate = 47.66%
bad_rate = 52.34%

AJ        -0.0576
OA        +0.0410
AJ_RD     +0.0020
AJ_RD_256 +0.0047
```

### damage_ceiling = 0.7

```text
accept = 125
safe16_rate = 48.00%
bad_rate = 52.00%

AJ        -0.0524
OA        +0.0331
AJ_RD     +0.0020
AJ_RD_256 +0.0047
```

This preserves the full AJ_RD/AJ_RD_256 gain while slightly improving AJ and safe/bad structure. It is the best default operating policy from this test.

### damage_ceiling = 0.6

```text
accept = 111
safe16_rate = 51.35%
bad_rate = 48.65%

AJ        -0.0318
OA        +0.0580
AJ_RD     +0.0017
AJ_RD_256 +0.0043
```

This is the most conservative policy and gives the healthiest safe/bad ratio, but sacrifices some AJ_RD_256 headroom.

### benefit_floor = 0.2 + damage_ceiling = 0.7

```text
accept = 119
safe16_rate = 47.90%
bad_rate = 52.10%

AJ        -0.0506
OA        +0.0253
AJ_RD     +0.0020
AJ_RD_256 +0.0047
```

This is also valid, but does not clearly beat damage_ceiling=0.7.

## 5-fold group robustness

### base_no_ceiling

```text
mean AJ        -0.0472
mean OA        +0.0626
mean AJ_RD     +0.00168
mean AJ_RD_256 +0.00404
success_folds = 2 / 5
mean_safe16_rate = 47.99%
mean_bad_rate = 52.01%
```

### damage_ceiling = 0.7

```text
mean AJ        -0.0396
mean OA        +0.0606
mean AJ_RD     +0.00170
mean AJ_RD_256 +0.00406
success_folds = 2 / 5
mean_safe16_rate = 48.24%
mean_bad_rate = 51.76%
```

This is a slight but consistent improvement over no ceiling in mean AJ, AJ_RD, AJ_RD_256, and safe/bad structure. It is the recommended V7-B2 default candidate.

### damage_ceiling = 0.6

```text
mean AJ        -0.0214
mean OA        +0.0762
mean AJ_RD     +0.00146
mean AJ_RD_256 +0.00372
success_folds = 2 / 5
mean_safe16_rate = 51.44%
mean_bad_rate = 48.56%
```

This is safer, but average AJ_RD_256 falls below the +0.004 target. It is useful as a conservative ablation, not the default.

### benefit_floor = 0.2 + damage_ceiling = 0.7

```text
mean AJ        -0.0414
mean OA        +0.0497
mean AJ_RD     +0.00170
mean AJ_RD_256 +0.00406
success_folds = 2 / 5
mean_safe16_rate = 48.56%
mean_bad_rate = 51.44%
```

Comparable to damage_ceiling=0.7, but slightly lower mean OA and no clear gain.

## Interpretation

V7-B1.2 confirms that damage ceilings are better than per-video top-k caps as an operating policy.

However, V7-B1.2 does not fully solve group-level robustness:

```text
All candidate policies still pass only 2 / 5 folds individually.
```

The most meaningful result is not a new large gain, but a cleaner default policy:

```text
damage_ceiling = 0.7 preserves full30 gain and slightly improves group mean metrics.
```

The conservative damage_ceiling=0.6 improves AJ/OA and safe/bad ratio, but loses too much average AJ_RD_256 to be the default.

## Decision

Promote the following as V7-B2 default candidate:

```text
V7-B2 default:
decision = P(benefit early4) - 1.35 * P(damage)
threshold = -0.01
damage_ceiling = 0.7
```

Report both variants:

```text
performance/default: damage_ceiling = 0.7
conservative ablation: damage_ceiling = 0.6
```

Do not continue top-k as the main policy.

## Next step

Proceed to V7-B2 final method packaging:

```text
1. Create final V7-B2 cache/method output for damage_ceiling=0.7.
2. Create conservative ablation cache for damage_ceiling=0.6.
3. Build final comparison table:
   native
   V6 best
   V7-B0
   V7-B1
   V7-B2 default damage_ceiling=0.7
   V7-B2 conservative damage_ceiling=0.6
   oracle useful / early4 where applicable
4. Write final method note with limitations:
   full30 positive, local basin stable, group-wise gains uneven.
```
