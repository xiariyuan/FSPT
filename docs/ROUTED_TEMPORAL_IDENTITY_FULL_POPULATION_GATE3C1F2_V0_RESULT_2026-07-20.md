# Route-D full-population confirmation Gate 3C1F2 v0 result — 2026-07-20

## Formal decision

```text
COMPLETED_FAIL_WITH_EXACT_REPLAY
STOP_BEFORE_OFFICIAL_TAPVID
```

The frozen causal entry plus temporal-identity top-1 plus four-level memory-writeback pipeline completed on all 128 third-population Kubric videos and reproduced exactly in a fresh process. It did not pass the preregistered complete-video magnitude and AJ significance gates, so no official TAP-Vid evaluation is authorized.

## Complete-population support

```text
videos:                              128 / 128
eligible query rows:                 5,596
causal entry triggers:                 537
top-1 actions:                          89
action-support videos:                  44
actions with visible future GT:         51
actions without visible future GT:      38
```

The entry population contains only one GT-defined clean point among 537 triggers. This confirms strong clean safety, but 347 triggers and 44 final actions belong to points outside the earlier commit-visible/four-future-frame audit definition.

## Complete-video metrics

Equal-video means:

| Metric | Native | Modified | Gain |
|---|---:|---:|---:|
| AJ | 0.27116820 | 0.27118237 | +0.00001417 (+0.0014 points) |
| delta_avg | 0.39358192 | 0.39451843 | +0.00093652 (+0.0937 points) |
| OA | 0.85456396 | 0.85615510 | +0.00159115 (+0.1591 points) |

Paired-video 95% bootstrap CIs:

```text
AJ:        [-0.00033962, +0.00043515]
delta_avg: [+0.00037431, +0.00160094]
OA:        [+0.00030221, +0.00304005]
```

In percentage-point units:

```text
AJ:        -0.0340 to +0.0435 points
delta_avg: +0.0374 to +0.1601 points
OA:        +0.0302 to +0.3040 points
```

Delta_avg and OA gains are statistically positive under the frozen paired-video bootstrap, but both are below the preregistered `+0.25`-point magnitude gate. AJ is essentially null and its CI crosses zero.

## Action mechanism

Among 51 actions with at least one visible future frame:

```text
mean future error reduction:       +15.26799 px
positive-action fraction:            98.0392%
harmful by more than 4 px:             0.0000%
```

The memory-writeback mechanism therefore remains strong and safe when a useful action is selected. The complete-video failure is dominated by coverage and metric dilution, not by action-level damage.

## Gate results

Passed:

```text
128-video completion
entry support
89-action support
44 action-support videos
positive delta_avg CI lower bound
positive OA CI lower bound
AJ nonnegative-video fraction
mean action benefit
positive-action fraction
action harm control
fresh-process exact replay
```

Failed:

```text
evaluable actions required >=64:        observed 51
AJ gain required >=+0.0005:             observed +0.00001417
AJ CI lower required >0:                observed -0.00033962
delta_avg gain required >=+0.0025:      observed +0.00093652
OA gain required >=+0.0025:             observed +0.00159115
```

## Exact replay

Primary and replay independently reran all 128 videos. Exact comparisons all pass:

```text
per-video scientific digests:       true
aggregate video-record digest:      true
complete metric summary and CIs:    true
scientific payload digest:          true
```

```text
primary file SHA256:
37776251167df9c023b9d2907017f85c93df27745aa7d46dcb832ae2b75a77eb

replay file SHA256:
a0c4cdb255f203c957bf4d689b12365f34e0d8ef57c292f72f790ee200dd9239

scientific payload SHA256:
e1000928de9565aba106612aa2bca7c89566f8859d3325e9c00b91ae2fe9b9e4
```

## Scientific interpretation

Gate 3C1F2 separates the remaining problem into two parts:

1. **Action quality is no longer the bottleneck.** Selected actions improve future error by more than 15 px on average, 98% are positive, and none is harmful by more than 4 px.
2. **Full-population recall and metric leverage are insufficient.** Only 51 actions are evaluable under the eight-frame future audit, and their large local improvements affect too small a share of complete-video AJ. Visibility/OA benefits are more consistent than localization/Jaccard benefits.

The current operating point is safe but too conservative for a paper-table AJ claim. Post-result diagnosis may use these 128 videos as exposed development data, but no threshold may be retroactively changed for this result. Any redesigned entry/action policy requires a newly preregistered raw-record-disjoint confirmation population.

## Claim boundary

No official TAP-Vid DAVIS/Kinetics result is authorized. Final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain locked for this branch.
