# B2 Re-Entry Localized Fusion Oracle — 2026-06-28

## Decision

B2 localized fusion is now the most promising next mainline.

The GT-window diagnostic shows that applying the aggressive B1 fusion only around re-entry windows can preserve global standard TAP performance while matching or slightly exceeding the B1 AJ_RD result.

Best diagnostic result:

```text
name = fixed_offline__override_vis4_gated288__pre0_post32
base = fixed_offline
override = vis4_gated288
pre = 0
post = 32
true_AJ_RD_256 = 0.629
AJ_256 = 70.1821
OA_256 = 92.6439
delta_avg_256 = 82.3494
reentry_missed_visible_rate = 0.122005
```

This result is a diagnostic oracle because it uses GT re-entry windows. It is not deployable yet.

## Reference points

| method | AJ_RD_256 | AJ_256 | note |
|---|---:|---:|---|
| fixed_offline | 0.5546 | 70.0510 | strong global TAP, weak re-entry |
| old3_gated144 | 0.6189 | 49.6122 | intermediate re-entry, weak global AJ |
| B1 vis4_gated288 | 0.6279 | 47.4046 | strong re-entry, standard-AJ tradeoff |
| B2 GT-window best | 0.6290 | 70.1821 | diagnostic localized oracle |

## Best-by-AJ_RD table

| rank | name | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | reentry_miss |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fixed_offline__override_vis4_gated288__pre0_post32 | 0.6290 | 70.1821 | 92.6439 | 82.3494 | 0.122005 |
| 2 | fixed_offline__override_vis4_gated288__pre1_post32 | 0.6289 | 70.0845 | 92.5240 | 82.3494 | 0.122005 |
| 3 | fixed_offline__override_vis4_gated288__pre2_post32 | 0.6289 | 70.0441 | 92.4698 | 82.3506 | 0.122005 |
| 4 | fixed_offline__override_vis4_gated288__pre4_post32 | 0.6288 | 69.9843 | 92.3830 | 82.3592 | 0.122005 |
| 5 | fixed_offline__override_vis4_gated288__pre0_post9999 | 0.6279 | 70.1156 | 92.5500 | 82.3553 | 0.122005 |
| 6 | fixed_offline__override_vis4_gated288__pre1_post9999 | 0.6279 | 70.0235 | 92.4376 | 82.3553 | 0.122005 |
| 7 | fixed_offline__override_vis4_gated288__pre2_post9999 | 0.6279 | 69.9843 | 92.3854 | 82.3565 | 0.122005 |
| 8 | fixed_offline__override_vis4_gated288__pre4_post9999 | 0.6279 | 69.9292 | 92.3056 | 82.3652 | 0.122005 |
| 9 | old3_gated144__override_vis4_gated288__pre0_post9999 | 0.6279 | 49.7034 | 68.4199 | 68.7687 | 0.122005 |
| 10 | old3_gated144__override_vis4_gated288__pre1_post9999 | 0.6279 | 49.7001 | 68.4141 | 68.7687 | 0.122005 |
| 11 | old3_gated144__override_vis4_gated288__pre2_post9999 | 0.6279 | 49.6994 | 68.4099 | 68.7707 | 0.122005 |
| 12 | old3_gated144__override_vis4_gated288__pre4_post9999 | 0.6279 | 49.6929 | 68.4005 | 68.7737 | 0.122005 |
| 13 | fixed_offline__override_vis4_gated288__pre0_post16 | 0.6275 | 70.2216 | 92.6832 | 82.3466 | 0.122005 |
| 14 | old3_gated144__override_vis4_gated288__pre0_post32 | 0.6275 | 49.6974 | 68.4195 | 68.7613 | 0.122005 |
| 15 | old3_gated144__override_vis4_gated288__pre1_post32 | 0.6275 | 49.6941 | 68.4138 | 68.7613 | 0.122005 |

## Pareto frontier

| name | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | reentry_miss |
|---|---:|---:|---:|---:|---:|
| fixed_offline__override_vis4_gated288__pre0_post32 | 0.6290 | 70.1821 | 92.6439 | 82.3494 | 0.122005 |
| fixed_offline__override_vis4_gated288__pre0_post16 | 0.6275 | 70.2216 | 92.6832 | 82.3466 | 0.122005 |
| fixed_offline__override_vis4_gated288__pre0_post8 | 0.6221 | 70.2650 | 92.6495 | 82.3312 | 0.122005 |
| fixed_offline__override_vis4_gated288__pre0_post4 | 0.6105 | 70.3022 | 92.6098 | 82.3216 | 0.122005 |

## Interpretation

The previous B1 result revealed a tradeoff: `vis4_gated288` improved AJ_RD but lowered standard AJ. B2 shows that this tradeoff is not fundamental. It is mostly caused by applying aggressive fusion globally.

The best B2 diagnostic uses `fixed_offline` as the global base and applies `vis4_gated288` only in GT re-entry windows. This gives:

```text
AJ_RD_256: 0.6290, slightly above B1 0.6279
AJ_256: 70.1821, close to fixed_offline 70.051 and far above B1 47.4046
reentry_missed_visible_rate: 0.122005, matching B1
```

This is the strongest evidence so far that the method should become localized re-entry reliability rather than global fusion.

## Caveat

This is a GT-window oracle. It uses true re-entry frames and is not deployable. It should be used to justify the next deployable step: predicted-window localized fusion.

## Next step

Build a deployable predicted-window B2 rule. It should use only teacher/base visibility and disagreement signals to predict re-entry windows, then apply the same localized override.
