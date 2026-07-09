# B2 Predicted Window Full Sweep Audit — 2026-06-28

## Decision

B2 predicted-window localized fusion is the current deployable mainline candidate.

Best deployable B2 result:

```text
name: fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre1_post9999
true_AJ_RD_256: 0.6278
AJ_256_pct: 68.9702
OA_256_pct: 91.1851
delta_avg_256_pct: 82.4604
reentry_missed_visible_rate: 0.122005
total_triggers: 2517
tracks_with_trigger: 2517
```

## Reference comparison

| method | AJ_RD_256 | AJ_256 | meaning |
|---|---:|---:|---|
| fixed_offline | 0.5546 | 70.0510 | strong global tracker, weak re-entry |
| global B1 vis4_gated288 | 0.6279 | 47.4046 | strong re-entry, severe standard-AJ tradeoff |
| B2 GT-window oracle | 0.6290 | 70.1821 | diagnostic target using GT windows |
| B2 predicted best | 0.6278 | 68.9702 | deployable localized override |

## Best-by-AJ_RD table

| rank | name | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | reentry_miss |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre1_post9999 | 0.6278 | 68.9702 | 91.1851 | 82.4604 | 0.122005 |
| 2 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre0_post9999 | 0.6276 | 68.9249 | 91.1307 | 82.4504 | 0.131368 |
| 3 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre1_post16 | 0.6266 | 68.9512 | 91.1924 | 82.4480 | 0.122005 |
| 4 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre1_post32 | 0.6265 | 68.9596 | 91.1865 | 82.4479 | 0.122005 |
| 5 | fixed_offline__override_all4_gated192__base_inv_over_vis__k1_pre1_post9999 | 0.6264 | 68.9914 | 91.1898 | 82.4523 | 0.126172 |
| 6 | fixed_offline__override_all4_gated192__base_inv_over_vis__k1_pre0_post9999 | 0.6263 | 68.9430 | 91.1358 | 82.4409 | 0.135535 |
| 7 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre0_post32 | 0.6262 | 68.9173 | 91.1405 | 82.4374 | 0.131368 |
| 8 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre0_post16 | 0.6259 | 68.9090 | 91.1448 | 82.4380 | 0.131368 |
| 9 | fixed_offline__override_all4_gated192__base_inv_over_vis__k1_pre1_post16 | 0.6256 | 68.9760 | 91.1971 | 82.4453 | 0.126172 |
| 10 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k2_pre1_post9999 | 0.6252 | 68.9154 | 91.1545 | 82.4203 | 0.132407 |
| 11 | fixed_offline__override_all4_gated192__base_inv_over_vis__k1_pre1_post32 | 0.6252 | 68.9805 | 91.1912 | 82.4415 | 0.126172 |
| 12 | fixed_offline__override_all4_gated192__base_inv_over_vis__k1_pre0_post16 | 0.6249 | 68.9304 | 91.1498 | 82.4335 | 0.135535 |
| 13 | fixed_offline__override_all4_gated192__base_inv_over_vis__k1_pre0_post32 | 0.6249 | 68.9352 | 91.1456 | 82.4296 | 0.135535 |
| 14 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k2_pre0_post9999 | 0.6248 | 68.9046 | 91.1390 | 82.4132 | 0.143560 |
| 15 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre1_post8 | 0.6243 | 68.9220 | 91.1938 | 82.4125 | 0.122200 |
| 16 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k2_pre1_post32 | 0.6243 | 68.9119 | 91.1670 | 82.4092 | 0.132407 |
| 17 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k2_pre0_post32 | 0.6241 | 68.9049 | 91.1586 | 82.4019 | 0.143560 |
| 18 | fixed_offline__override_all4_gated192__base_inv_over_vis__k2_pre1_post9999 | 0.6241 | 68.9312 | 91.1596 | 82.4093 | 0.136574 |
| 19 | fixed_offline__override_all4_gated192__base_inv_over_vis__k2_pre0_post9999 | 0.6237 | 68.9193 | 91.1445 | 82.4013 | 0.147727 |
| 20 | fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre0_post8 | 0.6235 | 68.8730 | 91.1361 | 82.3976 | 0.131563 |

## Pareto frontier

| name | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | reentry_miss |
|---|---:|---:|---:|---:|---:|
| fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre1_post9999 | 0.6278 | 68.9702 | 91.1851 | 82.4604 | 0.122005 |
| fixed_offline__override_all4_gated192__base_inv_over_vis__k1_pre1_post9999 | 0.6264 | 68.9914 | 91.1898 | 82.4523 | 0.126172 |
| fixed_offline__override_vis4_gated288__base_inv_over_vis__k4_pre0_post9999 | 0.6223 | 69.0163 | 91.2695 | 82.3921 | 0.181039 |
| fixed_offline__override_vis4_gated288__base_inv_over_vis__k4_pre0_post32 | 0.6222 | 69.0170 | 91.2763 | 82.3877 | 0.181039 |
| fixed_offline__override_all4_gated192__base_inv_over_vis__k4_pre0_post32 | 0.6209 | 69.0260 | 91.2821 | 82.3736 | 0.185206 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre0_post9999 | 0.6166 | 69.5231 | 91.7825 | 82.3652 | 0.189776 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre1_post9999 | 0.6166 | 69.5231 | 91.7825 | 82.3669 | 0.189776 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k2_pre0_post9999 | 0.6160 | 69.5354 | 91.8069 | 82.3659 | 0.191325 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k2_pre1_post9999 | 0.6160 | 69.5354 | 91.8069 | 82.3676 | 0.191325 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre0_post32 | 0.6144 | 69.5847 | 91.8778 | 82.3518 | 0.191246 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre1_post32 | 0.6144 | 69.5847 | 91.8778 | 82.3535 | 0.191246 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre0_post16 | 0.6116 | 69.6186 | 91.8900 | 82.3549 | 0.190601 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre1_post16 | 0.6116 | 69.6186 | 91.8900 | 82.3569 | 0.190601 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k2_pre0_post16 | 0.6112 | 69.6254 | 91.9034 | 82.3570 | 0.192150 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k2_pre1_post16 | 0.6112 | 69.6254 | 91.9034 | 82.3590 | 0.192150 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre0_post8 | 0.6013 | 69.6608 | 91.8502 | 82.3623 | 0.205134 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k1_pre1_post8 | 0.6013 | 69.6608 | 91.8502 | 82.3650 | 0.205134 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k2_pre0_post8 | 0.6008 | 69.6674 | 91.8586 | 82.3669 | 0.206270 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k2_pre1_post8 | 0.6008 | 69.6674 | 91.8586 | 82.3698 | 0.206270 |
| fixed_offline__override_vis4_gated288__base_inv_over_rise__k8_pre0_post9999 | 0.5977 | 69.7463 | 92.0791 | 82.3441 | 0.293538 |

## Interpretation

The B2 rule nearly matches the global B1 AJ_RD score while recovering most of the fixed-offline standard AJ. This shows that the B1 tradeoff was caused mainly by applying aggressive fusion globally. Localized re-entry override keeps the strong base tracker for ordinary frames and uses the fusion branch only after predicted visibility loss/reappearance.

## Current best rule

```text
base = fixed_offline
override = vis4_gated288
trigger = base invisible run >= 1 and override visible
pre = 1
post = full post segment
```

## Next audit

The next required validation is trigger quality: how often the predicted trigger corresponds to true GT re-entry and how much delay exists between predicted trigger and true re-entry.
