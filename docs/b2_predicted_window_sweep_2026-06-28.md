# B2 Predicted Re-Entry Window Sweep — 2026-06-28

## Decision

The first deployable B2 predicted-window rule is promising and should become the next implementation focus.

Best first-sweep result:

```text
name: fixed_offline__override_vis4_gated288__base_inv_over_vis__k1_pre1_post9999
true_AJ_RD_256: 0.6278
AJ_256: 68.9702
OA_256: 91.1851
delta_avg_256: 82.4604
reentry_missed_visible_rate: 0.122005
```

## Meaning

Compared with global B1 `vis4_gated288`:

```text
global B1: AJ_RD_256 = 0.6279, AJ_256 = 47.4046
B2 predicted: AJ_RD_256 = 0.6278, AJ_256 = 68.9702
```

This nearly preserves the B1 re-entry gain while recovering more than 21 AJ points in standard AJ_256.

## Current best deployable rule

```text
base = fixed_offline
override = vis4_gated288
rule = base invisible run >= 1 and override visible
pre = 1
post = full post segment
```

This is deployable because it uses predicted visibility signals, not GT re-entry windows.

## Caveat

This first sweep only covered the focused `base_inv_over_vis` rule family for `fixed_offline + vis4_gated288`. A broader sweep over additional rule families and overrides is still needed.

## Next step

Run the full predicted B2 sweep across trigger modes and overrides. If the result remains stable, promote B2 predicted-window localized fusion as the deployable mainline candidate.
