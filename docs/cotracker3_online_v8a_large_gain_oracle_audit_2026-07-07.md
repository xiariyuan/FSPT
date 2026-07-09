# CoTracker3 Online V8-A Large-Gain Oracle / Action Audit — 2026-07-07

## Purpose

V7-B2 damage-aware visibility opening passed the previous gate, but the gain was too small:

```text
V7-B2 default damage_ceiling=0.7:
AJ        -0.0524
OA        +0.0331
AJ_RD     +0.0020
AJ_RD_256 +0.0047
```

This is not enough for a main contribution. V8-A therefore asks a more important question:

```text
Which action has large headroom?
```

The audit tests output-space oracle actions:

```text
1. visibility-only opening over t, t:t+4, t:t+8, t:t+16, or GT-visible segment
2. coordinate + visibility oracle over the same windows
3. current learned V7-B2 policy as reference
```

Important: this is an oracle/action audit, not a deployable method. GT coordinates are used only to estimate headroom.

## Script and report

Script:

```text
scripts/audit_cotracker3_online_v8a_large_gain_oracles.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8a_large_gain_oracles/v8a_large_gain_oracles_report.json
```

Inputs:

```text
native cache:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt

event features:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz
```

Event labels:

```text
early4 = 97
early8 = 140
useful_open_t = 365
```

Native metric:

```text
AJ        65.2366
OA        90.8186
AJ_RD     0.3534
AJ_RD_256 0.5333
```

## Reference: V7-B2 learned visibility policy

```text
variant: v7b2_default_damage_ceiling_0p7
events = 125
touched_frames = 125

AJ        -0.0524
OA        +0.0331
AJ_RD     +0.0020
AJ_RD_256 +0.0047
```

This confirms V7-B2 is only a small positive result.

## Visibility-only oracle

Best visibility-only oracle:

```text
variant: oracle_vis_useful_open_t_w16_safe16
events = 365
touched_frames = 1054

AJ        +0.5943
OA        +1.8406
AJ_RD     +0.0107
AJ_RD_256 +0.0308
```

Other similar visibility-only results:

```text
oracle_vis_useful_open_t_w8_safe16:
AJ_RD_256 +0.0302

oracle_vis_useful_open_t_wsegment_safe16:
AJ_RD_256 +0.0302

oracle_vis_early8_wsegment_safe16:
AJ_RD_256 +0.0281

oracle_vis_early4_wsegment_safe16:
AJ_RD_256 +0.0279
```

Interpretation:

```text
Visibility-only has more headroom than V7-B2, but it requires oracle-level event/frame selection over many frames.
```

The learned V7-B2 opens only 125 single frames, while the visibility oracle opens around 1000+ safe frames. Therefore, simply tuning the V7 policy cannot reach this oracle without a much stronger event/segment selector.

## Coordinate + visibility oracle

This is the key result.

### useful_open_t, t:t+16

```text
variant: oracle_coordvis_useful_open_t_w16
events = 365
touched_frames = 1219

AJ        +3.0658
OA        +2.2280
AJ_RD     +0.1093
AJ_RD_256 +0.1055
```

### useful_open_t, GT-visible segment

```text
variant: oracle_coordvis_useful_open_t_wsegment
events = 365
touched_frames = 1279

AJ        +3.0501
OA        +2.1420
AJ_RD     +0.1054
AJ_RD_256 +0.1012
```

### early8, t:t+8

```text
variant: oracle_coordvis_early8_w8
events = 140
touched_frames = 491

AJ        +1.2699
OA        +0.9598
AJ_RD     +0.0839
AJ_RD_256 +0.0810
```

### early4, t:t+8

```text
variant: oracle_coordvis_early4_w8
events = 97
touched_frames = 435

AJ        +1.1428
OA        +0.8742
AJ_RD     +0.0788
AJ_RD_256 +0.0761
```

Even the smaller early4 t:t+4 oracle is already large:

```text
variant: oracle_coordvis_early4_w4
events = 97
touched_frames = 315

AJ        +0.8562
OA        +0.6660
AJ_RD     +0.0616
AJ_RD_256 +0.0597
```

## Main conclusion

V8-A answers the key question:

```text
Large headroom exists.
```

But it is not in single-frame visibility calibration.

The real headroom is in:

```text
coordinate + visibility recovery over short post-reentry windows.
```

Compared with V7-B2:

```text
V7-B2 learned visibility only:
AJ_RD_256 +0.0047

Visibility-only oracle over many safe frames:
AJ_RD_256 about +0.030

Coord+visibility oracle over early4/early8 short windows:
AJ_RD_256 about +0.060 to +0.081

Coord+visibility oracle over useful windows:
AJ_RD_256 about +0.10
```

Therefore, continuing V7 threshold/policy tuning is not justified.

## Decision

Stop V7-B as the main route.

Promote V8 as the main route:

```text
V8 = re-entry coordinate recovery + visibility recovery
```

The next non-oracle task is not to open visibility more aggressively. It is to find or learn a candidate coordinate source that can approximate the coord+vis oracle.

## Next step

Proceed to:

```text
V8-B candidate coordinate source audit
```

V8-B should test available non-GT coordinate sources, for example:

```text
1. full-sequence/offline CoTracker cache if available;
2. old first-input bridge / non-streaming cache if available;
3. internal spatial search candidates if available;
4. local temporal extrapolation / smoothing baselines;
5. later, external feature retrieval or DINO-style candidate search.
```

Success target for V8-B should be much higher than V7:

```text
AJ_RD_256 >= native + 0.02
```

Preferred target:

```text
AJ_RD_256 >= native + 0.03
```

If no non-GT candidate source gets close, then the required next step is a learned re-detection / coordinate recovery module, not more visibility calibration.
