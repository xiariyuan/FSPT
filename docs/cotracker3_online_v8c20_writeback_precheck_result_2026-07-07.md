# CoTracker3 Online V8-C2.0 State Writeback Feasibility Precheck

Date: 2026-07-07

Purpose: test whether CVRRM output corrections can be written back into CoTracker3 online recurrent state without obvious collapse.

## Setup

Default recovery source:

```text
CVRRM + TrackOn2 bridge, dist<=64, W=8
```

Two writeback modes were tested:

```text
coords_only: write candidate coordinates into online_coords_predicted only.
coords_visconf_high: write candidate coordinates and set visibility/confidence logits high.
```

The writeback variables are:

```text
online_coords_predicted
online_vis_predicted
online_conf_predicted
```

These variables are read by later online windows as overlap initialization, so this is a real state writeback test, not only output replacement.

## First 4-video precheck

Videos:

```text
bike-packing, parkour, camel, shooting
```

Native subset:

```text
AJ 51.0038, OA 84.3573, AJ_RD 0.1687, AJ_RD_256 0.2965
```

Output-level CVRRM delta:

```text
AJ +0.2887, OA +1.1421, AJ_RD +0.0286, AJ_RD_256 +0.0499
```

State writeback deltas:

```text
coords_only:
AJ +0.1350, OA +0.1838, AJ_RD +0.0104, AJ_RD_256 +0.0049

coords_visconf_high:
AJ +0.2416, OA +1.1246, AJ_RD +0.0285, AJ_RD_256 +0.0541
```

Interpretation:

```text
coords_only is insufficient.
coords_visconf_high can beat output-level AJ_RD_256 on this mixed 4-video subset.
```

Per-video comparison showed that gains came mainly from bike-packing and parkour; shooting worsened.

## Second 8-video precheck

Videos:

```text
bike-packing, parkour, loading, bmx-trees, camel, shooting, breakdance, car-shadow
```

Native subset:

```text
AJ 60.5729, OA 87.0920, AJ_RD 0.3356, AJ_RD_256 0.4896
```

Output-level CVRRM delta:

```text
AJ -0.1203, OA +0.5320, AJ_RD +0.0189, AJ_RD_256 +0.0360
```

State writeback deltas:

```text
coords_only:
AJ +0.2196, OA +0.3216, AJ_RD +0.0059, AJ_RD_256 +0.0064

coords_visconf_high:
AJ -0.1988, OA +0.3561, AJ_RD +0.0173, AJ_RD_256 +0.0386
```

Per-video state-minus-output AJ_RD_256 for coords_visconf_high:

```text
bike-packing +0.0111
bmx-trees    +0.0005
breakdance   +0.0018
camel         +0.0000
car-shadow    -0.0040
loading       +0.0162
parkour       +0.0041
shooting      -0.0057
```

Summary:

```text
5 videos better than output-level,
2 videos worse,
1 unchanged,
mean state-minus-output AJ_RD_256 about +0.0030.
```

## Reflection

The state writeback signal is real but fragile.

Positive:

```text
coords_visconf_high can propagate useful recovery into later windows.
It improves AJ_RD_256 over output-level on the 8-video subset.
```

Negative:

```text
coords_only is not enough.
coords_visconf_high can worsen negative cases such as shooting and car-shadow.
Full30 was not completed yet because the current script needs safer unique output naming / full30 execution handling.
```

## Decision

Do not promote state writeback yet.

Next step:

```text
V8-C2.1: make the writeback precheck script safe for full30.
```

Required fixes:

```text
1. Output filename must include subset/full30 tag to avoid overwriting previous reports.
2. Add a mode selector so full30 can run only coords_visconf_high first.
3. Save per-video deltas for output-level and writeback in the JSON.
4. Then run full30 once.
```

Only if full30 confirms that coords_visconf_high improves AJ_RD_256 without unacceptable AJ damage should state writeback become a main follow-up.
