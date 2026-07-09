# CoTracker3 Online V8-B Candidate Coordinate Source Audit — 2026-07-07

## Purpose

V8-A showed that large gains are possible only when coordinates are recovered, not just visibility:

```text
V7-B2 learned visibility-only: AJ_RD_256 +0.0047
GT coord+visibility oracle early4/early8/useful: AJ_RD_256 +0.06 to +0.10
```

V8-B asks whether any existing non-GT coordinate source can partially realize this headroom.

## Script and report

Script:

```text
scripts/audit_cotracker3_online_v8b_candidate_coord_sources.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8b_candidate_coord_sources/v8b_candidate_coord_sources_report.json
```

Native reference:

```text
AJ        65.2366
OA        90.8186
AJ_RD     0.3534
AJ_RD_256 0.5333
```

Event labels:

```text
early4 = 97
early8 = 140
useful_open_t = 365
```

## Candidate sources inspected

Directly aligned with current DAVIS full30 protocol:

```text
old_cotracker3_online_bridge
old_cotracker3_offline_bridge
trackon2_dinov3_bridge
```

Rejected / not directly evaluated:

```text
trackon2_dinov3_davis: small query/GT geometry mismatch versus current bridge protocol
route_sweep_v2: same videos but 219-query records, not current 25-query records
RGB-stacking caches: different dataset/video ids
```

## Event-frame coordinate safety

### old_cotracker3_online_bridge

```text
early4: safe16 95.88%, safe8 60.82%, candidate_visible 15.46%
early8: safe16 95.71%, safe8 59.29%, candidate_visible 19.29%
useful: safe16 97.26%, safe8 63.29%, candidate_visible 37.53%
```

Coordinates are often close, but candidate visibility is low and standalone metrics are worse than native.

### old_cotracker3_offline_bridge

```text
early4: safe16 74.23%, safe8 46.39%, candidate_visible 5.15%
early8: safe16 75.00%, safe8 43.57%, candidate_visible 5.71%
useful: safe16 81.37%, safe8 53.70%, candidate_visible 21.64%
```

Worse coordinate source than expected in this protocol.

### trackon2_dinov3_bridge

```text
early4: safe16 89.69%, safe8 61.86%, candidate_visible 37.11%
early8: safe16 87.86%, safe8 60.71%, candidate_visible 43.57%
useful: safe16 89.86%, safe8 68.22%, candidate_visible 57.81%
```

This is the best non-GT coordinate source overall, especially because it has both decent coordinate accuracy and much higher candidate visibility.

## Standalone candidate metrics

### old_cotracker3_online_bridge

```text
AJ        -0.3417
OA        +0.9797
AJ_RD     -0.0048
AJ_RD_256 -0.0087
```

Not useful as a direct replacement.

### old_cotracker3_offline_bridge

```text
AJ        -2.5800
OA        -2.6698
AJ_RD     -0.0392
AJ_RD_256 -0.0808
```

Not useful as a direct replacement.

### trackon2_dinov3_bridge

```text
AJ        +1.8041
OA        +1.2730
AJ_RD     +0.0180
AJ_RD_256 +0.0111
```

TrackOn2 bridge is already stronger as a standalone external baseline, but the AJ_RD_256 gain is still below the V8 target of +0.02 to +0.03.

## Candidate coordinate recovery with oracle event/window selection

The following results use candidate coordinates, not GT coordinates, but still use oracle event/window labels. They estimate the candidate source's recovery headroom.

### Best result: TrackOn2 bridge + useful_open_t + w16

```text
variant: trackon2_dinov3_bridge_useful_open_t_w16_force_gt_visible
events = 365
touched_frames = 1219
candidate_safe16_rate_on_touched_gtvis = 88.76%
candidate_safe8_rate_on_touched_gtvis = 70.43%
candidate_safe4_rate_on_touched_gtvis = 42.04%

AJ        +0.6289
OA        +2.2280
AJ_RD     +0.0171
AJ_RD_256 +0.0354
```

This exceeds the V8 minimum target of +0.02 and reaches the preferred +0.03 target.

### TrackOn2 bridge + useful_open_t + segment

```text
events = 365
touched_frames = 1279
AJ        +0.5439
OA        +2.1420
AJ_RD     +0.0159
AJ_RD_256 +0.0335
```

### TrackOn2 bridge + early8 + w16

```text
events = 140
touched_frames = 645
AJ        +0.3642
OA        +1.2194
AJ_RD     +0.0158
AJ_RD_256 +0.0323
```

### TrackOn2 bridge + early8 + w8

```text
events = 140
touched_frames = 491
AJ        +0.2556
OA        +0.9598
AJ_RD     +0.0127
AJ_RD_256 +0.0264
```

### TrackOn2 bridge + early4 + w8

```text
events = 97
touched_frames = 435
AJ        +0.2330
OA        +0.8742
AJ_RD     +0.0123
AJ_RD_256 +0.0254
```

## Comparison to V8-A oracle

```text
GT coord+vis useful_open_t w16: AJ_RD_256 +0.1055
TrackOn2 coord useful_open_t w16: AJ_RD_256 +0.0354
```

TrackOn2 recovers about one third of the GT coordinate-recovery headroom under oracle event/window selection.

Compared with V7-B2:

```text
V7-B2 learned visibility-only: AJ_RD_256 +0.0047
TrackOn2 candidate coord recovery: AJ_RD_256 +0.025 to +0.035
```

This is a meaningful step toward the desired large-gain route.

## Main conclusion

A viable non-GT coordinate source exists:

```text
trackon2_dinov3_bridge
```

But the large result still depends on oracle event/window selection. Therefore, the next problem is not coordinate-source availability; it is selector design:

```text
When should we replace CoTracker3 online coordinates with TrackOn2 coordinates, and for how many frames?
```

## Decision

Promote V8-C as the next main step:

```text
V8-C candidate-aware recovery selector
```

Use TrackOn2 bridge as the candidate coordinate source.

Initial V8-C action space:

```text
source = trackon2_dinov3_bridge
window = 4 or 8 or 16
operation = replace native coords with TrackOn2 coords and open visibility
```

Initial selector families:

```text
1. V7-B1 damage-aware event score + TrackOn2 candidate safety features
2. candidate agreement/disagreement features: native vs TrackOn2 distance, TrackOn2 visibility, native score, raw vis/conf
3. oracle-label training target: event/window is useful if TrackOn2 replacement improves AJ_RD/AJ or produces safe8/safe16 frames
```

New target should remain high:

```text
minimum: AJ_RD_256 >= native +0.02
preferred: AJ_RD_256 >= native +0.03
```
