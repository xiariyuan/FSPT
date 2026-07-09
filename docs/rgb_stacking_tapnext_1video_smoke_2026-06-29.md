# RGB-Stacking TAPNext 1-Video Smoke — 2026-06-29

## Decision

The RGB-Stacking dataset adapter and TAPNext 1-video export path are working.

## Dataset adapter

```text
datasets/tapvid_rgb_stacking.py
```

1-video schema smoke:

```text
video: (250, 3, 256, 256) torch.float32 in [0, 1]
query_points: (1148, 3)
target_points: (1148, 250, 2)
occluded: (1148, 250)
video_name: rgb_stacking_000000
original_size: [256, 256]
```

## TAPNext exporter

```text
scripts/export_tapnext_rgb_stacking_cache.py
```

Output cache:

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/tapnext_rgb_stacking_1video.pt
```

Report:

```text
n_records = 1
n_queries = 1148
frames = 250
vis_rate = 0.1551
export_sec = 22.898
peak_mem_mb = 2227.9
```

## Evaluation smoke

AJ_RD output:

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/tapnext_rgb_stacking_1video_ajrd.json
```

Standard TAP output:

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/tapnext_rgb_stacking_1video_standard.json
```

Metrics on RGB-Stacking video 0:

```text
true_AJ_RD_256 = 0.0647
true_AJ_RD = 0.0647
first_reentry_frame_proxy = 0.1802
AJ_256 = 15.0884
OA_256 = 30.6603
delta_avg_256 = 22.8275
n_reentry_queries = 404
```

## Interpretation

This is not yet B2 cross-dataset validation. It only proves that one teacher can be exported on RGB-Stacking and evaluated with the same cache schema / metric path.

The next step is to export the remaining teacher caches for the same 1 video:

```text
cotracker3_offline
cotracker3_online
trackon2
```

Then build the B1 override branch and run the clean B2 mainline runner.
