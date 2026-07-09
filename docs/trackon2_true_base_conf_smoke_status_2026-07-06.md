# TrackOn2 True-Base Confidence Export Smoke Status — 2026-07-06

## Goal

Recover the correct experimental logic:

```text
real TrackOn2 forward_online base + ReEntry/confidence module
```

not:

```text
TrackOn2 offline_batch stress + ReEntry
```

## What was implemented

New smoke script:

```text
scripts/export_trackon2_davis_first_input_conf_smoke.py
```

It mirrors `Predictor.forward()` but preserves raw visibility logits:

```text
pred_tracks
pred_visibility
pred_vis_logit
pred_vis_conf
```

It also preserves support-grid behavior and uses the local DINOv3 path:

```text
DINOV3_LOCAL_DIR=/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

Smoke output:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_conf_smoke_1video.pt
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_conf_smoke_1video_report.json
```

## Smoke result

On first DAVIS video `bike-packing`:

```text
queries = 25
frames = 69
infer_sec = 6.578
pred_vis_rate = 0.6754
conf_min = 0.00079
conf_max = 0.99994
conf_mean = 0.79656
logit_min = -7.1398
logit_max = 9.7517
pred_visibility == pred_vis_conf >= delta_v = true
delta_v = 0.8
anchor_err_px_mean = 0.3356
anchor_err_px_max = 0.6535
```

This confirms:

```text
TrackOn2 true-base confidence/logit export is technically possible.
```

## Parity comparison with old TrackOn2 first/input cache

Compared:

```text
old:
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt

new smoke:
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_conf_smoke_1video.pt
```

First-video metrics:

| Cache | AJ_256 | OA_256 | delta_avg_256 | AJ_RD_256 | AJ_RD |
|---|---:|---:|---:|---:|---:|
| old TrackOn2 | 58.4117 | 90.0588 | 75.4319 | 0.5257 | 0.3397 |
| new conf smoke | 57.0214 | 88.1176 | 75.8925 | 0.4730 | 0.3005 |

Array comparison:

```text
pred_tracks max_abs_diff = 0.2501, mean_abs_diff = 0.00655
pred_visibility bool_diff_rate = 0.0551
query_points max_abs_diff = 0.00382
gt_tracks max_abs_diff = 0.00382
gt_visibility diff_rate = 0.0
```

## Interpretation

The smoke confirms logit/conf export is possible, but it does **not yet pass parity** with the old real-base TrackOn2 cache.

Likely causes:

```text
1. Slight difference in query/GT coordinate normalization between raw DAVIS pickle path and old bridge cache.
2. Possible difference in exact query conversion to model pixels.
3. Possible difference in config/memory/default settings versus old exporter.
```

Therefore:

```text
Do not run 30-video TrackOn2 confidence export yet.
Do not run ReEntry module on this smoke cache as a paper result.
```

## Next step

Before full export:

```text
Build TrackOn2 confidence exporter that uses the existing old TrackOn2 first/input cache as the schema/query/GT source.
```

Required behavior:

```text
1. Load old real-base TrackOn2 cache records.
2. Use old cache query_points and original_size to reconstruct model queries.
3. Use the raw DAVIS videos only for frames.
4. Preserve old cache query_points, gt_tracks, gt_visibility exactly in the new output.
5. Export new pred_tracks, pred_visibility, pred_vis_logit, pred_vis_conf.
6. Compare new pred_tracks/pred_visibility to old cache on 1 video.
```

Pass gate:

```text
query_points exact same as old cache
gt_tracks exact same as old cache
gt_visibility exact same as old cache
first-video AJ/OA close to old cache
visibility diff_rate should be explained and preferably small
```

Only after parity passes:

```text
Run 30-video TrackOn2 true-base confidence export.
Then run global threshold sweep and local recovery audit using the same success gate:
  AJ >= base - 0.10
  OA >= base - 0.10
  AJ_RD >= base + 0.01
```
