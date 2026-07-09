# TrackOn2 True-Base Confidence Export — Parity Fix Plan — 2026-07-06

## Purpose

Recover a rigorous real-base experiment for TrackOn2:

```text
TrackOn2 normal forward_online base + confidence/logit export + ReEntry module audit
```

This must not use TrackOn2 offline_batch stress results.

## Current finding

A first smoke exporter successfully produced TrackOn2 visibility logits/confidence, but did not match the old first/input bridge exactly.

Old first-video metrics:

```text
AJ_256        58.4117
OA_256        90.0588
AJ_RD_256     0.5257
delta_avg_256 75.4319
```

New first smoke metrics:

```text
AJ_256        57.0214
OA_256        88.1176
AJ_RD_256     0.4730
delta_avg_256 75.8925
```

Main diagnosed cause:

```text
The old TrackOn2 bridge is in input256 space: target points are multiplied by 256 and normalized by 255.
The new smoke used original video space and normalized by original H/W.
```

Evidence from old exporter:

```text
points_pixel = points * [input_w, input_h]
query_points_norm = query_points_px / 255
pred_tracks_norm = pred_tracks_256 / 255
gt_tracks_norm = gt_tracks_256 / 255
source_space = input256
```

Therefore, the next parity smoke must run TrackOn2 in the same input256 coordinate regime.

## Next script to write

```text
scripts/export_trackon2_davis_first_input_conf_parity_smoke.py
```

## Inputs

Old real-base TrackOn2 cache:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
```

Raw DAVIS pkl only for video frames:

```text
datasets/tapvid_davis/tapvid_davis.pkl
```

TrackOn2 checkpoint/config:

```text
baselines/track_on/checkpoints_trackon2_dinov3.pt
baselines/track_on/config/test.yaml
```

Local DINOv3:

```text
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

## Required parity-smoke behavior

For first video only:

```text
1. Load old TrackOn2 cache first record.
2. Preserve old query_points, gt_tracks, gt_visibility, original_size, model_input_size exactly in output.
3. Load raw DAVIS video, resize frames to 256x256 before passing to TrackOn2.
4. Reconstruct model queries from old query_points:
     query_model = [t, x_px, y_px]
     where y_px = old_query_points[:,1] * 255
           x_px = old_query_points[:,2] * 255
5. Run TrackOn2 normal forward_online path with support_grid_size=20 and same config.
6. Export pred_tracks in old input256 schema:
     output TrackOn2 xy pixel in 256 space -> yx / 255
7. Export pred_visibility, pred_vis_logit, pred_vis_conf.
```

## Outputs

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_conf_parity_smoke_1video.pt
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_conf_parity_smoke_1video_report.json
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_conf_parity_smoke_metrics_compare.json
```

## Pass gate

Hard schema gate:

```text
query_points exact same as old cache
gt_tracks exact same as old cache
gt_visibility exact same as old cache
```

Metric gate:

```text
first-video AJ_256 close to old cache
first-video OA_256 close to old cache
first-video AJ_RD_256 close to old cache
```

Expected target:

```text
AJ_256 difference <= 0.25 pp preferred
OA_256 difference <= 0.25 pp preferred
AJ_RD_256 difference <= 0.01 preferred
pred_visibility diff_rate small and explained
```

If this fails:

```text
Do not run 30-video TrackOn2 confidence export.
Write parity-failed note and stop TrackOn2 true-base route for now.
```

If this passes:

```text
Run 30-video TrackOn2 true-base confidence export using the same input256-parity path.
Then run threshold sweep and local recovery audit.
Success gate for method claim:
  AJ >= base - 0.10
  OA >= base - 0.10
  AJ_RD >= base + 0.01
```

## Claim boundary

Before parity passes:

```text
No TrackOn2 method claim.
```

After parity export but before module success:

```text
TrackOn2 confidence export is available for true-base audit.
```

Only after module success gate passes:

```text
ReEntry improves TrackOn2 first/input under a true-base setting.
```
