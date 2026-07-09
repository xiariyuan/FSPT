# TrackOn2 Repo-Native Logit Smoke Result — 2026-07-06

## Purpose

Test whether the repo-native TrackOn2 DAVIS first/input path can export visibility logits/confidence while preserving the old real-base TrackOn2 result.

This is a parity gate, not a method experiment.

## Script

```text
scripts/export_trackon2_repo_native_logits_smoke.py
```

The script uses:

```text
repo-native TAPVid dataset
repo-native DAVIS preprocessing: resize to 256, target_points *= 256
repo-native query conversion: [t,y,x] -> [t,x,y]
TrackOn2 Predictor with M_i=24, support_grid_size=20, delta_v=0.8
local DINOv3 via DINOV3_LOCAL_DIR
```

It does not modify the default `Predictor.forward` behavior. It uses a local subclass only inside the smoke script to preserve `v_logit`.

## Outputs

New repo-native logit `.npz`:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke/davis/trackon2/000000.npz
```

Report:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke_report.json
```

Bridge cache for metric comparison:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke_bridge_1video.pt
```

Metric comparison:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke_bridge_metric_compare.json
```

## New `.npz` keys

The new `.npz` contains the required keys:

```text
tracks
visibility
visibility_logit
visibility_conf
```

## Raw `.npz` parity against old 000000.npz

Old reference:

```text
outputs/trackon2_dinov3_davis_cache/davis/trackon2/000000.npz
```

Raw full-array comparison:

```text
tracks max_abs_diff  = 32.36198 px
tracks mean_abs_diff = 0.11704 px
tracks p95_abs_diff  = 0.10825 px
visibility diff_count = 2
visibility diff_rate  = 0.00116
```

Strict all-frame equality does not pass because there is a large outlier in frames that are not GT-visible / not evaluation-useful.

Important localization:

```text
max L2 diff occurs at query 15, frame 64
old_vis = false
new_vis = false
gt_vis = false
```

So the worst track discrepancy occurs on a frame where both old and new predictions are invisible and GT is also invisible.

## Evaluation-aware parity

After bridging the new `.npz` into the same old unified schema, first-video metrics are:

Old TrackOn2 bridge:

```text
AJ_256        = 58.4117
OA_256        = 90.0588
delta_avg_256 = 75.4319
AJ_RD_256     = 0.5257
AJ_RD         = 0.3397
```

Repo-native logit bridge:

```text
AJ_256        = 58.4326
OA_256        = 90.0588
delta_avg_256 = 75.4511
AJ_RD_256     = 0.5277
AJ_RD         = 0.3405
```

Difference:

```text
AJ_256        = +0.0209 pp
OA_256        = +0.0000 pp
delta_avg_256 = +0.0192 pp
AJ_RD_256     = +0.0020
AJ_RD         = +0.0008
```

This is well within practical metric parity for the smoke.

## Evaluation-relevant array comparison

Using the old bridge schema:

```text
all frames:
  track_max = 0.12691 normalized / 33.73 px L2 max
  track_mean = 0.000459 normalized
  visibility diff_count = 2

frames after query:
  track_max = 0.12691 normalized
  track_mean = 0.000517 normalized
  visibility diff_count = 2

gt-visible frames after query:
  track_max = 0.02746 normalized
  track_mean = 0.0000715 normalized
  track_p95 = 0.0002018 normalized
  visibility diff_count = 0
```

Interpretation:

```text
For GT-visible evaluation-relevant frames, visibility is exactly identical on this smoke, and coordinate differences are tiny for almost all frames.
The large raw max comes from invisible frames and does not affect the reported metrics.
```

## Visibility confidence check

The new cache stores confidence/logits. One borderline mismatch appears when comparing saved binary visibility to `visibility_conf >= 0.8`:

```text
mismatch count = 1
example conf = 0.7999117, logit = 1.3857422, visibility = true
```

This is explained by mixed-precision/autocast rounding: binary visibility was computed inside the model path, while confidence was recomputed from the saved logit. The difference is at the exact threshold boundary.

For downstream sweeps, use `visibility_conf` as the confidence source and the saved `visibility` as the native base binary prediction.

## Decision

The smoke passes **evaluation-aware parity**, but not strict raw all-frame equality.

Because this is a paper experiment, the next full run is allowed only under the following wording:

```text
Proceed to 30-video repo-native logit export as a parity-preserving confidence augmentation of TrackOn2, with an explicit note that raw all-frame tracks are not bit-identical but first-video metrics and GT-visible-frame visibility parity pass.
```

## Next step

Run 30-video repo-native logit export using the same repo-native path, but still do not run ReEntry yet.

The 30-video export must produce:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_full/davis/trackon2/*.npz
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_full_report.json
```

After export, bridge the 30 `.npz` files into a unified `.pt` cache with old query/GT/schema and added:

```text
pred_vis_logit
pred_vis_conf
```

Then compare full metrics to old TrackOn2 base:

```text
old base: outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
new conf bridge: outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full.pt
```

Full-run pass gate:

```text
AJ difference <= 0.10 pp preferred
OA difference <= 0.10 pp preferred
delta_avg difference <= 0.10 pp preferred
AJ_RD difference <= 0.005 preferred
visibility diff on GT-visible after-query frames should be near 0
```

Only if this full-cache parity passes may we run TrackOn2 threshold/local ReEntry audits.
