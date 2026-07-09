# ReEntry-TAP RGB dev10 Translate-L16 Smoke Test — 2026-06-30

## Goal

Build the first controlled ReEntry-TAP stress test and verify whether the re-entry / standard-tracking tradeoff appears under synthetic but geometrically controlled re-entry.

This is a development smoke test only:

```text
Dataset: RGB dev0-9
Stress: translate_exit_reenter_L16
No RGB fresh20-49 used.
```

## Stress construction

```text
stress_type = translate_exit_reenter
length L = 16
amplitude = 0.4 * W
t0 = 40
ramp = 8
fill = frame_mean
```

Frames are translated without wrap-around, so points can truly leave the image and later re-enter. GT tracks and visibility are transformed analytically.

Artifacts:

```text
scripts/build_reentry_stress_rgb_dev10.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/stress_dataset.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/sanity_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/sample_visualizations/
```

## Sanity results

```text
num_videos = 10
num_original_queries = 12397
num_kept_queries = 12054
query_keep_rate = 0.972332
query_frame_visible_rate = 1.0
num_reentry_queries = 2870
reentry_query_rate = 0.238095
num_reentry_events = 5139
mean_reentry_events_per_query = 0.426332
occ_length_mean = 17.263281
occ_length_median = 13.0
nan_count = 0
visible_oob_count = 0
sample_gifs = 12
sanity_pass = true
```

The stress protocol passes the first-stage sanity criterion: at least 20% re-entry queries, no NaNs, no visible out-of-bounds points, and query-frame visibility is 100%.

## Methods evaluated

```text
CoTracker3 offline
CoTracker3 online
B2-W16-P2 = offline base + online override, P2 trigger, W16 window
```

Artifacts:

```text
scripts/export_cotracker_reentry_stress_cache.py
scripts/eval_reentry_stress_translate_l16.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_offline_translate_L16.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_online_translate_L16.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/b2_w16_p2_translate_L16.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/eval_summary.json
```

## Main metrics

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.4708 | 75.8656 | 90.7947 | 86.1429 |
| CoTracker3 online | 0.5194 | 43.9771 | 57.0539 | 67.2819 |
| B2-W16-P2 | 0.5485 | 75.8344 | 92.6729 | 86.3117 |

## Key gains

B2-W16-P2 vs offline:

```text
AJ_RD_256 +0.0777
AJ_256    -0.0312
OA_256    +1.8782
```

B2-W16-P2 vs online:

```text
AJ_RD_256 +0.0291
AJ_256    +31.8573
OA_256    +35.6190
```

Online vs offline:

```text
AJ_RD_256 +0.0486
AJ_256    -31.8885
OA_256    -33.7408
```

## Trigger statistics

```text
total_trigger_events = 12132
tracks_with_trigger = 3933
gt_reentry_tracks = 2870
triggered_reentry_tracks = 2668
triggered_nonreentry_tracks = 1265
missed_reentry_tracks = 202
trigger_precision_track = 0.678363
trigger_recall_track = 0.929617
```

## Interpretation

This smoke test strongly supports the ReEntry-TAP direction.

The stress protocol exposes the expected tradeoff:

```text
CoTracker3 online improves AJ_RD over offline, but collapses standard AJ.
```

But B2-W16-P2 does better than both:

```text
Compared with offline, it improves AJ_RD by +0.0777 with almost no AJ cost (-0.0312).
Compared with online, it further improves AJ_RD by +0.0291 while recovering +31.8573 AJ points.
```

This is stronger than a mere sanity check. It shows that a controlled re-entry stress test can make the re-entry / standard-tracking tradeoff highly visible, and that B2-W16-P2 remains a strong local intervention under stress-induced re-entry.

## Decision

Continue expanding ReEntry-TAP.

Recommended next steps:

```text
1. Add translate_L8 and translate_L32.
2. Produce severity curves over L = 8, 16, 32.
3. Run per-video and occlusion-length bucket analysis.
4. If translate severity remains positive, add moving-occluder stress.
```

This direction is now higher-value than continued B2-WA / B2-WT micro-tuning.
