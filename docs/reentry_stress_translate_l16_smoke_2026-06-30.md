# ReEntry-TAP RGB dev10 Translate-L16 Smoke Experiment — 2026-06-30

## Goal

Test whether a controlled translate exit-reenter stress protocol can expose the re-entry / standard-tracking tradeoff and whether B2-W16-P2 remains effective under stress-induced re-entry.

This is a development smoke experiment on RGB dev0-9 only. RGB fresh20-49 was not used.

## Stress construction

Stress name:

```text
translate_L16_A0p4_right
```

Parameters:

```text
stress type: translate_exit_reenter
length / hold L: 16 frames
amplitude: 0.4 W = 102.4 px
start frame t0: 40
ramp: 8 frames
direction: right
fill mode: frame_mean
```

Output:

```text
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16_A0p4_right/stress_dataset.pt
```

## Sanity check

```text
num_videos = 10
num_original_queries = 12397
num_kept_queries = 12062
query_keep_rate = 0.972977
num_reentry_queries = 2874
reentry_query_rate = 0.238269
num_reentry_events = 5139
mean_reentry_events_per_reentry_query = 1.7881
occlusion_length_mean = 17.2808
occlusion_length_median = 13.0
query_frame_visible_rate = 1.0
visible_out_of_bounds_count = 0
nan_count = 0
```

Occlusion length histogram:

```text
1-4:   1365
5-8:   553
9-16:  1215
17-32: 1282
33-64: 557
65+:   167
```

Decision: sanity passed. Stress generates sufficient re-entry events without invalid query frames or out-of-bounds visible GT.

## Methods evaluated

```text
CoTracker3 offline
CoTracker3 online
B2-W16-P2 = offline base + online override
```

Prediction caches:

```text
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16_A0p4_right/predictions/cotracker3_offline_translate_L16_A0p4_right.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16_A0p4_right/predictions/cotracker3_online_translate_L16_A0p4_right.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16_A0p4_right/eval/b2_w16_p2_translate_L16_A0p4_right.pt
```

## Main result

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.4736 | 75.9306 | 90.7920 | 86.1504 |
| CoTracker3 online | 0.5214 | 44.0848 | 57.0524 | 67.5244 |
| B2-W16-P2 | 0.5511 | 75.9406 | 92.7043 | 86.3433 |

## Key deltas

B2-W16-P2 vs offline:

```text
AJ_RD_256 +0.0775
AJ_256    +0.0100
OA_256    +1.9123
```

B2-W16-P2 vs online:

```text
AJ_RD_256 +0.0297
AJ_256    +31.8558
OA_256    +35.6519
```

Online vs offline:

```text
AJ_RD_256 +0.0478
AJ_256    -31.8458
```

## Trigger stats for B2-W16-P2

```text
gt_reentry_tracks = 2874
tracks_with_trigger = 3941
triggered_reentry_tracks = 2670
triggered_nonreentry_tracks = 1271
missed_reentry_tracks = 204
trigger_precision_track = 0.677493
trigger_recall_track = 0.929019
total_trigger_events = 12094
```

## Interpretation

This is a strong positive smoke result.

The stress protocol successfully exposes the same tradeoff seen in natural RGB:

```text
offline: high standard AJ, weaker re-entry
online: stronger re-entry, severe standard collapse
B2-W16-P2: strongest AJ_RD while preserving offline-level AJ
```

Most important: under translate-L16 stress, B2-W16-P2 improves AJ_RD over both offline and online, while keeping standard AJ essentially equal to offline.

This is stronger than the natural RGB fresh20-49 pattern because standard AJ cost is not only controlled but slightly positive in this stress smoke.

## Decision

Proceed to the next stress expansion:

```text
translate_L8_A0p4_right
translate_L32_A0p4_right
```

If L8/L16/L32 show a consistent severity curve, ReEntry-TAP can become a strong controlled-diagnostic contribution in the paper.
