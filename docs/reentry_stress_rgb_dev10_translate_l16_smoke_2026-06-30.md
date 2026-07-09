# ReEntry-TAP RGB dev10 translate-L16 smoke — 2026-06-30

## Goal

Test the first controlled ReEntry-TAP stress protocol:

```text
Dataset: RGB dev0-9
Stress: translate exit-reenter
Length: L=16
Amplitude: 0.4W
Ramp: 8 frames
Start frame: t0=40
Fill: frame_mean
```

This is a development smoke test. RGB fresh20-49 was not used.

## Artifacts

```text
scripts/build_reentry_stress_rgb_dev10.py
scripts/export_cotracker_reentry_stress_cache.py
scripts/eval_reentry_stress_translate_l16.py

outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/stress_dataset.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/sanity_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/sample_visualizations/*.gif

outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_offline_translate_L16.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_online_translate_L16.pt
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/b2_w16_p2_translate_L16.pt

outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/eval/summary.json
```

## Sanity

The stress dataset passed sanity checks:

```text
num_videos = 10
num_kept_queries = 12054
query_keep_rate = 0.972332
query_frame_visible_rate = 1.0
num_reentry_queries = 2870
reentry_query_rate = 0.238095
num_reentry_events = 5139
occ_length_mean = 17.263281
occ_length_median = 13.0
nan_count = 0
visible_oob_count = 0
sample_gifs = 12
sanity_pass = true
```

This satisfies the initial target of `>=20%` re-entry queries and no GT/visibility consistency errors.

## Main result

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | RR@0 8px | RR@4 8px | RR@8 8px |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 0.4708 | 75.8656 | 90.7947 | 86.1429 | 0.3415 | 0.5502 | 0.5746 |
| CoTracker3 online | 0.5194 | 43.9771 | 57.0539 | 67.2819 | 0.5976 | 0.7122 | 0.7098 |
| B2-W16-P2 | 0.5485 | 75.8344 | 92.6729 | 86.3117 | 0.6066 | 0.7171 | 0.7209 |

## Deltas

```text
B2-W16-P2 vs offline:
AJ_RD_256 +0.0777
AJ_256    -0.0312

online vs offline:
AJ_RD_256 +0.0486
AJ_256    -31.8885

B2-W16-P2 vs online:
AJ_RD_256 +0.0291
AJ_256    +31.8573
```

## Trigger stats

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

This is a strong positive smoke result.

The controlled translate-L16 stress reproduces the same tradeoff observed in natural data:

```text
offline: strong standard AJ, weaker re-entry
online: stronger re-entry, severe standard-AJ collapse
B2-W16-P2: best AJ_RD while preserving offline-level AJ
```

The stress result is especially clean because B2-W16-P2 improves AJ_RD by `+0.0777` while incurring only `-0.0312` AJ cost relative to offline. It also beats online on both re-entry and standard tracking:

```text
B2-W16-P2 vs online:
AJ_RD +0.0291
AJ +31.8573
```

This supports upgrading the paper narrative toward a controlled ReEntry-TAP stress-test + reliability-intervention framing.

## Decision

Continue ReEntry-TAP.

Recommended next step:

```text
Expand translate stress to L=8 and L=32 on RGB dev10.
Then evaluate whether the severity curve is monotonic and whether B2-W16-P2 remains robust across stress lengths.
```

Do not yet run RGB fresh20-49 stress until the stress protocol is frozen.
