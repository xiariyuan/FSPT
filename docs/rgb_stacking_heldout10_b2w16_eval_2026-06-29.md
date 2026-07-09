# RGB-Stacking Held-Out-10 B2-W16 Evaluation — 2026-06-29

## Decision

B2-W16 remains positive on held-out RGB-Stacking videos 10-19, without further tuning. It improves re-entry AJ_RD over both offline and online while preserving most of the offline standard AJ. However, the standard-AJ cost is larger than on the development subset and must be reported honestly.

This is stronger evidence than the previous RGB first-10 result because these videos were not used for selecting `W=16`.

## Protocol

```text
development subset: rgb_stacking_000000 ... rgb_stacking_000009
held-out subset:    rgb_stacking_000010 ... rgb_stacking_000019
method frozen before held-out: B2-W16
no additional held-out tuning
```

## Query-weighted metrics

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | note |
|---|---:|---:|---:|---:|---|
| CoTracker3 offline | 0.2659 | 80.817 | 92.7993 | 88.4013 | standard-strong base |
| CoTracker3 online | 0.2985 | 44.7124 | 55.854 | 74.1443 | re-entry stronger, standard weak |
| B2-W16 | 0.3179 | 79.4821 | 93.4963 | 88.3874 | frozen held-out method |

## Gains

```text
B2-W16 vs offline AJ_RD_256 = +0.0520
B2-W16 vs offline AJ_256 = -1.3349
B2-W16 vs online AJ_RD_256 = +0.0194
B2-W16 vs online AJ_256 = +34.7697
```

## Trigger behavior

```text
tracks_with_trigger = 2806
triggered_reentry_tracks = 1481
triggered_nonreentry_tracks = 1325
missed_reentry_tracks = 233
trigger_precision_track = 0.527798
trigger_recall_track = 0.864061
```

## Per-video stability

```text
n_videos = 10
b2_improves_AJRD_vs_offline = 8
b2_improves_AJRD_vs_offline_ge_0p01 = 7
b2_AJ_drop_vs_offline_le_1 = 6
b2_AJ_drop_vs_offline_gt_2 = 3
b2_beats_online_AJRD = 8
b2_beats_online_AJ_by_10 = 10
```

Video-weighted means:

| method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 | video-mean delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.26537 | 80.81702 | 92.79926 | 88.40133 |
| CoTracker3 online | 0.28239 | 44.71244 | 55.85404 | 74.14433 |
| B2-W16 | 0.3157 | 79.48207 | 93.49634 | 88.38739 |

## Best B2-W16 gains vs offline

| video | B2-offline AJ_RD | B2-offline AJ | B2 AJ_RD | offline AJ_RD |
|---|---:|---:|---:|---:|
| rgb_stacking_000019 | 0.1577 | -1.1663 | 0.2488 | 0.0911 |
| rgb_stacking_000016 | 0.0765 | -0.8951 | 0.273 | 0.1965 |
| rgb_stacking_000011 | 0.0638 | -0.7879 | 0.0638 | 0.0 |
| rgb_stacking_000017 | 0.0622 | -3.3177 | 0.3763 | 0.3141 |
| rgb_stacking_000012 | 0.0583 | -0.6052 | 0.246 | 0.1877 |

## Worst B2-W16 deltas vs offline

| video | B2-offline AJ_RD | B2-offline AJ | B2 AJ_RD | offline AJ_RD |
|---|---:|---:|---:|---:|
| rgb_stacking_000010 | -0.0036 | -3.7765 | 0.2645 | 0.2681 |
| rgb_stacking_000018 | 0.0 | -0.1515 | 0.4326 | 0.4326 |
| rgb_stacking_000013 | 0.0022 | -0.443 | 0.5232 | 0.521 |
| rgb_stacking_000015 | 0.0355 | -2.1038 | 0.2651 | 0.2296 |
| rgb_stacking_000014 | 0.0507 | -0.1025 | 0.4637 | 0.413 |

## Interpretation for paper target

This held-out result is important for a CCF-B / CAS-Q2-level target because it reduces the test-set-tuning risk: `W=16` was selected on the development subset, then evaluated on unseen RGB videos.

The result is positive but not perfect:

```text
positive: AJ_RD improves over offline by +0.0520 and over online by +0.0194
positive: B2 beats online standard AJ by +34.7697
caveat: standard AJ drops by -1.3349 versus offline
caveat: 3 / 10 videos lose more than 2 standard AJ points
```

Therefore the evidence supports the B2-W mechanism, but the paper must frame it as a reliability / tradeoff method, not a universal standard-AJ improvement method.

## Artifacts

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_heldout/cotracker3_offline_rgb_stacking_heldout10.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_heldout/cotracker3_online_rgb_stacking_heldout10.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_heldout/b2_w16_heldout10/b2_w16_rgb_heldout10.pt
scripts/audit_rgb_heldout_b2w16.py
outputs/paper_discovery_2026-06-27/rgb_stacking_heldout/b2_w16_heldout10_per_video_audit/summary.json
```
