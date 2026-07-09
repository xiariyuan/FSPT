# RGB-Stacking B2 10-Video Per-Video / Trigger Audit — 2026-06-29

## Decision

The RGB-Stacking 10-video B2 result is not driven by a single video. B2 improves AJ_RD over the offline base on 9 / 10 videos, while keeping standard AJ within 1 point of the offline base on 7 / 10 videos.

The main failure is `rgb_stacking_000008`, where the online override is worse than the offline base on AJ_RD and the trigger has very low precision.

## Aggregate video-weighted metrics

| method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 | video-mean delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.32744 | 80.07207 | 90.99566 | 88.99971 |
| CoTracker3 online | 0.37607 | 45.42387 | 55.91637 | 73.1913 |
| B2 offline base + online override | 0.3905 | 79.75618 | 92.9475 | 88.94836 |

Note: the main reported 10-video score is query-weighted (`AJ_RD_256=0.4660`, `AJ_256=79.7562`). The table above is video-weighted for stability analysis.

## Stability counts

```text
n_videos = 10
b2_improves_AJRD_vs_offline = 9
b2_improves_AJRD_vs_offline_ge_0p01 = 9
b2_AJ_drop_vs_offline_le_1 = 7
b2_AJ_drop_vs_offline_gt_2 = 1
b2_beats_online_AJRD = 8
b2_beats_online_AJ_by_10 = 10
```

## Best AJ_RD gains vs offline

| video | B2 - offline AJ_RD | B2 - offline AJ | trigger precision | trigger recall | false trigger rate |
|---|---:|---:|---:|---:|---:|
| rgb_stacking_000006 | 0.1463 | 0.4479 | 0.888325 | 0.935829 | 0.031451 |
| rgb_stacking_000003 | 0.1311 | -0.4636 | 0.490566 | 0.952381 | 0.254477 |
| rgb_stacking_000001 | 0.117 | -1.0365 | 0.810619 | 0.997821 | 0.077424 |
| rgb_stacking_000005 | 0.1018 | 0.475 | 0.279743 | 1.0 | 0.183156 |
| rgb_stacking_000007 | 0.0797 | 0.2843 | 0.658703 | 0.932367 | 0.073529 |

## Worst AJ_RD deltas vs offline

| video | B2 - offline AJ_RD | B2 - offline AJ | offline AJ_RD | online AJ_RD | B2 AJ_RD | trigger precision | trigger recall | false trigger rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb_stacking_000008 | -0.1611 | -1.0629 | 0.3989 | 0.2315 | 0.2378 | 0.167164 | 0.982456 | 0.224277 |
| rgb_stacking_000004 | 0.0264 | -0.5009 | 0.3202 | 0.2753 | 0.3466 | 0.751152 | 0.867021 | 0.044082 |
| rgb_stacking_000002 | 0.0447 | -2.0204 | 0.4433 | 0.4861 | 0.488 | 0.621118 | 0.921659 | 0.130342 |
| rgb_stacking_000009 | 0.0667 | -0.6604 | 0.1235 | 0.1902 | 0.1902 | 0.742268 | 1.0 | 0.035236 |
| rgb_stacking_000000 | 0.078 | 1.3786 | 0.6169 | 0.6759 | 0.6949 | 0.859122 | 0.920792 | 0.053136 |

## Worst standard-AJ drops vs offline

| video | B2 - offline AJ | B2 - offline AJ_RD | offline AJ | B2 AJ | trigger precision | false trigger rate |
|---|---:|---:|---:|---:|---:|---:|
| rgb_stacking_000002 | -2.0204 | 0.0447 | 81.9365 | 79.9161 | 0.621118 | 0.130342 |
| rgb_stacking_000008 | -1.0629 | -0.1611 | 79.7414 | 78.6785 | 0.167164 | 0.224277 |
| rgb_stacking_000001 | -1.0365 | 0.117 | 81.4483 | 80.4118 | 0.810619 | 0.077424 |
| rgb_stacking_000009 | -0.6604 | 0.0667 | 80.7539 | 80.0935 | 0.742268 | 0.035236 |
| rgb_stacking_000004 | -0.5009 | 0.0264 | 79.4047 | 78.9038 | 0.751152 | 0.044082 |

## Interpretation

The 10-video RGB audit supports the B2 story:

```text
1. B2 improves AJ_RD over the standard-strong offline base on most videos.
2. B2 keeps standard AJ close to offline on most videos.
3. B2 strongly beats online on standard AJ for every video.
4. The main failure mode is low-precision triggering when the online override is not actually better for re-entry.
```

The important negative case is `rgb_stacking_000008`:

```text
offline AJ_RD_256 = 0.3989
online AJ_RD_256 = 0.2315
B2 AJ_RD_256 = 0.2378
B2 - offline AJ_RD_256 = -0.1611
trigger precision = 0.167164
false trigger rate = 0.224277
```

This suggests the next RGB improvement should not be another blind scale-up. It should either audit `rgb_stacking_000008` qualitatively or add a light trigger-quality filter before scaling to 20/50 videos.

## Artifacts

```text
scripts/audit_rgb_stacking_b2_10video.py
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_b2_10video_per_video_audit/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_b2_10video_per_video_audit/per_video_rows.jsonl
```
