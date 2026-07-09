# RGB-Stacking Fresh20-49 Aggregate Evaluation — 2026-06-29

## Decision

This is the cleanest RGB validation result so far: it aggregates only fresh validation videos `rgb_stacking_000020` ... `rgb_stacking_000049`, excluding the development split 0-9 and the consumed diagnostic heldout split 10-19.

The frozen B2-W16-P2 protocol remains positive on 30 fresh videos: it improves re-entry AJ_RD over the offline base while preserving most standard AJ, and it avoids the catastrophic standard-AJ collapse of the online override.

## Protocol

```text
Frozen B2-W16-P2 aggregate over fresh validation videos 20-49 only. Excludes dev 0-9 and consumed diagnostic heldout 10-19.
```

## Query-weighted metrics

| method | videos | queries | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 30 | 37221 | 0.3816 | 79.5944 | 91.4636 | 88.4854 |
| CoTracker3 online | 30 | 37221 | 0.4121 | 44.6933 | 55.7585 | 72.8684 |
| B2-W16 | 30 | 37221 | 0.4456 | 79.0634 | 92.8891 | 88.4346 |
| B2-W16-P2 | 30 | 37221 | 0.4454 | 79.0664 | 92.8804 | 88.4385 |

## Main gains

```text
p2_vs_offline_AJ_RD_256 = +0.0638
p2_vs_offline_AJ_256 = -0.5280
p2_vs_online_AJ_RD_256 = +0.0333
p2_vs_online_AJ_256 = +34.3731
p2_vs_w16_AJ_RD_256 = -0.0002
p2_vs_w16_AJ_256 = +0.0030
```

## Per-video stability

```text
n_videos = 30
p2_improves_AJRD_vs_offline = 24
p2_improves_AJRD_vs_offline_ge_0p01 = 23
p2_AJ_drop_vs_offline_le_1 = 20
p2_AJ_drop_vs_offline_gt_2 = 3
p2_beats_online_AJRD = 23
p2_beats_online_AJ_by_10 = 30
p2_improves_AJRD_vs_w16 = 12
p2_improves_AJ_vs_w16 = 17
```

Video-weighted means:

| method | video-mean AJ_RD_256 | video-mean AJ_256 |
|---|---:|---:|
| offline | 0.332953 | 79.5944 |
| online | 0.355243 | 44.693293 |
| B2-W16 | 0.39457 | 79.063407 |
| B2-W16-P2 | 0.39428 | 79.06644 |

## Interpretation

The result supports the central paper claim. Online has stronger re-entry performance than offline but collapses standard tracking. B2-W16-P2 localizes the online/override branch to predicted re-entry windows, improving AJ_RD by +0.0638 over offline while losing only -0.5280 standard AJ. It also beats online AJ_RD by +0.0333 and standard AJ by +34.3731.

This should become the primary RGB validation table. The previously consumed 10-19 split should remain diagnostic/appendix evidence rather than the main validation split.

## Artifacts

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/per_video_rows.jsonl
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/b2_w16_rgb_stacking_fresh20_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/b2_w16_p2_rgb_stacking_fresh20_49.pt
```
