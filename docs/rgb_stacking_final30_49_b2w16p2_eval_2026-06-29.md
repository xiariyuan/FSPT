# RGB-Stacking Final30-49 B2-W16-P2 Evaluation — 2026-06-29

## Decision

Final RGB split `rgb_stacking_000030` ... `rgb_stacking_000049` supports the frozen B2-W16-P2 protocol. The method improves re-entry AJ_RD substantially over the offline base while preserving most of the offline standard AJ. This split was not used for method tuning.

## Protocol

```text
dev subset used for method selection: rgb_stacking_000000 ... rgb_stacking_000009
consumed diagnostic heldout:          rgb_stacking_000010 ... rgb_stacking_000019
fresh validation split:               rgb_stacking_000020 ... rgb_stacking_000029
final validation split:               rgb_stacking_000030 ... rgb_stacking_000049
frozen method: B2-W16-P2
```

## Query-weighted metrics

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.3859 | 78.7221 | 90.8754 | 87.9657 |
| CoTracker3 online | 0.4113 | 44.4676 | 55.6812 | 72.8116 |
| B2-W16 | 0.4432 | 78.1084 | 92.4878 | 87.9041 |
| B2-W16-P2 | 0.4429 | 78.1106 | 92.4764 | 87.9087 |

## Gains

```text
B2-W16-P2 vs offline AJ_RD_256 = +0.0570
B2-W16-P2 vs offline AJ_256    = -0.6115
B2-W16-P2 vs online AJ_RD_256  = +0.0316
B2-W16-P2 vs online AJ_256     = +33.6430
B2-W16-P2 vs B2-W16 AJ_RD_256  = -0.0003
B2-W16-P2 vs B2-W16 AJ_256     = +0.0022
```

## Trigger comparison

| method | trigger events | tracks with trigger | true-trigger tracks | false-trigger tracks | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| B2-W16 | 24357 | 7643 | 4123 | 3520 | 0.539448 | 0.888961 |
| B2-W16-P2 | 23102 | 7449 | 4052 | 3397 | 0.543966 | 0.873652 |

P2 reduces trigger events and false-trigger tracks, with a small recall tradeoff.

## Per-video stability

```text
n_videos = 20
p2_improves_AJRD_vs_offline = 16
p2_improves_AJRD_vs_offline_ge_0p01 = 15
p2_AJ_drop_vs_offline_le_1 = 14
p2_AJ_drop_vs_offline_gt_2 = 2
p2_beats_online_AJRD = 17
p2_beats_online_AJ_by_10 = 20
p2_improves_AJRD_vs_w16 = 6
p2_improves_AJ_vs_w16 = 10
```

Video-weighted means:

| method | video-mean AJ_RD_256 | video-mean AJ_256 |
|---|---:|---:|
| offline | 0.358085 | 78.722075 |
| online | 0.367175 | 44.467555 |
| B2-W16 | 0.414755 | 78.108355 |
| B2-W16-P2 | 0.414175 | 78.110625 |

## Interpretation

The final30-49 split is strong evidence for the paper: online has a better re-entry score than offline but catastrophic standard AJ; B2-W16-P2 harvests the re-entry benefit while preserving most standard tracking performance. The method remains a re-entry/standard-tracking tradeoff intervention, not a universal standard-tracking improvement.

## Artifacts

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_final30_49/cotracker3_offline_rgb_stacking_30_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_final30_49/cotracker3_online_rgb_stacking_30_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_final30_49/b2_w16_30_49/b2_w16_rgb_30_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_final30_49/b2_w16_p2_30_49/b2_w16_p2_rgb_30_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_final30_49/b2_w16_p2_30_49_per_video_audit/summary.json
```
