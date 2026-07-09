# RGB-Stacking Fresh20-29 B2-W16-P2 Evaluation — 2026-06-29

## Decision

Fresh RGB split `rgb_stacking_000020` ... `rgb_stacking_000029` supports the B2-W mechanism and the lightweight P2 guard. B2-W16-P2 improves re-entry AJ_RD over the offline base and the online override while preserving most of the offline standard AJ.

Compared with B2-W16, B2-W16-P2 is a small but consistent cleanup: slightly higher AJ_RD, slightly higher standard AJ, fewer trigger events, and fewer false-trigger tracks.

## Protocol

```text
dev subset used for method selection: rgb_stacking_000000 ... rgb_stacking_000009
consumed heldout analysis:          rgb_stacking_000010 ... rgb_stacking_000019
fresh validation split:             rgb_stacking_000020 ... rgb_stacking_000029
frozen method: B2-W16-P2
```

## Query-weighted metrics

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.3706 | 81.3391 | 92.64 | 89.5248 |
| CoTracker3 online | 0.4142 | 45.1448 | 55.9131 | 72.9818 |
| B2-W16 | 0.4516 | 80.9735 | 93.6916 | 89.4954 |
| B2-W16-P2 | 0.4518 | 80.9781 | 93.6886 | 89.4982 |

## Gains

```text
B2-W16-P2 vs offline AJ_RD_256 = +0.0812
B2-W16-P2 vs offline AJ_256 = -0.3610
B2-W16-P2 vs online AJ_RD_256 = +0.0376
B2-W16-P2 vs online AJ_256 = +35.8333
B2-W16-P2 vs B2-W16 AJ_RD_256 = +0.0002
B2-W16-P2 vs B2-W16 AJ_256 = +0.0046
```

## Trigger comparison

| method | trigger events | tracks with trigger | true-trigger tracks | false-trigger tracks | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| B2-W16 | 8600 | 2772 | 1657 | 1115 | 0.597763 | 0.908941 |
| B2-W16-P2 | 8116 | 2698 | 1622 | 1076 | 0.601186 | 0.889742 |

## Per-video stability

```text
n_videos = 10
p2_improves_AJRD_vs_offline = 8
p2_improves_AJRD_vs_offline_ge_0p01 = 8
p2_AJ_drop_vs_offline_le_1 = 6
p2_AJ_drop_vs_offline_gt_2 = 1
p2_beats_online_AJRD = 6
p2_beats_online_AJ_by_10 = 10
p2_improves_AJRD_vs_w16 = 6
p2_improves_AJ_vs_w16 = 7
```

Video-weighted means:

| method | video-mean AJ_RD_256 | video-mean AJ_256 |
|---|---:|---:|
| offline | 0.28269 | 81.33905 |
| online | 0.33138 | 45.14477 |
| B2-W16 | 0.3542 | 80.97351 |
| B2-W16-P2 | 0.35449 | 80.97807 |

## Interpretation

Fresh20-29 is the best validation so far for the corrected protocol: the method was chosen before this split, then evaluated without further tuning. The result supports the paper claim that B2-W is a localized re-entry reliability intervention that recovers re-entry performance while avoiding the catastrophic standard-AJ collapse of the online override.

However, the method still has a standard-AJ tradeoff versus offline. The correct claim is not universal tracking improvement; it is a favorable re-entry/standard-tracking tradeoff.

## Artifacts

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_29/cotracker3_offline_rgb_stacking_fresh20_29.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_29/cotracker3_online_rgb_stacking_fresh20_29.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_29/b2_w16_fresh20_29/b2_w16_rgb_fresh20_29.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_29/b2_w16_p2_fresh20_29/b2_w16_p2_rgb_fresh20_29.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_29/b2_w16_p2_fresh20_29_per_video_audit/summary.json
```
