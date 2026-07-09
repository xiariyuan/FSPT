# RGB-Stacking post16 Per-Video Audit — 2026-06-29

## Decision

The `post16` improvement is not a single-video artifact. Compared with full-post B2, `post16` improves AJ_RD on 9 / 10 RGB videos and improves standard AJ on 9 / 10 videos.

This supports promoting finite-window override as a serious B2 variant for RGB, while still requiring DAVIS finite-window validation before making it the unified method.

## Aggregate video-weighted metrics

| method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 | video-mean delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.32744 | 80.07207 | 90.99566 | 88.99971 |
| CoTracker3 online | 0.37607 | 45.42387 | 55.91637 | 73.1913 |
| B2 full-post | 0.3905 | 79.75618 | 92.9475 | 88.94836 |
| B2 post16 | 0.42322 | 79.90302 | 93.0717 | 89.09944 |

## Counts

```text
n_videos = 10
post16_improves_AJRD_vs_full = 9
post16_improves_AJRD_vs_full_ge_0p01 = 6
post16_improves_AJ_vs_full = 9
post16_improves_AJRD_vs_offline = 9
post16_improves_AJRD_vs_offline_ge_0p01 = 9
post16_AJ_drop_vs_offline_le_1 = 9
post16_AJ_drop_vs_offline_gt_2 = 0
post16_beats_online_AJRD = 9
post16_beats_online_AJ_by_10 = 10
```

## Best post16 gains vs full-post

| video | post16-full AJ_RD | post16-full AJ | post16-offline AJ_RD | post16-offline AJ |
|---|---:|---:|---:|---:|
| rgb_stacking_000008 | 0.1443 | 0.2465 | -0.0168 | -0.8164 |
| rgb_stacking_000004 | 0.0832 | 0.3985 | 0.1096 | -0.1024 |
| rgb_stacking_000007 | 0.0429 | 0.0223 | 0.1226 | 0.3066 |
| rgb_stacking_000009 | 0.0181 | 0.0891 | 0.0848 | -0.5713 |
| rgb_stacking_000001 | 0.0131 | 0.205 | 0.1301 | -0.8315 |

## Worst post16 deltas vs full-post

| video | post16-full AJ_RD | post16-full AJ | full AJ_RD | post16 AJ_RD |
|---|---:|---:|---:|---:|
| rgb_stacking_000005 | 0.0 | -0.0112 | 0.1194 | 0.1194 |
| rgb_stacking_000002 | 0.0037 | 0.2486 | 0.488 | 0.4917 |
| rgb_stacking_000006 | 0.0052 | 0.0453 | 0.6556 | 0.6608 |
| rgb_stacking_000000 | 0.006 | 0.0629 | 0.6949 | 0.7009 |
| rgb_stacking_000003 | 0.0107 | 0.1614 | 0.4807 | 0.4914 |

## Interpretation

`post16` behaves as expected for a localized intervention: it preserves the re-entry benefit near the trigger, but avoids long-term contamination from the online override. The largest gain is on `rgb_stacking_000008`, but most other videos also improve slightly.

The only non-improving AJ_RD case vs full-post is essentially a tie: `rgb_stacking_000005` has the same AJ_RD and only -0.0112 standard AJ.

## Next required validation

Run finite-window B2 on DAVIS. If DAVIS also tolerates post16/post32, the final method can be unified as a finite-window localized re-entry override. If DAVIS strongly prefers full-post, the paper should present full-post and finite-window as two ablations under the same B2 framework.

## Artifacts

```text
scripts/audit_rgb_post16_per_video.py
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_post16_per_video_audit/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_post16_per_video_audit/per_video_rows.jsonl
```
