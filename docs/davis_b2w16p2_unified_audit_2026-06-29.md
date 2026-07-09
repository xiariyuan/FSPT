# DAVIS B2-W16-P2 Unified Audit — 2026-06-29

## Decision

DAVIS supports B2-W16-P2 as a unified main-method variant. Compared with B2-W16/full-post, P2 is a conservative reliability guard: it slightly lowers AJ_RD but improves standard AJ and trigger precision, while retaining the long-occlusion/re-entry gains that define the paper claim.

## Methods

| method | cache |
|---|---|
| fixed_offline | `outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt` |
| global_b1_vis4 | `outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt` |
| b2_fullpost | `outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline/b2_predicted_mainline.pt` |
| b2_w16 | `outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_base.pt` |
| b2_w16_p2 | `outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_persist2.pt` |
| b2_gt_window_oracle | `outputs/paper_discovery_2026-06-27/teacher_expansion/b2_localized_oracle/fixed_offline__override_vis4_gated288__pre0_post32.pt` |

## Per-video stability

Video-weighted means:

| method | video-mean AJ_RD_256 | video-mean AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| fixed offline | 0.497408 | 70.050983 | 92.154353 | 82.320567 |
| global B1 | 0.580244 | 47.404633 | 72.676183 | 76.491333 |
| B2 full-post | 0.5801 | 68.970223 | 91.18506 | 82.460387 |
| B2-W16 | 0.578924 | 68.951163 | 91.192367 | 82.448013 |
| B2-W16-P2 | 0.576904 | 69.01188 | 91.238203 | 82.44254 |
| GT-window oracle | 0.580628 | 70.182103 | 92.643887 | 82.34938 |

Counts:

```text
n_videos = 30
p2_improves_AJ_RD_vs_fixed = 18
p2_improves_AJ_RD_vs_fixed_ge_0p01 = 17
p2_AJ_drop_vs_fixed_le_1p5 = 20
p2_AJ_drop_vs_fixed_gt_3 = 4
p2_AJ_RD_within_0p005_of_fullpost = 18
p2_AJ_RD_loses_to_fullpost_by_gt_0p01 = 3
p2_AJ_RD_within_0p005_of_b1 = 18
p2_AJ_beats_b1_by_10 = 26
p2_improves_AJRD_vs_w16 = 7
p2_improves_AJ_vs_w16 = 17
```

## Occlusion-length bucket

| bucket | fixed | B2 full-post | B2-W16 | B2-W16-P2 | P2-fixed | P2-W16 | P2-full | n_events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| occ_1_4 | 0.624646 | 0.661936 | 0.66055 | 0.659045 | 0.034399 | -0.001505 | -0.002891 | 810 |
| occ_5_8 | 0.626069 | 0.639648 | 0.637485 | 0.638506 | 0.012437 | 0.001021 | -0.001142 | 340 |
| occ_9_16 | 0.449964 | 0.553275 | 0.551686 | 0.550941 | 0.100977 | -0.000745 | -0.002334 | 351 |
| occ_17_32 | 0.492585 | 0.597964 | 0.597472 | 0.593976 | 0.101391 | -0.003496 | -0.003988 | 330 |
| occ_33_plus | 0.291353 | 0.409669 | 0.410303 | 0.410587 | 0.119234 | 0.000284 | 0.000918 | 32 |

Key observation: B2-W16-P2 preserves the core re-entry gains, especially for longer occlusions: +0.100977 for occ 9-16, +0.101391 for occ 17-32, and +0.119234 for occ 33+ compared with fixed offline.

## Trigger / false-trigger taxonomy

```text
total_tracks = 5882
gt_reentry_tracks = 1385
triggered_tracks = 2395
true_trigger_tracks = 1314
false_trigger_tracks = 1081
no_trigger_reentry_tracks = 71
nontrigger_nonreentry_tracks = 3416
trigger_precision_track = 0.548643
trigger_recall_track = 0.948736
```

False-trigger summary:

```text
false_triggers = 1081
gt_visible_at_trigger = 661
gt_invisible_at_trigger = 420
low_cost_full_delta_ge_-0.01 = 553
low_cost_rate = 0.511563
harmful_full_delta_lt_-0.05 = 426
harmful_rate = 0.39408
severe_full_delta_lt_-0.10 = 268
severe_rate = 0.247919
false_delta_full_mean = -0.026044
false_delta_full_median = -0.006472
```

True-trigger summary:

```text
true_triggers = 1314
true_delta_full_mean = 0.000263
true_delta_post_mean = 0.025076
true_delta_post_p90 = 0.23804
```

## Interpretation

B2-W16-P2 should be treated as a robust main variant rather than a RGB-only patch. On DAVIS, it maintains most of the B2/full-post re-entry benefit, improves standard AJ over B2-W16, and provides a clearer trigger tradeoff: fewer triggers and higher precision at the cost of a small recall/AJ_RD reduction.

The correct paper framing remains: B2-W16-P2 is a favorable re-entry reliability / standard-tracking tradeoff intervention. It is not a universal standard-tracking improvement.

## Artifacts

```text
scripts/audit_davis_b2w16p2_unified.py
outputs/paper_discovery_2026-06-27/davis_b2w16p2_unified_audit/summary.json
outputs/paper_discovery_2026-06-27/davis_b2w16p2_unified_audit/per_video_rows.jsonl
outputs/paper_discovery_2026-06-27/davis_b2w16p2_unified_audit/occ_length_summary.json
outputs/paper_discovery_2026-06-27/davis_b2w16p2_unified_audit/trigger_taxonomy_summary.json
```
