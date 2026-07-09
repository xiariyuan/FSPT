# B2 Clean Mainline Runner — 2026-06-29

## Decision

B2 is now reproducible from a clean mainline script instead of only from sweep artifacts.

## Script

```text
scripts/run_b2_mainline_eval.py
```

## Default inputs

```text
base_cache = outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt
override_cache = outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt
```

## Method

```text
method = B2 predicted post-reentry localized override
trigger = base invisible run >= k and override visible
k = 1
pre = 1
post = full post segment
```

## Reproduced output

Output directory:

```text
outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline_repro/
```

Files:

```text
b2_predicted_mainline.pt
b2_predicted_mainline_ajrd.json
b2_predicted_mainline_standard_visibility.json
manifest.json
```

Reproduced metrics:

```text
true_AJ_RD_256 = 0.6278
true_AJ_RD = 0.4298
first_reentry_frame_proxy = 0.4963
AJ_256 = 68.9702
OA_256 = 91.1851
delta_avg_256 = 82.4604
reentry_missed_visible_rate = 0.122005
trigger_precision_track = 0.529996
trigger_recall_track = 0.963177
```

This matches the selected B2 mainline sweep result.

## Why this matters

The method is no longer just a selected row from a sweep. It has a reproducible entrypoint that can be reused for DAVIS, RGB-Stacking, and future datasets.

## Next step

Use the same runner after producing RGB-Stacking base and override caches. The next major validation task is RGB-Stacking 1-video smoke.
