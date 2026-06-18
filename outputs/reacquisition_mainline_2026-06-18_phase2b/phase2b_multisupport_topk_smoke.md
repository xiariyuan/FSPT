# Phase 2b — Task B: Last-Visible Multi-Support Retrieval Smoke

- Max videos: 5, Max queries: 128
- Support frames: up to 8
- Top-K: 5, Crop size: 112

## Results

Variant                       n   top1@8   top5@8  top1@16  top5@16  t1miss_t5hit@16
------------------------------------------------------------------------------------
query_frame_anchor          128     0.0%     1.6%     0.0%     1.6%             1.6%
single_last_visible         128     0.0%     0.0%     0.0%     0.0%             0.0%
multi_support_mean          128     0.0%     0.0%     0.0%     2.3%             2.3%
multi_support_max           128     0.0%     0.8%     0.0%     1.6%             1.6%

## Stop Decision
- single_last_visible: **STOP** — top5@16px = 0.0% < 10%
- multi_support_mean: **STOP** — top5@16px = 2.3% < 10%
- multi_support_max: **STOP** — top5@16px = 1.6% < 10%
