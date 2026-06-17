# Track-On2 Memory Hygiene Diagnostic

- dataset: `davis`
- query_mode: `first`
- max_videos: `30`
- long_occ_min_run: `20`
- memory_size_eval: `24`
- support_grid_size: `20`

| Variant | AJ | delta_avg | OA | <4px | long_occ_AJ | long_occ_delta_avg | reentry_mean_px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 66.95 | 79.82 | 92.02 | 87.79 | 45.25 | 68.10 | 25.12 |
| oracle_mask | 67.33 | 79.45 | 92.77 | 87.36 | 48.07 | 64.86 | 44.82 |
| anchor_only | 66.84 | 79.64 | 91.98 | 87.84 | 45.93 | 68.44 | 25.58 |
| oracle_mask_anchor | 67.17 | 79.30 | 92.75 | 87.35 | 48.21 | 65.05 | 42.44 |
