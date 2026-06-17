# Track-On2 Memory Hygiene Diagnostic

- dataset: `davis`
- query_mode: `first`
- max_videos: `2`
- long_occ_min_run: `20`
- memory_size_eval: `24`
- support_grid_size: `20`

| Variant | AJ | delta_avg | OA | <4px | long_occ_AJ | long_occ_delta_avg | reentry_mean_px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 77.89 | 86.97 | 94.45 | 93.14 | 40.64 | 70.08 | 5.75 |
| oracle_mask | 79.33 | 86.65 | 96.78 | 92.71 | 51.34 | 69.02 | 9.39 |
| anchor_only | 78.28 | 87.13 | 94.72 | 93.14 | 47.83 | 74.59 | 7.49 |
| oracle_mask_anchor | 79.65 | 87.06 | 96.78 | 93.23 | 54.45 | 72.18 | 7.50 |
