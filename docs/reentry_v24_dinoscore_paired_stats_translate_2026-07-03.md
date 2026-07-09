# Paired Video Statistics — fresh20_49_translate_L16_v24_dinoscore_vs_v22Q

## Methods

| Method | Cache |
|---|---|
| v22Q | `outputs/paper_discovery_2026-06-27/reentry_interval_v22/protocol_clean_fresh/fresh20_49_translate_L16_v22Q/reentry_viscalibrator_v22Q_block001_min2_edge015.pt` |
| v24_dinoscore | `outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/translate_v24_micro.pt` |

## Video-weighted method means

| Method | Videos | Re-entry queries | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|---:|---:|
| v22Q | 30 | 8104 | 0.556553 | 74.9919 | 92.1943 |
| v24_dinoscore | 30 | 8104 | 0.556390 | 74.9987 | 92.1951 |

## Paired comparisons against `v24_dinoscore`

### v24_dinoscore_minus_v22Q

| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD_256 | -0.000163 | [-0.000330, -0.000027] | 7 / 12 / 11 | 0.359283 |
| AJ_256 | 0.006792 | [0.002758, 0.011738] | 23 / 5 / 2 | 0.000912 |
| OA_256 | 0.000823 | [-0.005200, 0.007547] | 12 / 15 / 3 | 0.701108 |

## Per-video table

| video_id | v22Q_AJ_RD | v22Q_AJ | v22Q_OA | v24_dinoscore_AJ_RD | v24_dinoscore_AJ | v24_dinoscore_OA |
|---|---:|---:|---:|---:|---:|---:|
| rgb_stacking_000020_translate_L16 | 0.456800 | 80.8076 | 96.2732 | 0.456600 | 80.8078 | 96.2707 |
| rgb_stacking_000021_translate_L16 | 0.667900 | 77.4495 | 94.3227 | 0.667900 | 77.4935 | 94.3805 |
| rgb_stacking_000022_translate_L16 | 0.726400 | 77.9301 | 92.1702 | 0.726000 | 77.9265 | 92.1665 |
| rgb_stacking_000023_translate_L16 | 0.653800 | 79.9179 | 96.8071 | 0.653800 | 79.9179 | 96.8071 |
| rgb_stacking_000024_translate_L16 | 0.633700 | 74.5869 | 88.6387 | 0.633700 | 74.5942 | 88.6306 |
| rgb_stacking_000025_translate_L16 | 0.313100 | 75.5234 | 89.0173 | 0.313000 | 75.5266 | 89.0143 |
| rgb_stacking_000026_translate_L16 | 0.473300 | 79.3254 | 90.9555 | 0.473100 | 79.3417 | 90.9730 |
| rgb_stacking_000027_translate_L16 | 0.827900 | 73.1960 | 95.0180 | 0.827900 | 73.1960 | 95.0180 |
| rgb_stacking_000028_translate_L16 | 0.568000 | 81.2676 | 94.7648 | 0.568000 | 81.2717 | 94.7700 |
| rgb_stacking_000029_translate_L16 | 0.571300 | 64.6183 | 90.9361 | 0.570900 | 64.6133 | 90.9227 |
| rgb_stacking_000030_translate_L16 | 0.576500 | 79.1511 | 94.7431 | 0.576600 | 79.1525 | 94.7444 |
| rgb_stacking_000031_translate_L16 | 0.638600 | 90.7661 | 96.9205 | 0.638600 | 90.7666 | 96.9210 |
| rgb_stacking_000032_translate_L16 | 0.541500 | 77.0979 | 95.2658 | 0.541600 | 77.0995 | 95.2658 |
| rgb_stacking_000033_translate_L16 | 0.332500 | 65.4765 | 88.2388 | 0.332600 | 65.4906 | 88.2453 |
| rgb_stacking_000034_translate_L16 | 0.584500 | 80.3950 | 93.7876 | 0.584600 | 80.4065 | 93.7907 |
| rgb_stacking_000035_translate_L16 | 0.699000 | 83.4475 | 94.0138 | 0.697200 | 83.4593 | 93.9716 |
| rgb_stacking_000036_translate_L16 | 0.620100 | 58.9232 | 88.4174 | 0.620100 | 58.9235 | 88.4170 |
| rgb_stacking_000037_translate_L16 | 0.593100 | 72.3834 | 90.3363 | 0.592400 | 72.3868 | 90.3328 |
| rgb_stacking_000038_translate_L16 | 0.577500 | 74.5647 | 92.9909 | 0.577800 | 74.5705 | 92.9984 |
| rgb_stacking_000039_translate_L16 | 0.317100 | 76.4334 | 92.7308 | 0.317100 | 76.4390 | 92.7256 |
| rgb_stacking_000040_translate_L16 | 0.369500 | 65.0294 | 85.8598 | 0.369400 | 65.0267 | 85.8488 |
| rgb_stacking_000041_translate_L16 | 0.459900 | 68.9908 | 91.0969 | 0.458900 | 68.9848 | 91.0878 |
| rgb_stacking_000042_translate_L16 | 0.505000 | 81.6140 | 94.7336 | 0.504300 | 81.6257 | 94.7166 |
| rgb_stacking_000043_translate_L16 | 0.412900 | 73.5852 | 92.9512 | 0.412800 | 73.5896 | 92.9436 |
| rgb_stacking_000044_translate_L16 | 0.717400 | 79.0163 | 95.6222 | 0.717400 | 79.0162 | 95.6219 |
| rgb_stacking_000045_translate_L16 | 0.553900 | 71.9526 | 88.9332 | 0.554300 | 72.0075 | 88.9885 |
| rgb_stacking_000046_translate_L16 | 0.696300 | 77.0497 | 92.9137 | 0.696500 | 77.0549 | 92.8999 |
| rgb_stacking_000047_translate_L16 | 0.330400 | 57.0724 | 81.9284 | 0.330400 | 57.0735 | 81.9303 |
| rgb_stacking_000048_translate_L16 | 0.767200 | 83.4160 | 97.5107 | 0.767200 | 83.4177 | 97.5123 |
| rgb_stacking_000049_translate_L16 | 0.511500 | 68.7705 | 87.9297 | 0.511000 | 68.7816 | 87.9367 |
