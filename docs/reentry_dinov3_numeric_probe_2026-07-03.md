# DINO/ViT + Numeric Probe

Train NPZ: `outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev0_6_dinov3_uniform3000_crop33.npz`
Val NPZ: `outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev7_9_dinov3_uniform3000_crop33.npz`

## Summary, all validation samples

| features | label | classifier | AUC | AP | val positive rate |
|---|---|---|---:|---:|---:|
| numeric | y_safe16 | plain | 0.6572 | 0.8554 | 0.7677 |
| numeric | y_safe16 | balanced | 0.6623 | 0.8578 | 0.7677 |
| numeric | y_gt_visible | plain | 0.6026 | 0.8664 | 0.8177 |
| numeric | y_gt_visible | balanced | 0.6035 | 0.8671 | 0.8177 |
| numeric | y_safe8 | plain | 0.7398 | 0.7898 | 0.6087 |
| numeric | y_safe8 | balanced | 0.7409 | 0.7906 | 0.6087 |
| numeric | y_utility | plain | 0.7900 | 0.6035 | 0.3283 |
| numeric | y_utility | balanced | 0.7904 | 0.6031 | 0.3283 |
| numeric | y_safe4 | plain | 0.7900 | 0.6035 | 0.3283 |
| numeric | y_safe4 | balanced | 0.7904 | 0.6031 | 0.3283 |
| dinov3 | y_safe16 | plain | 0.6895 | 0.8659 | 0.7677 |
| dinov3 | y_safe16 | balanced | 0.6905 | 0.8662 | 0.7677 |
| dinov3 | y_gt_visible | plain | 0.6765 | 0.8972 | 0.8177 |
| dinov3 | y_gt_visible | balanced | 0.6800 | 0.8979 | 0.8177 |
| dinov3 | y_safe8 | plain | 0.7135 | 0.7984 | 0.6087 |
| dinov3 | y_safe8 | balanced | 0.7136 | 0.7989 | 0.6087 |
| dinov3 | y_utility | plain | 0.8162 | 0.7011 | 0.3283 |
| dinov3 | y_utility | balanced | 0.8153 | 0.6993 | 0.3283 |
| dinov3 | y_safe4 | plain | 0.8162 | 0.7011 | 0.3283 |
| dinov3 | y_safe4 | balanced | 0.8153 | 0.6993 | 0.3283 |
| numeric_plus_dinov3 | y_safe16 | plain | 0.6852 | 0.8679 | 0.7677 |
| numeric_plus_dinov3 | y_safe16 | balanced | 0.6903 | 0.8709 | 0.7677 |
| numeric_plus_dinov3 | y_gt_visible | plain | 0.6192 | 0.8748 | 0.8177 |
| numeric_plus_dinov3 | y_gt_visible | balanced | 0.6300 | 0.8804 | 0.8177 |
| numeric_plus_dinov3 | y_safe8 | plain | 0.7538 | 0.8070 | 0.6087 |
| numeric_plus_dinov3 | y_safe8 | balanced | 0.7553 | 0.8085 | 0.6087 |
| numeric_plus_dinov3 | y_utility | plain | 0.8086 | 0.6536 | 0.3283 |
| numeric_plus_dinov3 | y_utility | balanced | 0.8095 | 0.6526 | 0.3283 |
| numeric_plus_dinov3 | y_safe4 | plain | 0.8086 | 0.6536 | 0.3283 |
| numeric_plus_dinov3 | y_safe4 | balanced | 0.8095 | 0.6526 | 0.3283 |

## Hard subsets, balanced classifier

### y_safe16

| subset | features | n | pos rate | AUC | AP |
|---|---|---:|---:|---:|---:|
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric | 2651 | 0.8204 | 0.6071 | 0.8722 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | dinov3 | 2651 | 0.8204 | 0.6837 | 0.8944 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric_plus_dinov3 | 2651 | 0.8204 | 0.6441 | 0.8880 |
| distance_ambiguous_025_050 | numeric | 66 | 0.2424 | 0.5025 | 0.2339 |
| distance_ambiguous_025_050 | dinov3 | 66 | 0.2424 | 0.3600 | 0.2460 |
| distance_ambiguous_025_050 | numeric_plus_dinov3 | 66 | 0.2424 | 0.5775 | 0.2652 |
| low_v1_prob_010_030 | numeric | 452 | 0.6350 | 0.5895 | 0.7001 |
| low_v1_prob_010_030 | dinov3 | 452 | 0.6350 | 0.6709 | 0.7535 |
| low_v1_prob_010_030 | numeric_plus_dinov3 | 452 | 0.6350 | 0.6738 | 0.7660 |
| high_event_low_dist_low_v1 | numeric | 239 | 0.8033 | 0.6991 | 0.8971 |
| high_event_low_dist_low_v1 | dinov3 | 239 | 0.8033 | 0.6395 | 0.8606 |
| high_event_low_dist_low_v1 | numeric_plus_dinov3 | 239 | 0.8033 | 0.7246 | 0.8897 |
| low_segment_max_lt_050 | numeric | 894 | 0.6521 | 0.6379 | 0.7601 |
| low_segment_max_lt_050 | dinov3 | 894 | 0.6521 | 0.5857 | 0.7353 |
| low_segment_max_lt_050 | numeric_plus_dinov3 | 894 | 0.6521 | 0.6826 | 0.7662 |

### y_gt_visible

| subset | features | n | pos rate | AUC | AP |
|---|---|---:|---:|---:|---:|
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric | 2651 | 0.8472 | 0.6050 | 0.8873 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | dinov3 | 2651 | 0.8472 | 0.6917 | 0.9192 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric_plus_dinov3 | 2651 | 0.8472 | 0.6477 | 0.9074 |
| distance_ambiguous_025_050 | numeric | 66 | 0.3636 | 0.3512 | 0.3218 |
| distance_ambiguous_025_050 | dinov3 | 66 | 0.3636 | 0.2698 | 0.2775 |
| distance_ambiguous_025_050 | numeric_plus_dinov3 | 66 | 0.3636 | 0.2579 | 0.2815 |
| low_v1_prob_010_030 | numeric | 452 | 0.7434 | 0.5861 | 0.8090 |
| low_v1_prob_010_030 | dinov3 | 452 | 0.7434 | 0.6469 | 0.8399 |
| low_v1_prob_010_030 | numeric_plus_dinov3 | 452 | 0.7434 | 0.6388 | 0.8233 |
| high_event_low_dist_low_v1 | numeric | 239 | 0.8536 | 0.7706 | 0.9552 |
| high_event_low_dist_low_v1 | dinov3 | 239 | 0.8536 | 0.7080 | 0.9366 |
| high_event_low_dist_low_v1 | numeric_plus_dinov3 | 239 | 0.8536 | 0.8123 | 0.9647 |
| low_segment_max_lt_050 | numeric | 894 | 0.7864 | 0.5866 | 0.8417 |
| low_segment_max_lt_050 | dinov3 | 894 | 0.7864 | 0.5561 | 0.8359 |
| low_segment_max_lt_050 | numeric_plus_dinov3 | 894 | 0.7864 | 0.5711 | 0.8259 |

### y_safe8

| subset | features | n | pos rate | AUC | AP |
|---|---|---:|---:|---:|---:|
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric | 2651 | 0.6741 | 0.6924 | 0.8025 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | dinov3 | 2651 | 0.6741 | 0.7115 | 0.8297 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric_plus_dinov3 | 2651 | 0.6741 | 0.7096 | 0.8202 |
| distance_ambiguous_025_050 | numeric | 66 | 0.2273 | 0.5007 | 0.2283 |
| distance_ambiguous_025_050 | dinov3 | 66 | 0.2273 | 0.3843 | 0.1874 |
| distance_ambiguous_025_050 | numeric_plus_dinov3 | 66 | 0.2273 | 0.5176 | 0.2304 |
| low_v1_prob_010_030 | numeric | 452 | 0.2124 | 0.7384 | 0.3844 |
| low_v1_prob_010_030 | dinov3 | 452 | 0.2124 | 0.5380 | 0.3271 |
| low_v1_prob_010_030 | numeric_plus_dinov3 | 452 | 0.2124 | 0.7575 | 0.4102 |
| high_event_low_dist_low_v1 | numeric | 239 | 0.3305 | 0.7184 | 0.5049 |
| high_event_low_dist_low_v1 | dinov3 | 239 | 0.3305 | 0.5714 | 0.4633 |
| high_event_low_dist_low_v1 | numeric_plus_dinov3 | 239 | 0.3305 | 0.7276 | 0.5234 |
| low_segment_max_lt_050 | numeric | 894 | 0.3121 | 0.6730 | 0.4095 |
| low_segment_max_lt_050 | dinov3 | 894 | 0.3121 | 0.4744 | 0.3383 |
| low_segment_max_lt_050 | numeric_plus_dinov3 | 894 | 0.3121 | 0.6923 | 0.4500 |

### y_utility

| subset | features | n | pos rate | AUC | AP |
|---|---|---:|---:|---:|---:|
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric | 2651 | 0.3644 | 0.7700 | 0.6106 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | dinov3 | 2651 | 0.3644 | 0.8116 | 0.7149 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric_plus_dinov3 | 2651 | 0.3644 | 0.7920 | 0.6606 |
| distance_ambiguous_025_050 | numeric | 66 | 0.0455 | 0.3915 | 0.0490 |
| distance_ambiguous_025_050 | dinov3 | 66 | 0.0455 | 0.6032 | 0.0743 |
| distance_ambiguous_025_050 | numeric_plus_dinov3 | 66 | 0.0455 | 0.3968 | 0.0488 |
| low_v1_prob_010_030 | numeric | 452 | 0.0575 | 0.7579 | 0.1287 |
| low_v1_prob_010_030 | dinov3 | 452 | 0.0575 | 0.7700 | 0.3432 |
| low_v1_prob_010_030 | numeric_plus_dinov3 | 452 | 0.0575 | 0.7934 | 0.1520 |
| high_event_low_dist_low_v1 | numeric | 239 | 0.0795 | 0.7931 | 0.1963 |
| high_event_low_dist_low_v1 | dinov3 | 239 | 0.0795 | 0.8129 | 0.4729 |
| high_event_low_dist_low_v1 | numeric_plus_dinov3 | 239 | 0.0795 | 0.8388 | 0.2270 |
| low_segment_max_lt_050 | numeric | 894 | 0.0962 | 0.7469 | 0.2048 |
| low_segment_max_lt_050 | dinov3 | 894 | 0.0962 | 0.5516 | 0.2203 |
| low_segment_max_lt_050 | numeric_plus_dinov3 | 894 | 0.0962 | 0.7578 | 0.2547 |

### y_safe4

| subset | features | n | pos rate | AUC | AP |
|---|---|---:|---:|---:|---:|
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric | 2651 | 0.3644 | 0.7700 | 0.6106 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | dinov3 | 2651 | 0.3644 | 0.8116 | 0.7149 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | numeric_plus_dinov3 | 2651 | 0.3644 | 0.7920 | 0.6606 |
| distance_ambiguous_025_050 | numeric | 66 | 0.0455 | 0.3915 | 0.0490 |
| distance_ambiguous_025_050 | dinov3 | 66 | 0.0455 | 0.6032 | 0.0743 |
| distance_ambiguous_025_050 | numeric_plus_dinov3 | 66 | 0.0455 | 0.3968 | 0.0488 |
| low_v1_prob_010_030 | numeric | 452 | 0.0575 | 0.7579 | 0.1287 |
| low_v1_prob_010_030 | dinov3 | 452 | 0.0575 | 0.7700 | 0.3432 |
| low_v1_prob_010_030 | numeric_plus_dinov3 | 452 | 0.0575 | 0.7934 | 0.1520 |
| high_event_low_dist_low_v1 | numeric | 239 | 0.0795 | 0.7931 | 0.1963 |
| high_event_low_dist_low_v1 | dinov3 | 239 | 0.0795 | 0.8129 | 0.4729 |
| high_event_low_dist_low_v1 | numeric_plus_dinov3 | 239 | 0.0795 | 0.8388 | 0.2270 |
| low_segment_max_lt_050 | numeric | 894 | 0.0962 | 0.7469 | 0.2048 |
| low_segment_max_lt_050 | dinov3 | 894 | 0.0962 | 0.5516 | 0.2203 |
| low_segment_max_lt_050 | numeric_plus_dinov3 | 894 | 0.0962 | 0.7578 | 0.2547 |
