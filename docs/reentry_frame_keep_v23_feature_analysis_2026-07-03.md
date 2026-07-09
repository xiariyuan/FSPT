# ReEntry Frame-Keep V2.3 Feature Separability Analysis

## Dataset summary

| split | samples | feature_dim | safe16+ | gt_visible+ | safe8+ | utility+ | mean utility |
|---|---|---|---|---|---|---|---|
| train | 95925 | 49 | 0.6761 | 0.7235 | 0.5560 | 0.4061 | -0.1694 |
| val | 46041 | 49 | 0.7710 | 0.8209 | 0.6080 | 0.3312 | -0.2429 |

## Top feature separability for `y_safe16` on validation

| feature | best AUC | dir | AP | pos mean | neg mean | effect |
|---|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.7941 | + | 0.9263 | 0.4453 | 0.2435 | 1.1849 |
| frame_base_border_dist_norm | 0.7495 | + | 0.9028 | 0.4482 | 0.2731 | 0.9526 |
| frame_override_y_norm | 0.7251 | + | 0.8555 | 0.2393 | 0.1501 | 0.8715 |
| event_gate_prob | 0.7102 | + | 0.8779 | 0.3655 | 0.1879 | 0.6888 |
| frame_base_y_norm | 0.6970 | + | 0.8295 | 0.2390 | 0.1616 | 0.7184 |
| frame_base_speed_norm | 0.6938 | - | 0.8706 | 0.0248 | 0.0421 | -0.2861 |
| frame_override_speed_norm | 0.6864 | - | 0.8620 | 0.0251 | 0.0428 | -0.2633 |
| frame_base_invisible_run_norm | 0.6639 | - | 0.8526 | 1.4080 | 2.0853 | -0.5275 |
| trigger_t_norm | 0.6622 | + | 0.8525 | 0.7000 | 0.5979 | 0.5287 |
| frame_override_visible_run_norm | 0.6594 | + | 0.8739 | 2.6663 | 1.5641 | 0.6477 |
| frame_base_acc_norm | 0.6537 | - | 0.8483 | 0.0159 | 0.0172 | -0.0275 |
| frame_speed_diff_norm | 0.6516 | - | 0.8463 | 0.0060 | 0.0120 | -0.2731 |
| segment_prob_max | 0.6514 | + | 0.8411 | 0.6546 | 0.5427 | 0.5385 |
| segment_prob_min | 0.6512 | + | 0.8461 | 0.4695 | 0.3600 | 0.5265 |
| frame_override_acc_norm | 0.6464 | - | 0.8465 | 0.0163 | 0.0148 | 0.0281 |
| frame_base_override_dist_norm | 0.6370 | - | 0.8134 | 0.0581 | 0.1835 | -0.6440 |
| segment_prob_mean | 0.6353 | + | 0.8362 | 0.5612 | 0.4680 | 0.4770 |
| v1_prob | 0.6330 | + | 0.8404 | 0.5591 | 0.4611 | 0.4772 |
| frame_base_x_norm | 0.6078 | - | 0.8529 | 0.5180 | 0.5776 | -0.3493 |
| frame_override_x_norm | 0.6066 | - | 0.8381 | 0.5119 | 0.5827 | -0.3899 |

Interpretation: strongest single feature AUC is 0.794, so numeric features contain a meaningful signal for `y_safe16`.

## Top feature separability for `y_gt_visible` on validation

| feature | best AUC | dir | AP | pos mean | neg mean | effect |
|---|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8000 | + | 0.9460 | 0.4361 | 0.2295 | 1.2370 |
| frame_base_border_dist_norm | 0.7840 | + | 0.9412 | 0.4439 | 0.2439 | 1.1451 |
| frame_override_y_norm | 0.7823 | + | 0.9092 | 0.2386 | 0.1283 | 1.1254 |
| frame_base_y_norm | 0.7733 | + | 0.9136 | 0.2406 | 0.1327 | 1.0829 |
| frame_override_speed_norm | 0.7672 | - | 0.9300 | 0.0255 | 0.0462 | -0.3759 |
| frame_base_speed_norm | 0.7577 | - | 0.9261 | 0.0252 | 0.0452 | -0.3857 |
| trigger_t_norm | 0.7270 | + | 0.9157 | 0.7029 | 0.5563 | 0.7792 |
| frame_base_acc_norm | 0.7055 | - | 0.9080 | 0.0161 | 0.0169 | -0.0200 |
| frame_override_acc_norm | 0.7043 | - | 0.9127 | 0.0164 | 0.0140 | 0.0472 |
| frame_speed_diff_norm | 0.6760 | - | 0.8933 | 0.0065 | 0.0118 | -0.2720 |
| event_gate_prob | 0.6603 | + | 0.8922 | 0.3507 | 0.2064 | 0.5561 |
| frame_override_visible_run_norm | 0.6591 | + | 0.9051 | 2.6110 | 1.5103 | 0.6596 |
| frame_base_invisible_run_norm | 0.6531 | - | 0.8851 | 1.4539 | 2.0639 | -0.4764 |
| query_t_norm | 0.6348 | + | 0.8856 | 0.2805 | 0.1916 | 0.5108 |
| segment_prob_min | 0.6285 | + | 0.8725 | 0.4611 | 0.3679 | 0.4391 |
| frame_base_override_ydiff_norm | 0.6070 | + | 0.8587 | -0.0080 | -0.0176 | 0.0857 |
| raw_recovery_count_norm | 0.6024 | + | 0.8194 | 0.5207 | 0.4871 | 0.3179 |
| segment_prob_max | 0.5997 | + | 0.8613 | 0.6420 | 0.5694 | 0.3470 |
| segment_len_norm | 0.5982 | + | 0.8186 | 0.5066 | 0.4708 | 0.2740 |
| segment_prob_mean | 0.5900 | + | 0.8601 | 0.5511 | 0.4884 | 0.3192 |

Interpretation: strongest single feature AUC is 0.800, so numeric features contain a meaningful signal for `y_gt_visible`.

## Top feature separability for `y_safe8` on validation

| feature | best AUC | dir | AP | pos mean | neg mean | effect |
|---|---|---|---|---|---|---|
| event_gate_prob | 0.8139 | + | 0.8492 | 0.4332 | 0.1567 | 1.1461 |
| segment_prob_max | 0.7582 | + | 0.7782 | 0.7055 | 0.5103 | 1.0136 |
| segment_prob_min | 0.7498 | + | 0.7938 | 0.5155 | 0.3341 | 0.9418 |
| segment_prob_mean | 0.7485 | + | 0.7766 | 0.6067 | 0.4361 | 0.9445 |
| v1_prob | 0.7416 | + | 0.7800 | 0.6057 | 0.4295 | 0.9212 |
| frame_base_override_dist_norm | 0.7328 | - | 0.7505 | 0.0472 | 0.1482 | -0.6464 |
| frame_override_border_dist_norm | 0.6436 | + | 0.7427 | 0.4368 | 0.3405 | 0.5021 |
| frame_base_invisible_run_norm | 0.6406 | - | 0.7273 | 1.3225 | 1.9362 | -0.4784 |
| frame_base_acc_norm | 0.6379 | - | 0.7070 | 0.0136 | 0.0203 | -0.1267 |
| frame_override_visible_run_norm | 0.6234 | + | 0.6946 | 2.7120 | 1.9515 | 0.4059 |
| frame_speed_diff_norm | 0.6233 | - | 0.6927 | 0.0052 | 0.0108 | -0.2803 |
| frame_base_override_xdiff_norm | 0.6078 | + | 0.6220 | -0.0139 | -0.0149 | 0.0081 |
| frame_base_speed_norm | 0.6029 | - | 0.6999 | 0.0254 | 0.0341 | -0.1374 |
| frame_override_y_norm | 0.6025 | + | 0.6725 | 0.2360 | 0.1923 | 0.4130 |
| segment_len_norm | 0.5893 | + | 0.7025 | 0.5152 | 0.4769 | 0.3053 |
| raw_recovery_count_norm | 0.5878 | + | 0.7023 | 0.5251 | 0.4986 | 0.2675 |
| frame_base_border_dist_norm | 0.5849 | + | 0.6856 | 0.4350 | 0.3665 | 0.3423 |
| frame_base_y_norm | 0.5811 | + | 0.6460 | 0.2352 | 0.1997 | 0.3272 |
| frame_base_override_ydiff_norm | 0.5694 | + | 0.6070 | 0.0030 | -0.0296 | 0.2396 |
| dist_to_segment_end_norm | 0.5574 | + | 0.6599 | 0.4993 | 0.4347 | 0.2008 |

Interpretation: strongest single feature AUC is 0.814, so numeric features contain a meaningful signal for `y_safe8`.

## Top feature separability for `y_utility` on validation

| feature | best AUC | dir | AP | pos mean | neg mean | effect |
|---|---|---|---|---|---|---|
| event_gate_prob | 0.8255 | + | 0.6813 | 0.5436 | 0.2165 | 1.3424 |
| segment_prob_max | 0.7819 | + | 0.5713 | 0.7635 | 0.5623 | 1.1236 |
| segment_prob_mean | 0.7628 | + | 0.5624 | 0.6585 | 0.4811 | 1.0340 |
| v1_prob | 0.7587 | + | 0.5665 | 0.6600 | 0.4756 | 1.0075 |
| frame_base_override_dist_norm | 0.7509 | - | 0.5341 | 0.0354 | 0.1123 | -0.5982 |
| segment_prob_min | 0.7467 | + | 0.5805 | 0.5669 | 0.3837 | 0.9457 |
| frame_base_acc_norm | 0.6751 | - | 0.4733 | 0.0103 | 0.0192 | -0.1858 |
| frame_override_border_dist_norm | 0.6662 | + | 0.5272 | 0.4656 | 0.3661 | 0.5154 |
| frame_base_speed_norm | 0.6411 | - | 0.4702 | 0.0213 | 0.0325 | -0.1922 |
| trigger_t_norm | 0.6367 | - | 0.4398 | 0.6176 | 0.7059 | -0.4524 |
| frame_base_invisible_run_norm | 0.6300 | - | 0.4852 | 1.2662 | 1.7101 | -0.3473 |
| frame_speed_diff_norm | 0.6222 | - | 0.4166 | 0.0045 | 0.0089 | -0.2652 |
| frame_base_override_xdiff_norm | 0.6203 | + | 0.3701 | -0.0060 | -0.0184 | 0.1166 |
| frame_base_border_dist_norm | 0.6175 | + | 0.4677 | 0.4637 | 0.3806 | 0.4195 |
| frame_time_since_query_norm | 0.5969 | - | 0.4466 | 0.3999 | 0.4743 | -0.3316 |
| frame_minus_query_norm | 0.5969 | - | 0.4466 | 0.3906 | 0.4632 | -0.3316 |
| frame_override_visible_run_norm | 0.5957 | + | 0.3978 | 2.8102 | 2.2176 | 0.3092 |
| frame_override_x_norm | 0.5868 | + | 0.3435 | 0.5545 | 0.5150 | 0.2279 |
| frame_base_past8_invis_rate | 0.5818 | - | 0.4792 | 0.8799 | 0.9645 | -0.4203 |
| frame_base_x_norm | 0.5801 | + | 0.3429 | 0.5560 | 0.5197 | 0.2162 |

Interpretation: strongest single feature AUC is 0.825, so numeric features contain a meaningful signal for `y_utility`.

## Top feature separability for `y_safe4` on validation

| feature | best AUC | dir | AP | pos mean | neg mean | effect |
|---|---|---|---|---|---|---|
| event_gate_prob | 0.8255 | + | 0.6813 | 0.5436 | 0.2165 | 1.3424 |
| segment_prob_max | 0.7819 | + | 0.5713 | 0.7635 | 0.5623 | 1.1236 |
| segment_prob_mean | 0.7628 | + | 0.5624 | 0.6585 | 0.4811 | 1.0340 |
| v1_prob | 0.7587 | + | 0.5665 | 0.6600 | 0.4756 | 1.0075 |
| frame_base_override_dist_norm | 0.7509 | - | 0.5341 | 0.0354 | 0.1123 | -0.5982 |
| segment_prob_min | 0.7467 | + | 0.5805 | 0.5669 | 0.3837 | 0.9457 |
| frame_base_acc_norm | 0.6751 | - | 0.4733 | 0.0103 | 0.0192 | -0.1858 |
| frame_override_border_dist_norm | 0.6662 | + | 0.5272 | 0.4656 | 0.3661 | 0.5154 |
| frame_base_speed_norm | 0.6411 | - | 0.4702 | 0.0213 | 0.0325 | -0.1922 |
| trigger_t_norm | 0.6367 | - | 0.4398 | 0.6176 | 0.7059 | -0.4524 |
| frame_base_invisible_run_norm | 0.6300 | - | 0.4852 | 1.2662 | 1.7101 | -0.3473 |
| frame_speed_diff_norm | 0.6222 | - | 0.4166 | 0.0045 | 0.0089 | -0.2652 |
| frame_base_override_xdiff_norm | 0.6203 | + | 0.3701 | -0.0060 | -0.0184 | 0.1166 |
| frame_base_border_dist_norm | 0.6175 | + | 0.4677 | 0.4637 | 0.3806 | 0.4195 |
| frame_time_since_query_norm | 0.5969 | - | 0.4466 | 0.3999 | 0.4743 | -0.3316 |
| frame_minus_query_norm | 0.5969 | - | 0.4466 | 0.3906 | 0.4632 | -0.3316 |
| frame_override_visible_run_norm | 0.5957 | + | 0.3978 | 2.8102 | 2.2176 | 0.3092 |
| frame_override_x_norm | 0.5868 | + | 0.3435 | 0.5545 | 0.5150 | 0.2279 |
| frame_base_past8_invis_rate | 0.5818 | - | 0.4792 | 0.8799 | 0.9645 | -0.4203 |
| frame_base_x_norm | 0.5801 | + | 0.3429 | 0.5560 | 0.5197 | 0.2162 |

Interpretation: strongest single feature AUC is 0.825, so numeric features contain a meaningful signal for `y_safe4`.

## Largest train/validation feature shifts

| feature | train mean | val mean | train std | val std | std gap |
|---|---|---|---|---|---|
| frame_base_y_norm | 0.1833 | 0.2213 | 0.1035 | 0.1085 | 0.3668 |
| frame_override_y_norm | 0.1820 | 0.2188 | 0.1026 | 0.1066 | 0.3595 |
| frame_base_override_dist_norm | 0.0545 | 0.0868 | 0.0911 | 0.1496 | 0.3544 |
| frame_base_x_norm | 0.4741 | 0.5317 | 0.1702 | 0.1777 | 0.3385 |
| frame_override_x_norm | 0.4734 | 0.5281 | 0.1729 | 0.1845 | 0.3163 |
| frame_base_border_dist_norm | 0.3479 | 0.4081 | 0.1925 | 0.2009 | 0.3127 |
| frame_override_visible_run_norm | 1.9176 | 2.4139 | 1.6658 | 1.9249 | 0.2979 |
| frame_override_border_dist_norm | 0.3440 | 0.3991 | 0.1902 | 0.1964 | 0.2894 |
| segment_prob_mean | 0.5940 | 0.5398 | 0.2325 | 0.1961 | -0.2330 |
| v1_prob | 0.5888 | 0.5366 | 0.2408 | 0.2074 | -0.2167 |
| segment_prob_max | 0.6754 | 0.6290 | 0.2227 | 0.2107 | -0.2085 |
| segment_prob_min | 0.4916 | 0.4444 | 0.2487 | 0.2120 | -0.1897 |
| frame_override_speed_norm | 0.0354 | 0.0292 | 0.0393 | 0.0668 | -0.1578 |
| frame_time_since_query_norm | 0.4142 | 0.4497 | 0.2340 | 0.2242 | 0.1518 |
| frame_minus_query_norm | 0.4045 | 0.4392 | 0.2285 | 0.2189 | 0.1518 |
| frame_base_speed_norm | 0.0354 | 0.0288 | 0.0444 | 0.0612 | -0.1479 |
| query_t_norm | 0.2958 | 0.2646 | 0.2181 | 0.1876 | -0.1429 |
| frame_base_override_xdiff_norm | -0.0026 | -0.0143 | 0.0887 | 0.1197 | -0.1316 |
| event_gate_prob | 0.3624 | 0.3248 | 0.2947 | 0.2831 | -0.1274 |
| candidate_len_norm | 0.5095 | 0.5212 | 0.1057 | 0.0910 | 0.1107 |
| raw_recovery_count_norm | 0.5046 | 0.5147 | 0.1118 | 0.0976 | 0.0904 |
| frame_override_future8_vis_rate | 0.9382 | 0.9539 | 0.1818 | 0.1548 | 0.0865 |
| frame_base_past8_invis_rate | 0.9196 | 0.9365 | 0.2110 | 0.1872 | 0.0800 |
| frame_override_acc_norm | 0.0139 | 0.0160 | 0.0269 | 0.0632 | 0.0765 |
| frame_base_override_ydiff_norm | -0.0054 | -0.0097 | 0.0581 | 0.1237 | -0.0745 |

## Validation subgroup analysis for `y_safe16`

### v1_prob

| bin | count | rate | positive rate | value mean |
|---|---|---|---|---|
| [0.10,0.15) | 1416 | 0.0308 | 0.4640 | 0.1273 |
| [0.15,0.30) | 5582 | 0.1212 | 0.6695 | 0.2279 |
| [0.30,0.50) | 12908 | 0.2804 | 0.7206 | 0.4095 |
| [0.50,+inf) | 26135 | 0.5676 | 0.8342 | 0.6876 |

### event_gate_prob

| bin | count | rate | positive rate | value mean |
|---|---|---|---|---|
| [-inf,0.005) | 2373 | 0.0515 | 0.1344 | 0.0016 |
| [0.005,0.01) | 882 | 0.0192 | 0.5374 | 0.0076 |
| [0.01,0.02) | 1550 | 0.0337 | 0.6516 | 0.0148 |
| [0.02,+inf) | 41236 | 0.8956 | 0.8171 | 0.3619 |

### segment_len_norm

| bin | count | rate | positive rate | value mean |
|---|---|---|---|---|
| [0,0.0625) len<2 | 172 | 0.0037 | 0.5407 | 0.0312 |
| [0.0625,0.125) len2-3 | 847 | 0.0184 | 0.6387 | 0.0857 |
| [0.125,0.28125) len4-8 | 2904 | 0.0631 | 0.6618 | 0.1905 |
| [0.28125,+inf) len9+ | 42118 | 0.9148 | 0.7821 | 0.5318 |

### position_in_segment_norm

| bin | count | rate | positive rate | value mean |
|---|---|---|---|---|
| start-ish | 6140 | 0.1334 | 0.7679 | 0.0242 |
| early-mid | 14899 | 0.3236 | 0.7868 | 0.2678 |
| late-mid | 19389 | 0.4211 | 0.7690 | 0.6750 |
| end-ish | 5613 | 0.1219 | 0.7395 | 0.9746 |

### dist_to_segment_start_norm

| bin | count | rate | positive rate | value mean |
|---|---|---|---|---|
| 0 | 3749 | 0.0814 | 0.7471 | 0.0000 |
| (0,0.125) | 3435 | 0.0746 | 0.7578 | 0.0625 |
| [0.125,0.5) | 17317 | 0.3761 | 0.7707 | 0.2728 |
| [0.5,+inf) | 21540 | 0.4678 | 0.7775 | 0.7622 |

### dist_to_segment_end_norm

| bin | count | rate | positive rate | value mean |
|---|---|---|---|---|
| 0 | 3493 | 0.0759 | 0.7097 | 0.0000 |
| (0,0.125) | 3331 | 0.0723 | 0.7319 | 0.0625 |
| [0.125,0.5) | 17118 | 0.3718 | 0.7565 | 0.2732 |
| [0.5,+inf) | 22099 | 0.4800 | 0.7978 | 0.7665 |

### frame_base_override_dist_norm

| bin | count | rate | positive rate | value mean |
|---|---|---|---|---|
| [0,0.25) | 43703 | 0.9492 | 0.8054 | 0.0580 |
| [0.25,0.5) | 963 | 0.0209 | 0.2658 | 0.3372 |
| [0.5,1.0) | 1050 | 0.0228 | 0.0190 | 0.7467 |
| [1.0,+inf) | 325 | 0.0071 | 0.0708 | 1.0921 |

## Decision note

The current numeric features show useful safe16 separability. Next step should be improving training/selection: class-balanced or calibrated model, checkpoint selection by downstream AJ_RD/AJ, and possibly a temporal model over recovery segments.
