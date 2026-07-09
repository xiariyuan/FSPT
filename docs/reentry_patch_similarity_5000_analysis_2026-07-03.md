# ReEntry Patch Similarity Feature Analysis

Samples: `5000`
Patch feature dim: `51`; numeric dim: `49`

## `y_safe16` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s33_grad_mad | 0.9216 | - | 0.9670 | 0.0259 | 0.0627 |
| query_base_s17_grad_mad | 0.9174 | - | 0.9559 | 0.0236 | 0.0736 |
| last_visible_base_s17_grad_mad | 0.9015 | - | 0.9352 | 0.0285 | 0.0696 |
| query_base_s9_ncc | 0.8983 | + | 0.9515 | 0.8818 | 0.4625 |
| query_base_s17_ncc | 0.8982 | + | 0.9497 | 0.8436 | 0.4931 |
| last_visible_base_s17_ncc | 0.8956 | + | 0.9327 | 0.8325 | 0.4996 |
| last_visible_base_s9_ncc | 0.8917 | + | 0.9379 | 0.8850 | 0.4621 |
| query_base_s9_grad_mad | 0.8883 | - | 0.9446 | 0.0220 | 0.0780 |
| last_visible_base_s9_grad_mad | 0.8758 | - | 0.9263 | 0.0241 | 0.0683 |
| last_visible_base_s33_grad_mad | 0.8709 | - | 0.9343 | 0.0311 | 0.0586 |
| query_base_s33_ncc | 0.8678 | + | 0.9492 | 0.7519 | 0.4492 |
| query_base_s17_mad | 0.8357 | - | 0.9052 | 0.0911 | 0.1589 |
| last_visible_base_s33_ncc | 0.8292 | + | 0.8954 | 0.7427 | 0.4806 |
| query_base_s17_mse | 0.8081 | - | 0.8919 | 0.0303 | 0.0599 |
| last_visible_base_s17_mad | 0.8029 | - | 0.8642 | 0.0927 | 0.1518 |
| query_base_s9_cosine | 0.7998 | + | 0.9049 | 0.9386 | 0.8301 |
| last_visible_base_s9_cosine | 0.7951 | + | 0.8939 | 0.9380 | 0.8308 |
| query_base_s9_mse | 0.7908 | - | 0.8622 | 0.0193 | 0.0542 |
| query_base_s9_mad | 0.7903 | - | 0.8695 | 0.0746 | 0.1448 |
| last_visible_base_s17_cosine | 0.7740 | + | 0.8524 | 0.9097 | 0.8398 |

## `y_safe16` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8202 | + | 0.9079 | 0.4769 | 0.2653 |
| frame_override_y_norm | 0.8197 | + | 0.9006 | 0.2430 | 0.1352 |
| frame_base_border_dist_norm | 0.7841 | + | 0.8930 | 0.4742 | 0.2996 |
| frame_base_y_norm | 0.7839 | + | 0.8892 | 0.2414 | 0.1520 |
| event_gate_prob | 0.7838 | + | 0.8628 | 0.4007 | 0.1698 |
| frame_base_override_ydiff_norm | 0.7634 | + | 0.8430 | 0.0065 | -0.0671 |
| frame_override_speed_norm | 0.7627 | - | 0.9037 | 0.0241 | 0.0469 |
| frame_base_override_dist_norm | 0.7564 | - | 0.8409 | 0.0329 | 0.1445 |
| frame_base_speed_norm | 0.7456 | - | 0.8919 | 0.0233 | 0.0434 |
| segment_prob_min | 0.7378 | + | 0.8472 | 0.5180 | 0.3348 |

## `y_gt_visible` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s17_grad_mad | 0.9514 | - | 0.9825 | 0.0236 | 0.0774 |
| query_base_s33_grad_mad | 0.9444 | - | 0.9798 | 0.0259 | 0.0652 |
| last_visible_base_s17_grad_mad | 0.9385 | - | 0.9728 | 0.0283 | 0.0730 |
| last_visible_base_s9_ncc | 0.9269 | + | 0.9706 | 0.8852 | 0.4303 |
| query_base_s9_ncc | 0.9199 | + | 0.9681 | 0.8806 | 0.4347 |
| query_base_s9_grad_mad | 0.9187 | - | 0.9672 | 0.0220 | 0.0821 |
| query_base_s17_ncc | 0.9134 | + | 0.9669 | 0.8420 | 0.4717 |
| last_visible_base_s17_ncc | 0.9113 | + | 0.9515 | 0.8308 | 0.4796 |
| last_visible_base_s9_grad_mad | 0.9107 | - | 0.9637 | 0.0240 | 0.0718 |
| last_visible_base_s33_grad_mad | 0.9107 | - | 0.9660 | 0.0309 | 0.0614 |
| query_base_s33_ncc | 0.8752 | + | 0.9561 | 0.7487 | 0.4355 |
| query_base_s17_mad | 0.8548 | - | 0.9300 | 0.0914 | 0.1632 |
| last_visible_base_s33_ncc | 0.8443 | + | 0.9210 | 0.7410 | 0.4658 |
| last_visible_base_s17_mad | 0.8443 | - | 0.9062 | 0.0921 | 0.1577 |
| last_visible_base_s9_cosine | 0.8242 | + | 0.9256 | 0.9383 | 0.8221 |
| query_base_s17_mse | 0.8224 | - | 0.9147 | 0.0305 | 0.0617 |
| query_base_s9_cosine | 0.8134 | + | 0.9204 | 0.9378 | 0.8243 |
| query_base_s9_mse | 0.8031 | - | 0.8810 | 0.0196 | 0.0558 |
| query_base_s9_mad | 0.7994 | - | 0.8841 | 0.0753 | 0.1482 |
| last_visible_base_s17_mse | 0.7922 | - | 0.8762 | 0.0317 | 0.0568 |

## `y_gt_visible` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8349 | + | 0.9188 | 0.4759 | 0.2523 |
| frame_override_y_norm | 0.8344 | + | 0.9117 | 0.2424 | 0.1288 |
| event_gate_prob | 0.8245 | + | 0.8997 | 0.4061 | 0.1378 |
| frame_base_border_dist_norm | 0.7959 | + | 0.9033 | 0.4734 | 0.2889 |
| frame_base_y_norm | 0.7957 | + | 0.8996 | 0.2409 | 0.1468 |
| frame_base_override_dist_norm | 0.7855 | - | 0.8727 | 0.0326 | 0.1535 |
| frame_override_speed_norm | 0.7842 | - | 0.9252 | 0.0240 | 0.0489 |
| segment_prob_min | 0.7748 | + | 0.8926 | 0.5220 | 0.3101 |
| frame_base_override_ydiff_norm | 0.7707 | + | 0.8553 | 0.0062 | -0.0717 |
| frame_base_speed_norm | 0.7693 | - | 0.9166 | 0.0230 | 0.0455 |

## `y_safe8` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s33_grad_mad | 0.9030 | - | 0.9496 | 0.0252 | 0.0598 |
| query_base_s17_grad_mad | 0.8863 | - | 0.9245 | 0.0233 | 0.0687 |
| last_visible_base_s33_grad_mad | 0.8667 | - | 0.9201 | 0.0302 | 0.0575 |
| last_visible_base_s17_grad_mad | 0.8634 | - | 0.8932 | 0.0283 | 0.0653 |
| query_base_s17_ncc | 0.8622 | + | 0.9099 | 0.8480 | 0.5234 |
| query_base_s33_ncc | 0.8610 | + | 0.9386 | 0.7604 | 0.4654 |
| query_base_s9_ncc | 0.8417 | + | 0.9036 | 0.8809 | 0.5115 |
| last_visible_base_s17_ncc | 0.8410 | + | 0.8653 | 0.8348 | 0.5322 |
| last_visible_base_s33_ncc | 0.8395 | + | 0.8816 | 0.7545 | 0.4855 |
| query_base_s9_grad_mad | 0.8390 | - | 0.9035 | 0.0221 | 0.0715 |
| last_visible_base_s9_ncc | 0.8303 | + | 0.8816 | 0.8837 | 0.5125 |
| last_visible_base_s9_grad_mad | 0.8287 | - | 0.8834 | 0.0242 | 0.0631 |
| query_base_s17_mad | 0.8032 | - | 0.8595 | 0.0902 | 0.1532 |
| query_base_s17_mse | 0.7865 | - | 0.8593 | 0.0296 | 0.0580 |
| query_base_s33_mad | 0.7619 | - | 0.8882 | 0.1143 | 0.1650 |
| last_visible_base_s17_mad | 0.7615 | - | 0.8114 | 0.0924 | 0.1458 |
| query_base_s9_mse | 0.7508 | - | 0.8122 | 0.0192 | 0.0504 |
| query_base_s9_cosine | 0.7500 | + | 0.8591 | 0.9371 | 0.8455 |
| query_base_s17_cosine | 0.7437 | + | 0.8410 | 0.9157 | 0.8420 |
| query_base_s9_mad | 0.7433 | - | 0.8106 | 0.0748 | 0.1366 |

## `y_safe8` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8434 | + | 0.9069 | 0.4894 | 0.2629 |
| frame_override_y_norm | 0.8429 | + | 0.8992 | 0.2495 | 0.1337 |
| frame_base_border_dist_norm | 0.8137 | + | 0.8927 | 0.4867 | 0.2931 |
| frame_base_y_norm | 0.8136 | + | 0.8888 | 0.2478 | 0.1486 |
| event_gate_prob | 0.7928 | + | 0.8490 | 0.4126 | 0.1708 |
| segment_prob_min | 0.7568 | + | 0.8230 | 0.5296 | 0.3310 |
| frame_override_speed_norm | 0.7381 | - | 0.8752 | 0.0243 | 0.0440 |
| frame_speed_diff_norm | 0.7356 | - | 0.8279 | 0.0052 | 0.0149 |
| frame_base_override_ydiff_norm | 0.7348 | + | 0.7969 | 0.0067 | -0.0592 |
| frame_base_override_dist_norm | 0.7318 | - | 0.8021 | 0.0318 | 0.1342 |

## `y_utility` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_visible_base_s33_grad_mad | 0.9204 | - | 0.8772 | 0.0199 | 0.0503 |
| query_base_s33_grad_mad | 0.8779 | - | 0.7118 | 0.0174 | 0.0477 |
| last_visible_base_s33_ncc | 0.8736 | + | 0.7654 | 0.8325 | 0.5700 |
| query_base_s33_mad | 0.8694 | - | 0.8215 | 0.0859 | 0.1572 |
| query_base_s33_ncc | 0.8656 | + | 0.7491 | 0.8367 | 0.5635 |
| last_visible_base_s33_mad | 0.8424 | - | 0.6942 | 0.0880 | 0.1540 |
| query_base_s33_mse | 0.8358 | - | 0.7709 | 0.0305 | 0.0634 |
| query_base_s33_cosine | 0.8293 | + | 0.7191 | 0.9116 | 0.8159 |
| last_visible_base_s33_cosine | 0.8289 | + | 0.6987 | 0.9084 | 0.8275 |
| last_visible_base_s33_mse | 0.8172 | - | 0.6760 | 0.0313 | 0.0606 |
| query_base_s33_mean_color_l2 | 0.8057 | - | 0.7500 | 0.0786 | 0.1483 |
| last_visible_base_s17_grad_mad | 0.7724 | - | 0.6011 | 0.0256 | 0.0489 |
| query_base_s33_std_color_l2 | 0.7524 | - | 0.6402 | 0.0776 | 0.1297 |
| query_base_s17_grad_mad | 0.7459 | - | 0.5023 | 0.0220 | 0.0474 |
| query_base_s17_mean_color_l2 | 0.7208 | - | 0.5540 | 0.1064 | 0.1559 |
| last_visible_base_s17_mse | 0.6892 | - | 0.4905 | 0.0293 | 0.0437 |
| query_base_s17_mse | 0.6833 | - | 0.4993 | 0.0276 | 0.0454 |
| last_visible_base_s33_mean_color_l2 | 0.6733 | - | 0.5106 | 0.0708 | 0.0888 |
| last_visible_base_s9_std_color_l2 | 0.6581 | + | 0.4718 | 0.1470 | 0.1013 |
| query_candidate_age_norm | 0.6566 | - | 0.4871 | 0.4878 | 0.6053 |

## `y_utility` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_speed_norm | 0.9039 | - | 0.8314 | 0.0077 | 0.0442 |
| frame_base_speed_norm | 0.9003 | - | 0.8383 | 0.0074 | 0.0419 |
| frame_base_y_norm | 0.8975 | + | 0.7611 | 0.2964 | 0.1681 |
| frame_override_y_norm | 0.8971 | + | 0.7550 | 0.2951 | 0.1630 |
| frame_base_border_dist_norm | 0.8946 | + | 0.7287 | 0.5799 | 0.3320 |
| frame_override_border_dist_norm | 0.8941 | + | 0.7225 | 0.5761 | 0.3216 |
| event_gate_prob | 0.8868 | + | 0.7745 | 0.5682 | 0.1962 |
| frame_base_acc_norm | 0.8634 | - | 0.8123 | 0.0054 | 0.0204 |
| frame_base_override_dist_norm | 0.8413 | - | 0.6797 | 0.0180 | 0.0926 |
| frame_override_acc_norm | 0.8088 | - | 0.7192 | 0.0058 | 0.0170 |

## `y_safe4` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_visible_base_s33_grad_mad | 0.9204 | - | 0.8772 | 0.0199 | 0.0503 |
| query_base_s33_grad_mad | 0.8779 | - | 0.7118 | 0.0174 | 0.0477 |
| last_visible_base_s33_ncc | 0.8736 | + | 0.7654 | 0.8325 | 0.5700 |
| query_base_s33_mad | 0.8694 | - | 0.8215 | 0.0859 | 0.1572 |
| query_base_s33_ncc | 0.8656 | + | 0.7491 | 0.8367 | 0.5635 |
| last_visible_base_s33_mad | 0.8424 | - | 0.6942 | 0.0880 | 0.1540 |
| query_base_s33_mse | 0.8358 | - | 0.7709 | 0.0305 | 0.0634 |
| query_base_s33_cosine | 0.8293 | + | 0.7191 | 0.9116 | 0.8159 |
| last_visible_base_s33_cosine | 0.8289 | + | 0.6987 | 0.9084 | 0.8275 |
| last_visible_base_s33_mse | 0.8172 | - | 0.6760 | 0.0313 | 0.0606 |
| query_base_s33_mean_color_l2 | 0.8057 | - | 0.7500 | 0.0786 | 0.1483 |
| last_visible_base_s17_grad_mad | 0.7724 | - | 0.6011 | 0.0256 | 0.0489 |
| query_base_s33_std_color_l2 | 0.7524 | - | 0.6402 | 0.0776 | 0.1297 |
| query_base_s17_grad_mad | 0.7459 | - | 0.5023 | 0.0220 | 0.0474 |
| query_base_s17_mean_color_l2 | 0.7208 | - | 0.5540 | 0.1064 | 0.1559 |
| last_visible_base_s17_mse | 0.6892 | - | 0.4905 | 0.0293 | 0.0437 |
| query_base_s17_mse | 0.6833 | - | 0.4993 | 0.0276 | 0.0454 |
| last_visible_base_s33_mean_color_l2 | 0.6733 | - | 0.5106 | 0.0708 | 0.0888 |
| last_visible_base_s9_std_color_l2 | 0.6581 | + | 0.4718 | 0.1470 | 0.1013 |
| query_candidate_age_norm | 0.6566 | - | 0.4871 | 0.4878 | 0.6053 |

## `y_safe4` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_speed_norm | 0.9039 | - | 0.8314 | 0.0077 | 0.0442 |
| frame_base_speed_norm | 0.9003 | - | 0.8383 | 0.0074 | 0.0419 |
| frame_base_y_norm | 0.8975 | + | 0.7611 | 0.2964 | 0.1681 |
| frame_override_y_norm | 0.8971 | + | 0.7550 | 0.2951 | 0.1630 |
| frame_base_border_dist_norm | 0.8946 | + | 0.7287 | 0.5799 | 0.3320 |
| frame_override_border_dist_norm | 0.8941 | + | 0.7225 | 0.5761 | 0.3216 |
| event_gate_prob | 0.8868 | + | 0.7745 | 0.5682 | 0.1962 |
| frame_base_acc_norm | 0.8634 | - | 0.8123 | 0.0054 | 0.0204 |
| frame_base_override_dist_norm | 0.8413 | - | 0.6797 | 0.0180 | 0.0926 |
| frame_override_acc_norm | 0.8088 | - | 0.7192 | 0.0058 | 0.0170 |
