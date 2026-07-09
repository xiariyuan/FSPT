# ReEntry Patch Similarity Feature Analysis

Samples: `3000`
Patch feature dim: `13`; numeric dim: `49`

## `y_safe16` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| best_ref_candidate_l2 | 0.7188 | - | 0.8915 | 0.6176 | 0.7880 |
| best_ref_candidate_cosine | 0.7188 | + | 0.8915 | 0.7832 | 0.6740 |
| last_candidate_cosine | 0.6998 | + | 0.8825 | 0.7441 | 0.6344 |
| last_candidate_l2 | 0.6998 | - | 0.8825 | 0.6709 | 0.8357 |
| query_candidate_cosine | 0.6647 | + | 0.8719 | 0.7054 | 0.6188 |
| query_candidate_l2 | 0.6647 | - | 0.8719 | 0.7388 | 0.8582 |
| worst_ref_candidate_cosine | 0.6607 | + | 0.8665 | 0.6663 | 0.5792 |
| worst_ref_candidate_l2 | 0.6607 | - | 0.8665 | 0.7921 | 0.9059 |
| last_visible_age_norm | 0.6549 | - | 0.8427 | 0.1803 | 0.2598 |
| query_last_l2 | 0.5869 | + | 0.8251 | 0.6998 | 0.6243 |
| query_last_cosine | 0.5869 | - | 0.8251 | 0.7284 | 0.7713 |
| query_candidate_age_norm | 0.5831 | + | 0.8404 | 0.4558 | 0.3885 |
| has_last_visible_ref | 0.5000 | + | 0.7710 | 1.0000 | 1.0000 |

## `y_safe16` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.7875 | + | 0.9216 | 0.4449 | 0.2482 |
| frame_base_border_dist_norm | 0.7432 | + | 0.8977 | 0.4480 | 0.2778 |
| frame_override_y_norm | 0.7199 | + | 0.8481 | 0.2393 | 0.1526 |
| event_gate_prob | 0.7021 | + | 0.8686 | 0.3646 | 0.1942 |
| frame_base_speed_norm | 0.6976 | - | 0.8723 | 0.0252 | 0.0416 |
| frame_override_speed_norm | 0.6916 | - | 0.8678 | 0.0257 | 0.0419 |
| frame_base_y_norm | 0.6892 | + | 0.8217 | 0.2391 | 0.1642 |
| frame_speed_diff_norm | 0.6729 | - | 0.8548 | 0.0058 | 0.0114 |
| trigger_t_norm | 0.6640 | + | 0.8516 | 0.7014 | 0.5996 |
| frame_base_acc_norm | 0.6627 | - | 0.8488 | 0.0164 | 0.0169 |

## `y_gt_visible` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| best_ref_candidate_cosine | 0.7142 | + | 0.9151 | 0.7772 | 0.6708 |
| best_ref_candidate_l2 | 0.7142 | - | 0.9151 | 0.6272 | 0.7918 |
| last_candidate_cosine | 0.6922 | + | 0.9073 | 0.7378 | 0.6324 |
| last_candidate_l2 | 0.6922 | - | 0.9073 | 0.6805 | 0.8377 |
| query_candidate_cosine | 0.6640 | + | 0.9018 | 0.7016 | 0.6122 |
| query_candidate_l2 | 0.6640 | - | 0.9018 | 0.7445 | 0.8656 |
| worst_ref_candidate_cosine | 0.6614 | + | 0.8973 | 0.6622 | 0.5738 |
| worst_ref_candidate_l2 | 0.6614 | - | 0.8973 | 0.7978 | 0.9115 |
| last_visible_age_norm | 0.6427 | - | 0.8768 | 0.1858 | 0.2567 |
| query_candidate_age_norm | 0.5729 | + | 0.8727 | 0.4511 | 0.3913 |
| query_last_l2 | 0.5383 | + | 0.8521 | 0.6882 | 0.6553 |
| query_last_cosine | 0.5383 | - | 0.8521 | 0.7350 | 0.7535 |
| has_last_visible_ref | 0.5000 | + | 0.7849 | 1.0000 | 1.0000 |

## `y_gt_visible` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.7920 | + | 0.9414 | 0.4358 | 0.2351 |
| frame_base_border_dist_norm | 0.7761 | + | 0.9363 | 0.4439 | 0.2496 |
| frame_override_y_norm | 0.7716 | + | 0.9012 | 0.2385 | 0.1324 |
| frame_override_speed_norm | 0.7659 | - | 0.9286 | 0.0259 | 0.0456 |
| frame_base_y_norm | 0.7624 | + | 0.9064 | 0.2407 | 0.1366 |
| frame_base_speed_norm | 0.7594 | - | 0.9266 | 0.0255 | 0.0449 |
| trigger_t_norm | 0.7291 | + | 0.9134 | 0.7044 | 0.5580 |
| frame_base_acc_norm | 0.7088 | - | 0.9088 | 0.0165 | 0.0166 |
| frame_override_acc_norm | 0.7049 | - | 0.9116 | 0.0164 | 0.0134 |
| frame_speed_diff_norm | 0.6952 | - | 0.8967 | 0.0060 | 0.0119 |

## `y_safe8` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_candidate_cosine | 0.7545 | + | 0.8295 | 0.7718 | 0.6358 |
| last_candidate_l2 | 0.7545 | - | 0.8295 | 0.6272 | 0.8366 |
| best_ref_candidate_cosine | 0.7358 | + | 0.8229 | 0.8021 | 0.6889 |
| best_ref_candidate_l2 | 0.7358 | - | 0.8229 | 0.5838 | 0.7713 |
| worst_ref_candidate_l2 | 0.6871 | - | 0.7992 | 0.7655 | 0.9009 |
| worst_ref_candidate_cosine | 0.6871 | + | 0.7992 | 0.6848 | 0.5859 |
| query_candidate_cosine | 0.6511 | + | 0.7773 | 0.7150 | 0.6390 |
| query_candidate_l2 | 0.6511 | - | 0.7773 | 0.7222 | 0.8356 |
| last_visible_age_norm | 0.6340 | - | 0.7170 | 0.1702 | 0.2430 |
| query_last_cosine | 0.5649 | + | 0.6307 | 0.7526 | 0.7162 |
| query_last_l2 | 0.5649 | - | 0.6307 | 0.6646 | 0.7096 |
| query_candidate_age_norm | 0.5497 | - | 0.6555 | 0.4261 | 0.4621 |
| has_last_visible_ref | 0.5000 | + | 0.6766 | 1.0000 | 1.0000 |

## `y_safe8` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| event_gate_prob | 0.8071 | + | 0.8377 | 0.4304 | 0.1611 |
| segment_prob_max | 0.7583 | + | 0.7821 | 0.7047 | 0.5097 |
| segment_prob_mean | 0.7458 | + | 0.7750 | 0.6046 | 0.4358 |
| segment_prob_min | 0.7433 | + | 0.7860 | 0.5121 | 0.3357 |
| v1_prob | 0.7379 | + | 0.7782 | 0.6040 | 0.4307 |
| frame_base_override_dist_norm | 0.7222 | - | 0.7426 | 0.0479 | 0.1475 |
| frame_base_acc_norm | 0.6441 | - | 0.7137 | 0.0133 | 0.0215 |
| frame_override_border_dist_norm | 0.6394 | + | 0.7407 | 0.4355 | 0.3427 |
| frame_base_invisible_run_norm | 0.6350 | - | 0.7168 | 1.3303 | 1.9230 |
| frame_speed_diff_norm | 0.6292 | - | 0.6971 | 0.0049 | 0.0103 |

## `y_utility` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_candidate_cosine | 0.8266 | + | 0.6628 | 0.8340 | 0.6622 |
| last_candidate_l2 | 0.8266 | - | 0.6628 | 0.5247 | 0.7993 |
| best_ref_candidate_l2 | 0.7816 | - | 0.6383 | 0.5026 | 0.7327 |
| best_ref_candidate_cosine | 0.7816 | + | 0.6383 | 0.8461 | 0.7146 |
| worst_ref_candidate_cosine | 0.7770 | + | 0.7159 | 0.7465 | 0.5970 |
| worst_ref_candidate_l2 | 0.7770 | - | 0.7159 | 0.6768 | 0.8878 |
| query_candidate_cosine | 0.7120 | + | 0.6567 | 0.7586 | 0.6494 |
| query_candidate_l2 | 0.7120 | - | 0.6567 | 0.6547 | 0.8212 |
| query_last_cosine | 0.7094 | + | 0.4975 | 0.8133 | 0.7017 |
| query_last_l2 | 0.7094 | - | 0.4974 | 0.5659 | 0.7391 |
| last_visible_age_norm | 0.6172 | - | 0.4594 | 0.1652 | 0.2151 |
| query_candidate_age_norm | 0.5946 | - | 0.4398 | 0.3929 | 0.4633 |
| has_last_visible_ref | 0.5000 | + | 0.3725 | 1.0000 | 1.0000 |

## `y_utility` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| event_gate_prob | 0.8178 | + | 0.6613 | 0.5406 | 0.2196 |
| segment_prob_max | 0.7817 | + | 0.5730 | 0.7625 | 0.5628 |
| segment_prob_mean | 0.7615 | + | 0.5627 | 0.6566 | 0.4808 |
| v1_prob | 0.7579 | + | 0.5669 | 0.6589 | 0.4762 |
| frame_base_override_dist_norm | 0.7515 | - | 0.5316 | 0.0363 | 0.1116 |
| segment_prob_min | 0.7405 | + | 0.5742 | 0.5629 | 0.3845 |
| frame_base_acc_norm | 0.6729 | - | 0.4696 | 0.0115 | 0.0190 |
| frame_override_border_dist_norm | 0.6613 | + | 0.5218 | 0.4641 | 0.3674 |
| frame_base_speed_norm | 0.6415 | - | 0.4728 | 0.0222 | 0.0324 |
| trigger_t_norm | 0.6290 | - | 0.4341 | 0.6209 | 0.7055 |

## `y_safe4` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_candidate_cosine | 0.8266 | + | 0.6628 | 0.8340 | 0.6622 |
| last_candidate_l2 | 0.8266 | - | 0.6628 | 0.5247 | 0.7993 |
| best_ref_candidate_l2 | 0.7816 | - | 0.6383 | 0.5026 | 0.7327 |
| best_ref_candidate_cosine | 0.7816 | + | 0.6383 | 0.8461 | 0.7146 |
| worst_ref_candidate_cosine | 0.7770 | + | 0.7159 | 0.7465 | 0.5970 |
| worst_ref_candidate_l2 | 0.7770 | - | 0.7159 | 0.6768 | 0.8878 |
| query_candidate_cosine | 0.7120 | + | 0.6567 | 0.7586 | 0.6494 |
| query_candidate_l2 | 0.7120 | - | 0.6567 | 0.6547 | 0.8212 |
| query_last_cosine | 0.7094 | + | 0.4975 | 0.8133 | 0.7017 |
| query_last_l2 | 0.7094 | - | 0.4974 | 0.5659 | 0.7391 |
| last_visible_age_norm | 0.6172 | - | 0.4594 | 0.1652 | 0.2151 |
| query_candidate_age_norm | 0.5946 | - | 0.4398 | 0.3929 | 0.4633 |
| has_last_visible_ref | 0.5000 | + | 0.3725 | 1.0000 | 1.0000 |

## `y_safe4` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| event_gate_prob | 0.8178 | + | 0.6613 | 0.5406 | 0.2196 |
| segment_prob_max | 0.7817 | + | 0.5730 | 0.7625 | 0.5628 |
| segment_prob_mean | 0.7615 | + | 0.5627 | 0.6566 | 0.4808 |
| v1_prob | 0.7579 | + | 0.5669 | 0.6589 | 0.4762 |
| frame_base_override_dist_norm | 0.7515 | - | 0.5316 | 0.0363 | 0.1116 |
| segment_prob_min | 0.7405 | + | 0.5742 | 0.5629 | 0.3845 |
| frame_base_acc_norm | 0.6729 | - | 0.4696 | 0.0115 | 0.0190 |
| frame_override_border_dist_norm | 0.6613 | + | 0.5218 | 0.4641 | 0.3674 |
| frame_base_speed_norm | 0.6415 | - | 0.4728 | 0.0222 | 0.0324 |
| trigger_t_norm | 0.6290 | - | 0.4341 | 0.6209 | 0.7055 |
