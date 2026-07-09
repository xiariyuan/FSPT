# ReEntry Patch Similarity Feature Analysis

Samples: `1500`
Patch feature dim: `13`; numeric dim: `49`

## `y_safe16` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| best_ref_candidate_cosine | 0.7344 | + | 0.9021 | 0.7846 | 0.6682 |
| best_ref_candidate_l2 | 0.7344 | - | 0.9021 | 0.6156 | 0.7955 |
| last_candidate_cosine | 0.7095 | + | 0.8911 | 0.7456 | 0.6299 |
| last_candidate_l2 | 0.7095 | - | 0.8911 | 0.6691 | 0.8409 |
| last_visible_age_norm | 0.6677 | - | 0.8615 | 0.1792 | 0.2633 |
| query_candidate_l2 | 0.6677 | - | 0.8799 | 0.7402 | 0.8623 |
| query_candidate_cosine | 0.6677 | + | 0.8799 | 0.7044 | 0.6161 |
| worst_ref_candidate_cosine | 0.6614 | + | 0.8706 | 0.6654 | 0.5778 |
| worst_ref_candidate_l2 | 0.6614 | - | 0.8706 | 0.7937 | 0.9077 |
| query_last_l2 | 0.5944 | + | 0.8405 | 0.7004 | 0.6193 |
| query_last_cosine | 0.5944 | - | 0.8405 | 0.7281 | 0.7751 |
| query_candidate_age_norm | 0.5853 | + | 0.8484 | 0.4556 | 0.3875 |
| has_last_visible_ref | 0.5000 | + | 0.7740 | 1.0000 | 1.0000 |

## `y_safe16` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.7971 | + | 0.9318 | 0.4454 | 0.2443 |
| frame_base_border_dist_norm | 0.7535 | + | 0.9102 | 0.4484 | 0.2749 |
| frame_override_y_norm | 0.7289 | + | 0.8673 | 0.2403 | 0.1503 |
| event_gate_prob | 0.7187 | + | 0.8881 | 0.3717 | 0.1832 |
| frame_base_y_norm | 0.6995 | + | 0.8387 | 0.2398 | 0.1627 |
| frame_override_speed_norm | 0.6915 | - | 0.8692 | 0.0276 | 0.0446 |
| frame_base_speed_norm | 0.6896 | - | 0.8687 | 0.0266 | 0.0416 |
| frame_override_visible_run_norm | 0.6758 | + | 0.8839 | 2.7202 | 1.5306 |
| frame_base_invisible_run_norm | 0.6668 | - | 0.8590 | 1.4093 | 2.0847 |
| frame_base_acc_norm | 0.6618 | - | 0.8476 | 0.0162 | 0.0173 |

## `y_gt_visible` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| best_ref_candidate_cosine | 0.7309 | + | 0.9239 | 0.7786 | 0.6639 |
| best_ref_candidate_l2 | 0.7309 | - | 0.9239 | 0.6251 | 0.8006 |
| last_candidate_cosine | 0.7101 | + | 0.9168 | 0.7399 | 0.6245 |
| last_candidate_l2 | 0.7101 | - | 0.9168 | 0.6778 | 0.8474 |
| last_visible_age_norm | 0.6658 | - | 0.8987 | 0.1839 | 0.2645 |
| worst_ref_candidate_cosine | 0.6628 | + | 0.8983 | 0.6614 | 0.5723 |
| worst_ref_candidate_l2 | 0.6628 | - | 0.8983 | 0.7993 | 0.9127 |
| query_candidate_cosine | 0.6620 | + | 0.9037 | 0.7001 | 0.6117 |
| query_candidate_l2 | 0.6620 | - | 0.9037 | 0.7466 | 0.8659 |
| query_candidate_age_norm | 0.5705 | + | 0.8792 | 0.4506 | 0.3923 |
| query_last_l2 | 0.5370 | + | 0.8608 | 0.6879 | 0.6557 |
| query_last_cosine | 0.5370 | - | 0.8608 | 0.7355 | 0.7534 |
| has_last_visible_ref | 0.5000 | + | 0.7902 | 1.0000 | 1.0000 |

## `y_gt_visible` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8021 | + | 0.9500 | 0.4362 | 0.2316 |
| frame_override_y_norm | 0.7876 | + | 0.9197 | 0.2397 | 0.1280 |
| frame_base_border_dist_norm | 0.7865 | + | 0.9460 | 0.4442 | 0.2468 |
| frame_base_y_norm | 0.7771 | + | 0.9227 | 0.2414 | 0.1337 |
| frame_override_speed_norm | 0.7733 | - | 0.9331 | 0.0278 | 0.0484 |
| frame_base_speed_norm | 0.7564 | - | 0.9258 | 0.0266 | 0.0455 |
| trigger_t_norm | 0.7145 | + | 0.9129 | 0.7025 | 0.5632 |
| frame_base_acc_norm | 0.7127 | - | 0.9105 | 0.0165 | 0.0162 |
| frame_override_acc_norm | 0.6893 | - | 0.9138 | 0.0171 | 0.0149 |
| frame_speed_diff_norm | 0.6751 | - | 0.8936 | 0.0069 | 0.0129 |

## `y_safe8` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_candidate_cosine | 0.7695 | + | 0.8440 | 0.7756 | 0.6311 |
| last_candidate_l2 | 0.7695 | - | 0.8440 | 0.6220 | 0.8431 |
| best_ref_candidate_l2 | 0.7436 | - | 0.8335 | 0.5809 | 0.7745 |
| best_ref_candidate_cosine | 0.7436 | + | 0.8335 | 0.8038 | 0.6868 |
| worst_ref_candidate_cosine | 0.6880 | + | 0.8021 | 0.6841 | 0.5851 |
| worst_ref_candidate_l2 | 0.6880 | - | 0.8021 | 0.7666 | 0.9024 |
| query_candidate_cosine | 0.6415 | + | 0.7759 | 0.7123 | 0.6408 |
| query_candidate_l2 | 0.6415 | - | 0.7759 | 0.7256 | 0.8339 |
| last_visible_age_norm | 0.6353 | - | 0.7318 | 0.1699 | 0.2426 |
| query_last_cosine | 0.5662 | + | 0.6377 | 0.7528 | 0.7161 |
| query_last_l2 | 0.5661 | - | 0.6377 | 0.6644 | 0.7107 |
| query_candidate_age_norm | 0.5476 | - | 0.6623 | 0.4269 | 0.4618 |
| has_last_visible_ref | 0.5000 | + | 0.6846 | 1.0000 | 1.0000 |

## `y_safe8` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| event_gate_prob | 0.8164 | + | 0.8584 | 0.4393 | 0.1555 |
| segment_prob_max | 0.7621 | + | 0.7820 | 0.7072 | 0.5052 |
| segment_prob_mean | 0.7507 | + | 0.7764 | 0.6083 | 0.4336 |
| v1_prob | 0.7486 | + | 0.7873 | 0.6088 | 0.4245 |
| segment_prob_min | 0.7457 | + | 0.7912 | 0.5151 | 0.3364 |
| frame_base_override_dist_norm | 0.7362 | - | 0.7587 | 0.0475 | 0.1514 |
| frame_base_acc_norm | 0.6456 | - | 0.7014 | 0.0134 | 0.0213 |
| frame_base_invisible_run_norm | 0.6376 | - | 0.7323 | 1.3307 | 1.9240 |
| frame_override_border_dist_norm | 0.6358 | + | 0.7463 | 0.4347 | 0.3462 |
| frame_override_visible_run_norm | 0.6225 | + | 0.6986 | 2.7467 | 1.9910 |

## `y_utility` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_candidate_cosine | 0.8311 | + | 0.6688 | 0.8375 | 0.6623 |
| last_candidate_l2 | 0.8311 | - | 0.6688 | 0.5195 | 0.7992 |
| best_ref_candidate_l2 | 0.7840 | - | 0.6426 | 0.4999 | 0.7319 |
| best_ref_candidate_cosine | 0.7840 | + | 0.6409 | 0.8476 | 0.7152 |
| worst_ref_candidate_cosine | 0.7722 | + | 0.7168 | 0.7442 | 0.5978 |
| worst_ref_candidate_l2 | 0.7722 | - | 0.7168 | 0.6791 | 0.8875 |
| query_candidate_cosine | 0.7026 | + | 0.6523 | 0.7543 | 0.6507 |
| query_candidate_l2 | 0.7026 | - | 0.6523 | 0.6595 | 0.8202 |
| query_last_cosine | 0.7008 | + | 0.4958 | 0.8105 | 0.7035 |
| query_last_l2 | 0.7008 | - | 0.4957 | 0.5695 | 0.7373 |
| last_visible_age_norm | 0.6256 | - | 0.4633 | 0.1622 | 0.2155 |
| query_candidate_age_norm | 0.5904 | - | 0.4399 | 0.3942 | 0.4630 |
| has_last_visible_ref | 0.5000 | + | 0.3703 | 1.0000 | 1.0000 |

## `y_utility` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| event_gate_prob | 0.8201 | + | 0.6656 | 0.5466 | 0.2236 |
| segment_prob_max | 0.7829 | + | 0.5637 | 0.7667 | 0.5619 |
| segment_prob_mean | 0.7638 | + | 0.5532 | 0.6622 | 0.4815 |
| v1_prob | 0.7612 | + | 0.5615 | 0.6645 | 0.4756 |
| frame_base_override_dist_norm | 0.7502 | - | 0.5268 | 0.0335 | 0.1142 |
| segment_prob_min | 0.7436 | + | 0.5668 | 0.5679 | 0.3865 |
| frame_base_acc_norm | 0.6923 | - | 0.4715 | 0.0087 | 0.0202 |
| frame_override_border_dist_norm | 0.6661 | + | 0.5358 | 0.4659 | 0.3686 |
| trigger_t_norm | 0.6434 | - | 0.4391 | 0.6173 | 0.7079 |
| frame_base_speed_norm | 0.6397 | - | 0.4604 | 0.0205 | 0.0346 |

## `y_safe4` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| last_candidate_cosine | 0.8311 | + | 0.6688 | 0.8375 | 0.6623 |
| last_candidate_l2 | 0.8311 | - | 0.6688 | 0.5195 | 0.7992 |
| best_ref_candidate_l2 | 0.7840 | - | 0.6426 | 0.4999 | 0.7319 |
| best_ref_candidate_cosine | 0.7840 | + | 0.6409 | 0.8476 | 0.7152 |
| worst_ref_candidate_cosine | 0.7722 | + | 0.7168 | 0.7442 | 0.5978 |
| worst_ref_candidate_l2 | 0.7722 | - | 0.7168 | 0.6791 | 0.8875 |
| query_candidate_cosine | 0.7026 | + | 0.6523 | 0.7543 | 0.6507 |
| query_candidate_l2 | 0.7026 | - | 0.6523 | 0.6595 | 0.8202 |
| query_last_cosine | 0.7008 | + | 0.4958 | 0.8105 | 0.7035 |
| query_last_l2 | 0.7008 | - | 0.4957 | 0.5695 | 0.7373 |
| last_visible_age_norm | 0.6256 | - | 0.4633 | 0.1622 | 0.2155 |
| query_candidate_age_norm | 0.5904 | - | 0.4399 | 0.3942 | 0.4630 |
| has_last_visible_ref | 0.5000 | + | 0.3703 | 1.0000 | 1.0000 |

## `y_safe4` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| event_gate_prob | 0.8201 | + | 0.6656 | 0.5466 | 0.2236 |
| segment_prob_max | 0.7829 | + | 0.5637 | 0.7667 | 0.5619 |
| segment_prob_mean | 0.7638 | + | 0.5532 | 0.6622 | 0.4815 |
| v1_prob | 0.7612 | + | 0.5615 | 0.6645 | 0.4756 |
| frame_base_override_dist_norm | 0.7502 | - | 0.5268 | 0.0335 | 0.1142 |
| segment_prob_min | 0.7436 | + | 0.5668 | 0.5679 | 0.3865 |
| frame_base_acc_norm | 0.6923 | - | 0.4715 | 0.0087 | 0.0202 |
| frame_override_border_dist_norm | 0.6661 | + | 0.5358 | 0.4659 | 0.3686 |
| trigger_t_norm | 0.6434 | - | 0.4391 | 0.6173 | 0.7079 |
| frame_base_speed_norm | 0.6397 | - | 0.4604 | 0.0205 | 0.0346 |
