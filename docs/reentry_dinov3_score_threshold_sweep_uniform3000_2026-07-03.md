# V2.4-DINOScore Threshold Sweep

NPZ: `outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev7_9_dinov3_uniform3000_crop33.npz`
Samples: `3000`

## Dataset label rates

| label | positive rate |
|---|---:|
| y_safe16 | 0.7677 |
| y_gt_visible | 0.8177 |
| y_safe8 | 0.6087 |
| y_utility | 0.3283 |
| y_safe4 | 0.3283 |

## Regions

| region | n | rate |
|---|---:|---:|
| all | 3000 | 1.0000 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | 2651 | 0.8837 |
| low_v1_prob_010_030 | 452 | 0.1507 |
| low_segment_max_lt_050 | 894 | 0.2980 |
| union_ambiguous | 2966 | 0.9887 |
| distance_ambiguous_025_050 | 66 | 0.0220 |

## Top diagnostic thresholds, utility-negative precision objective

| feature | region | direction | threshold | n_drop | drop total | util neg precision | util neg recall | safe16 pos loss | safe16 drop pos rate | score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_candidate_cosine | all | low_is_bad | 0.39499 | 30 | 0.0100 | 1.0000 | 0.0149 | 0.0004 | 0.0333 | 1.0071 |
| last_candidate_l2 | all | high_is_bad | 1.10001 | 30 | 0.0100 | 1.0000 | 0.0149 | 0.0004 | 0.0333 | 1.0071 |
| last_candidate_cosine | union_ambiguous | low_is_bad | 0.39780 | 30 | 0.0100 | 1.0000 | 0.0149 | 0.0009 | 0.0667 | 1.0068 |
| last_candidate_l2 | union_ambiguous | high_is_bad | 1.09745 | 30 | 0.0100 | 1.0000 | 0.0149 | 0.0009 | 0.0667 | 1.0068 |
| last_candidate_cosine | numeric_thinks_safe_event_ge_002_dist_lt_025 | low_is_bad | 0.43694 | 27 | 0.0090 | 1.0000 | 0.0134 | 0.0017 | 0.1481 | 1.0054 |
| last_candidate_l2 | numeric_thinks_safe_event_ge_002_dist_lt_025 | high_is_bad | 1.06119 | 27 | 0.0090 | 1.0000 | 0.0134 | 0.0017 | 0.1481 | 1.0054 |
| best_ref_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.69279 | 181 | 0.0603 | 0.9724 | 0.0873 | 0.0386 | 0.4917 | 0.9871 |
| best_ref_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.78385 | 181 | 0.0603 | 0.9724 | 0.0873 | 0.0386 | 0.4917 | 0.9871 |
| last_candidate_cosine | numeric_thinks_safe_event_ge_002_dist_lt_025 | low_is_bad | 0.45273 | 53 | 0.0177 | 0.9811 | 0.0258 | 0.0096 | 0.4151 | 0.9869 |
| last_candidate_l2 | numeric_thinks_safe_event_ge_002_dist_lt_025 | high_is_bad | 1.04620 | 53 | 0.0177 | 0.9811 | 0.0258 | 0.0096 | 0.4151 | 0.9869 |
| best_ref_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.65206 | 113 | 0.0377 | 0.9735 | 0.0546 | 0.0187 | 0.3805 | 0.9867 |
| best_ref_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.83420 | 113 | 0.0377 | 0.9735 | 0.0546 | 0.0187 | 0.3805 | 0.9867 |
| best_ref_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.66566 | 136 | 0.0453 | 0.9706 | 0.0655 | 0.0248 | 0.4191 | 0.9848 |
| best_ref_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.81773 | 136 | 0.0453 | 0.9706 | 0.0655 | 0.0248 | 0.4191 | 0.9848 |
| best_ref_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.62733 | 91 | 0.0303 | 0.9670 | 0.0437 | 0.0143 | 0.3626 | 0.9781 |
| best_ref_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.86333 | 91 | 0.0303 | 0.9670 | 0.0437 | 0.0143 | 0.3626 | 0.9781 |
| last_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.61022 | 181 | 0.0603 | 0.9669 | 0.0868 | 0.0439 | 0.5580 | 0.9774 |
| last_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.88293 | 181 | 0.0603 | 0.9669 | 0.0868 | 0.0439 | 0.5580 | 0.9774 |
| best_ref_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.55005 | 34 | 0.0113 | 0.9706 | 0.0164 | 0.0022 | 0.1471 | 0.9771 |
| best_ref_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.94863 | 34 | 0.0113 | 0.9706 | 0.0164 | 0.0022 | 0.1471 | 0.9771 |
| last_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.54463 | 91 | 0.0303 | 0.9670 | 0.0437 | 0.0169 | 0.4286 | 0.9762 |
| last_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.95433 | 91 | 0.0303 | 0.9670 | 0.0437 | 0.0169 | 0.4286 | 0.9762 |
| last_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.56583 | 113 | 0.0377 | 0.9646 | 0.0541 | 0.0230 | 0.4690 | 0.9744 |
| last_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.93185 | 113 | 0.0377 | 0.9646 | 0.0541 | 0.0230 | 0.4690 | 0.9744 |
| last_candidate_cosine | low_segment_max_lt_050 | low_is_bad | 0.59949 | 358 | 0.1193 | 0.9581 | 0.1702 | 0.0990 | 0.6369 | 0.9690 |
| last_candidate_l2 | low_segment_max_lt_050 | high_is_bad | 0.89499 | 358 | 0.1193 | 0.9581 | 0.1702 | 0.0990 | 0.6369 | 0.9690 |
| last_candidate_cosine | low_segment_max_lt_050 | low_is_bad | 0.57210 | 268 | 0.0893 | 0.9552 | 0.1270 | 0.0699 | 0.6007 | 0.9663 |
| last_candidate_l2 | low_segment_max_lt_050 | high_is_bad | 0.92509 | 268 | 0.0893 | 0.9552 | 0.1270 | 0.0699 | 0.6007 | 0.9663 |
| last_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.58336 | 136 | 0.0453 | 0.9559 | 0.0645 | 0.0304 | 0.5147 | 0.9653 |
| last_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.91284 | 136 | 0.0453 | 0.9559 | 0.0645 | 0.0304 | 0.5147 | 0.9653 |
| best_ref_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.60548 | 68 | 0.0227 | 0.9559 | 0.0323 | 0.0096 | 0.3235 | 0.9648 |
| best_ref_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.88828 | 68 | 0.0227 | 0.9559 | 0.0323 | 0.0096 | 0.3235 | 0.9648 |
| best_ref_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.56837 | 46 | 0.0153 | 0.9565 | 0.0218 | 0.0043 | 0.2174 | 0.9642 |
| best_ref_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.92912 | 46 | 0.0153 | 0.9565 | 0.0218 | 0.0043 | 0.2174 | 0.9642 |
| last_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.53457 | 68 | 0.0227 | 0.9559 | 0.0323 | 0.0126 | 0.4265 | 0.9626 |
| last_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.96482 | 68 | 0.0227 | 0.9559 | 0.0323 | 0.0126 | 0.4265 | 0.9626 |
| last_candidate_cosine | low_v1_prob_010_030 | low_is_bad | 0.51711 | 46 | 0.0153 | 0.9565 | 0.0218 | 0.0083 | 0.4130 | 0.9613 |
| last_candidate_l2 | low_v1_prob_010_030 | high_is_bad | 0.98274 | 46 | 0.0153 | 0.9565 | 0.0218 | 0.0083 | 0.4130 | 0.9613 |
| last_candidate_cosine | low_segment_max_lt_050 | low_is_bad | 0.55482 | 224 | 0.0747 | 0.9509 | 0.1057 | 0.0569 | 0.5848 | 0.9611 |
| last_candidate_l2 | low_segment_max_lt_050 | high_is_bad | 0.94359 | 224 | 0.0747 | 0.9509 | 0.1057 | 0.0569 | 0.5848 | 0.9611 |

## Interpretation note

This is label-level threshold diagnostics only. It does not yet evaluate AJ_RD/AJ/OA. Promising rows should be converted into a full evaluator only if they drop enough frames with high utility-negative precision and low safe16-positive loss.