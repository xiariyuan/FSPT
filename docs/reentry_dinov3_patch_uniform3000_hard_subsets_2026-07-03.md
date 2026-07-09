# DINO/ViT Hard-Subset Analysis

Samples: `3000`

## all

n = `3000`

### y_safe16

positive_rate = `0.7677`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| best_ref_candidate_l2 | 0.7188 | - |
| best_ref_candidate_cosine | 0.7188 | + |
| last_candidate_cosine | 0.6998 | + |
| last_candidate_l2 | 0.6998 | - |
| query_candidate_cosine | 0.6647 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7875 | + |
| frame_base_border_dist_norm | 0.7432 | + |
| frame_override_y_norm | 0.7199 | + |
| event_gate_prob | 0.7021 | + |
| frame_base_speed_norm | 0.6976 | - |

### y_gt_visible

positive_rate = `0.8177`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| best_ref_candidate_cosine | 0.7142 | + |
| best_ref_candidate_l2 | 0.7142 | - |
| last_candidate_cosine | 0.6922 | + |
| last_candidate_l2 | 0.6922 | - |
| query_candidate_cosine | 0.6640 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7920 | + |
| frame_base_border_dist_norm | 0.7761 | + |
| frame_override_y_norm | 0.7716 | + |
| frame_override_speed_norm | 0.7659 | - |
| frame_base_y_norm | 0.7624 | + |

### y_safe8

positive_rate = `0.6087`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.7545 | + |
| last_candidate_l2 | 0.7545 | - |
| best_ref_candidate_cosine | 0.7358 | + |
| best_ref_candidate_l2 | 0.7358 | - |
| worst_ref_candidate_l2 | 0.6871 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.8071 | + |
| segment_prob_max | 0.7583 | + |
| segment_prob_mean | 0.7458 | + |
| segment_prob_min | 0.7433 | + |
| v1_prob | 0.7379 | + |

### y_utility

positive_rate = `0.3283`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.8266 | + |
| last_candidate_l2 | 0.8266 | - |
| best_ref_candidate_l2 | 0.7816 | - |
| best_ref_candidate_cosine | 0.7816 | + |
| worst_ref_candidate_cosine | 0.7770 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.8178 | + |
| segment_prob_max | 0.7817 | + |
| segment_prob_mean | 0.7615 | + |
| v1_prob | 0.7579 | + |
| frame_base_override_dist_norm | 0.7515 | - |

### y_safe4

positive_rate = `0.3283`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.8266 | + |
| last_candidate_l2 | 0.8266 | - |
| best_ref_candidate_l2 | 0.7816 | - |
| best_ref_candidate_cosine | 0.7816 | + |
| worst_ref_candidate_cosine | 0.7770 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.8178 | + |
| segment_prob_max | 0.7817 | + |
| segment_prob_mean | 0.7615 | + |
| v1_prob | 0.7579 | + |
| frame_base_override_dist_norm | 0.7515 | - |

## numeric_thinks_safe_event_ge_002_dist_lt_025

n = `2651`

### y_safe16

positive_rate = `0.8204`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| best_ref_candidate_cosine | 0.7103 | + |
| best_ref_candidate_l2 | 0.7103 | - |
| last_candidate_cosine | 0.6899 | + |
| last_candidate_l2 | 0.6899 | - |
| query_candidate_cosine | 0.6688 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7710 | + |
| frame_base_border_dist_norm | 0.7538 | + |
| frame_override_speed_norm | 0.7186 | - |
| frame_base_y_norm | 0.7150 | + |
| frame_override_y_norm | 0.7143 | + |

### y_gt_visible

positive_rate = `0.8472`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| best_ref_candidate_cosine | 0.7068 | + |
| best_ref_candidate_l2 | 0.7068 | - |
| last_candidate_cosine | 0.6775 | + |
| last_candidate_l2 | 0.6775 | - |
| query_candidate_l2 | 0.6751 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7799 | + |
| frame_base_border_dist_norm | 0.7688 | + |
| frame_override_speed_norm | 0.7634 | - |
| frame_override_y_norm | 0.7525 | + |
| frame_base_speed_norm | 0.7468 | - |

### y_safe8

positive_rate = `0.6741`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.7558 | + |
| last_candidate_l2 | 0.7558 | - |
| best_ref_candidate_l2 | 0.7392 | - |
| best_ref_candidate_cosine | 0.7392 | + |
| worst_ref_candidate_l2 | 0.7008 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.7596 | + |
| segment_prob_max | 0.7016 | + |
| segment_prob_min | 0.6914 | + |
| segment_prob_mean | 0.6856 | + |
| v1_prob | 0.6815 | + |

### y_utility

positive_rate = `0.3644`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.8237 | + |
| last_candidate_l2 | 0.8237 | - |
| worst_ref_candidate_cosine | 0.7841 | + |
| worst_ref_candidate_l2 | 0.7841 | - |
| best_ref_candidate_l2 | 0.7777 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.8000 | + |
| segment_prob_max | 0.7564 | + |
| segment_prob_mean | 0.7323 | + |
| v1_prob | 0.7300 | + |
| frame_base_override_dist_norm | 0.7256 | - |

### y_safe4

positive_rate = `0.3644`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.8237 | + |
| last_candidate_l2 | 0.8237 | - |
| worst_ref_candidate_cosine | 0.7841 | + |
| worst_ref_candidate_l2 | 0.7841 | - |
| best_ref_candidate_l2 | 0.7777 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.8000 | + |
| segment_prob_max | 0.7564 | + |
| segment_prob_mean | 0.7323 | + |
| v1_prob | 0.7300 | + |
| frame_base_override_dist_norm | 0.7256 | - |

## distance_ambiguous_025_050

n = `66`

### y_safe16

positive_rate = `0.2424`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_cosine | 0.9125 | - |
| query_last_l2 | 0.9125 | + |
| query_candidate_cosine | 0.8213 | - |
| query_candidate_l2 | 0.8213 | + |
| worst_ref_candidate_cosine | 0.8100 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_base_x_norm | 0.8387 | + |
| frame_base_override_ydiff_norm | 0.8375 | + |
| frame_base_speed_norm | 0.8300 | - |
| frame_base_override_xdiff_norm | 0.8287 | - |
| frame_base_acc_norm | 0.8275 | - |

### y_gt_visible

positive_rate = `0.3636`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_candidate_cosine | 0.8016 | - |
| query_candidate_l2 | 0.8016 | + |
| worst_ref_candidate_cosine | 0.7847 | - |
| worst_ref_candidate_l2 | 0.7847 | + |
| best_ref_candidate_cosine | 0.7758 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_base_override_ydiff_norm | 0.8958 | + |
| frame_base_x_norm | 0.8135 | + |
| frame_override_speed_norm | 0.8026 | - |
| frame_base_speed_norm | 0.7986 | - |
| frame_override_visible_run_norm | 0.7525 | - |

### y_safe8

positive_rate = `0.2273`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_cosine | 0.9516 | - |
| query_last_l2 | 0.9516 | + |
| query_candidate_cosine | 0.8784 | - |
| query_candidate_l2 | 0.8784 | + |
| worst_ref_candidate_cosine | 0.8667 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_base_x_norm | 0.8902 | + |
| frame_base_acc_norm | 0.8444 | - |
| frame_base_speed_norm | 0.8366 | - |
| frame_base_override_ydiff_norm | 0.8144 | + |
| frame_base_override_xdiff_norm | 0.8013 | - |

### y_utility

positive_rate = `0.0455`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_cosine | 0.9286 | - |
| query_last_l2 | 0.9286 | + |
| query_candidate_cosine | 0.8466 | - |
| query_candidate_l2 | 0.8466 | + |
| worst_ref_candidate_cosine | 0.8307 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_visible_run_norm | 0.9074 | - |
| frame_base_override_ydiff_norm | 0.8307 | + |
| frame_base_x_norm | 0.8201 | + |
| frame_base_override_xdiff_norm | 0.8042 | - |
| frame_override_acc_norm | 0.7937 | - |

### y_safe4

positive_rate = `0.0455`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_cosine | 0.9286 | - |
| query_last_l2 | 0.9286 | + |
| query_candidate_cosine | 0.8466 | - |
| query_candidate_l2 | 0.8466 | + |
| worst_ref_candidate_cosine | 0.8307 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_visible_run_norm | 0.9074 | - |
| frame_base_override_ydiff_norm | 0.8307 | + |
| frame_base_x_norm | 0.8201 | + |
| frame_base_override_xdiff_norm | 0.8042 | - |
| frame_override_acc_norm | 0.7937 | - |

## low_v1_prob_010_030

n = `452`

### y_safe16

positive_rate = `0.6350`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_l2 | 0.8317 | + |
| query_last_cosine | 0.8316 | - |
| last_visible_age_norm | 0.7550 | - |
| query_candidate_age_norm | 0.7282 | + |
| best_ref_candidate_l2 | 0.6677 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.8300 | + |
| frame_base_border_dist_norm | 0.8081 | + |
| frame_base_x_norm | 0.7973 | - |
| frame_override_x_norm | 0.7857 | - |
| trigger_t_norm | 0.7794 | + |

### y_gt_visible

positive_rate = `0.7434`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_l2 | 0.7738 | + |
| query_last_cosine | 0.7738 | - |
| last_visible_age_norm | 0.7441 | - |
| query_candidate_age_norm | 0.6770 | + |
| best_ref_candidate_l2 | 0.6107 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_y_norm | 0.8780 | + |
| trigger_t_norm | 0.8596 | + |
| frame_base_y_norm | 0.8594 | + |
| frame_override_border_dist_norm | 0.8346 | + |
| frame_base_border_dist_norm | 0.8271 | + |

### y_safe8

positive_rate = `0.2124`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_visible_age_norm | 0.6072 | - |
| best_ref_candidate_cosine | 0.5861 | + |
| best_ref_candidate_l2 | 0.5861 | - |
| query_candidate_age_norm | 0.5675 | + |
| query_last_cosine | 0.5586 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_base_override_dist_norm | 0.7651 | - |
| event_gate_prob | 0.7568 | + |
| segment_prob_max | 0.7429 | + |
| segment_prob_mean | 0.7108 | + |
| prob_div_segment_max | 0.7062 | - |

### y_utility

positive_rate = `0.0575`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_visible_age_norm | 0.7620 | - |
| last_candidate_cosine | 0.7385 | + |
| last_candidate_l2 | 0.7385 | - |
| best_ref_candidate_cosine | 0.7177 | + |
| best_ref_candidate_l2 | 0.7177 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_speed_norm | 0.7797 | + |
| segment_prob_std | 0.7764 | + |
| frame_base_invisible_run_norm | 0.7606 | - |
| frame_base_y_norm | 0.7601 | - |
| frame_base_speed_norm | 0.7468 | + |

### y_safe4

positive_rate = `0.0575`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_visible_age_norm | 0.7620 | - |
| last_candidate_cosine | 0.7385 | + |
| last_candidate_l2 | 0.7385 | - |
| best_ref_candidate_cosine | 0.7177 | + |
| best_ref_candidate_l2 | 0.7177 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_speed_norm | 0.7797 | + |
| segment_prob_std | 0.7764 | + |
| frame_base_invisible_run_norm | 0.7606 | - |
| frame_base_y_norm | 0.7601 | - |
| frame_base_speed_norm | 0.7468 | + |

## high_event_low_dist_low_v1

n = `239`

### y_safe16

positive_rate = `0.8033`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_cosine | 0.8384 | - |
| query_last_l2 | 0.8384 | + |
| query_candidate_age_norm | 0.7388 | + |
| last_visible_age_norm | 0.6196 | - |
| query_candidate_cosine | 0.5809 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7850 | + |
| frame_override_y_norm | 0.7724 | + |
| frame_base_border_dist_norm | 0.7702 | + |
| frame_base_y_norm | 0.7681 | + |
| trigger_t_norm | 0.7441 | + |

### y_gt_visible

positive_rate = `0.8536`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_cosine | 0.8830 | - |
| query_last_l2 | 0.8830 | + |
| query_candidate_age_norm | 0.7651 | + |
| last_candidate_cosine | 0.6118 | - |
| last_candidate_l2 | 0.6118 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_y_norm | 0.8878 | + |
| frame_override_border_dist_norm | 0.8875 | + |
| frame_base_y_norm | 0.8735 | + |
| frame_base_border_dist_norm | 0.8734 | + |
| trigger_t_norm | 0.8275 | + |

### y_safe8

positive_rate = `0.3305`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| best_ref_candidate_cosine | 0.6210 | + |
| best_ref_candidate_l2 | 0.6210 | - |
| last_visible_age_norm | 0.5886 | - |
| query_candidate_age_norm | 0.5684 | + |
| query_candidate_cosine | 0.5567 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_base_override_dist_norm | 0.8182 | - |
| frame_base_override_xdiff_norm | 0.7765 | + |
| segment_prob_max | 0.7495 | + |
| event_gate_prob | 0.7452 | + |
| prob_div_segment_max | 0.7407 | - |

### y_utility

positive_rate = `0.0795`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.8478 | + |
| last_candidate_l2 | 0.8478 | - |
| last_visible_age_norm | 0.8472 | - |
| best_ref_candidate_cosine | 0.8074 | + |
| best_ref_candidate_l2 | 0.8074 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_speed_norm | 0.8866 | + |
| frame_base_speed_norm | 0.8811 | + |
| segment_prob_max | 0.8502 | + |
| frame_base_invisible_run_norm | 0.8462 | - |
| frame_base_override_dist_norm | 0.8397 | - |

### y_safe4

positive_rate = `0.0795`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.8478 | + |
| last_candidate_l2 | 0.8478 | - |
| last_visible_age_norm | 0.8472 | - |
| best_ref_candidate_cosine | 0.8074 | + |
| best_ref_candidate_l2 | 0.8074 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_speed_norm | 0.8866 | + |
| frame_base_speed_norm | 0.8811 | + |
| segment_prob_max | 0.8502 | + |
| frame_base_invisible_run_norm | 0.8462 | - |
| frame_base_override_dist_norm | 0.8397 | - |

## high_event_low_dist_high_v1

n = `1687`

### y_safe16

positive_rate = `0.8352`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| best_ref_candidate_cosine | 0.7552 | + |
| best_ref_candidate_l2 | 0.7552 | - |
| last_candidate_cosine | 0.7506 | + |
| last_candidate_l2 | 0.7506 | - |
| worst_ref_candidate_cosine | 0.7370 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7794 | + |
| frame_base_border_dist_norm | 0.7711 | + |
| frame_override_speed_norm | 0.7512 | - |
| frame_base_speed_norm | 0.7293 | - |
| frame_override_visible_run_norm | 0.7233 | + |

### y_gt_visible

positive_rate = `0.8488`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| best_ref_candidate_cosine | 0.7665 | + |
| best_ref_candidate_l2 | 0.7665 | - |
| last_candidate_l2 | 0.7603 | - |
| last_candidate_cosine | 0.7603 | + |
| worst_ref_candidate_cosine | 0.7485 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7773 | + |
| frame_base_border_dist_norm | 0.7705 | + |
| frame_override_speed_norm | 0.7640 | - |
| frame_base_speed_norm | 0.7458 | - |
| frame_base_y_norm | 0.7422 | + |

### y_safe8

positive_rate = `0.7730`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.7557 | + |
| last_candidate_l2 | 0.7557 | - |
| best_ref_candidate_cosine | 0.7461 | + |
| best_ref_candidate_l2 | 0.7461 | - |
| worst_ref_candidate_cosine | 0.7402 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_visible_run_norm | 0.7255 | + |
| event_gate_prob | 0.6937 | + |
| frame_override_border_dist_norm | 0.6827 | + |
| frame_base_acc_norm | 0.6629 | - |
| frame_base_speed_norm | 0.6570 | - |

### y_utility

positive_rate = `0.4707`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| worst_ref_candidate_l2 | 0.7945 | - |
| worst_ref_candidate_cosine | 0.7945 | + |
| last_candidate_cosine | 0.7823 | + |
| last_candidate_l2 | 0.7823 | - |
| best_ref_candidate_cosine | 0.7616 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.7599 | + |
| segment_prob_max | 0.7054 | + |
| frame_base_override_dist_norm | 0.7010 | - |
| frame_base_acc_norm | 0.6902 | - |
| frame_base_speed_norm | 0.6832 | - |

### y_safe4

positive_rate = `0.4707`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| worst_ref_candidate_l2 | 0.7945 | - |
| worst_ref_candidate_cosine | 0.7945 | + |
| last_candidate_cosine | 0.7823 | + |
| last_candidate_l2 | 0.7823 | - |
| best_ref_candidate_cosine | 0.7616 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.7599 | + |
| segment_prob_max | 0.7054 | + |
| frame_base_override_dist_norm | 0.7010 | - |
| frame_base_acc_norm | 0.6902 | - |
| frame_base_speed_norm | 0.6832 | - |

## low_segment_max_lt_050

n = `894`

### y_safe16

positive_rate = `0.6521`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_l2 | 0.7743 | + |
| query_last_cosine | 0.7743 | - |
| query_candidate_age_norm | 0.7579 | + |
| last_visible_age_norm | 0.6413 | - |
| best_ref_candidate_cosine | 0.5635 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| frame_override_border_dist_norm | 0.7860 | + |
| frame_time_since_query_norm | 0.7579 | + |
| frame_minus_query_norm | 0.7579 | + |
| trigger_t_norm | 0.7551 | + |
| event_gate_prob | 0.7542 | + |

### y_gt_visible

positive_rate = `0.7864`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_candidate_age_norm | 0.6739 | + |
| last_visible_age_norm | 0.6656 | - |
| query_last_l2 | 0.6630 | + |
| query_last_cosine | 0.6630 | - |
| best_ref_candidate_l2 | 0.5645 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| trigger_t_norm | 0.8594 | + |
| frame_base_speed_norm | 0.8532 | - |
| frame_override_y_norm | 0.8519 | + |
| frame_override_speed_norm | 0.8366 | - |
| frame_override_border_dist_norm | 0.8255 | + |

### y_safe8

positive_rate = `0.3121`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| query_last_cosine | 0.6249 | - |
| query_last_l2 | 0.6249 | + |
| worst_ref_candidate_cosine | 0.5826 | - |
| worst_ref_candidate_l2 | 0.5826 | + |
| query_candidate_cosine | 0.5762 | - |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| event_gate_prob | 0.7682 | + |
| segment_prob_max | 0.7301 | + |
| segment_prob_min | 0.7078 | + |
| segment_prob_mean | 0.6989 | + |
| v1_prob | 0.6826 | + |

### y_utility

positive_rate = `0.0962`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.6886 | + |
| last_candidate_l2 | 0.6886 | - |
| query_candidate_cosine | 0.5965 | - |
| query_candidate_l2 | 0.5965 | + |
| best_ref_candidate_cosine | 0.5465 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| v1_prob | 0.8053 | + |
| segment_prob_mean | 0.7894 | + |
| segment_prob_max | 0.7816 | + |
| segment_prob_min | 0.7349 | + |
| frame_base_x_norm | 0.7166 | + |

### y_safe4

positive_rate = `0.0962`

DINO/ViT top:
| feature | AUC | dir |
|---|---|---|
| last_candidate_cosine | 0.6886 | + |
| last_candidate_l2 | 0.6886 | - |
| query_candidate_cosine | 0.5965 | - |
| query_candidate_l2 | 0.5965 | + |
| best_ref_candidate_cosine | 0.5465 | + |

Numeric top:
| feature | AUC | dir |
|---|---|---|
| v1_prob | 0.8053 | + |
| segment_prob_mean | 0.7894 | + |
| segment_prob_max | 0.7816 | + |
| segment_prob_min | 0.7349 | + |
| frame_base_x_norm | 0.7166 | + |
