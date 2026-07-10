# V9-A2.4 Paper-Ready Ablation Result

## Frozen protocol

```text
all_logreg + event_max + fixed threshold 0.05; preserve W8 common; control W16 extension only
```

No dense trajectory threshold search was used in V9-A2.4.

## Unified absolute metrics

| Variant | AJ | OA | delta_avg | delta_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| native | 65.2366 | 90.8186 | 77.9458 | 85.8670 | 0.3534 | 0.5333 |
| w8_preserve_common_only | 65.3330 | 91.7648 | 78.3429 | 86.3755 | 0.3687 | 0.5607 |
| w16_accept_all_joint | 65.3176 | 91.7886 | 78.3797 | 86.4263 | 0.3696 | 0.5619 |
| v9a1_extratrees_oof_f1 | 65.4072 | 91.8437 | 78.2952 | 86.3531 | 0.3637 | 0.5545 |
| v9a2_all_logreg_event_max_fixed_0.05 | 65.3391 | 91.7897 | 78.3874 | 86.4518 | 0.3709 | 0.5627 |
| v9a2_all_logreg_event_max_oof_f1 | 65.3341 | 91.7932 | 78.3850 | 86.4374 | 0.3703 | 0.5626 |
| oracle_ext_candidate_good | 65.4321 | 91.8390 | 78.4531 | 86.4822 | 0.3725 | 0.5654 |

## Unified deltas and stability

| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| native | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 | 0/0/25 |
| w8_preserve_common_only | +0.0964 | +0.9462 | +0.3971 | +0.5085 | +0.0153 | +0.0274 | 1456 | 16/4/5 |
| w16_accept_all_joint | +0.0810 | +0.9701 | +0.4339 | +0.5593 | +0.0162 | +0.0286 | 2013 | 15/5/5 |
| v9a1_extratrees_oof_f1 | +0.1707 | +1.0252 | +0.3494 | +0.4861 | +0.0103 | +0.0212 | 804 | 10/2/13 |
| v9a2_all_logreg_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.4416 | +0.5848 | +0.0175 | +0.0294 | 1937 | 17/3/5 |
| v9a2_all_logreg_event_max_oof_f1 | +0.0975 | +0.9746 | +0.4391 | +0.5704 | +0.0169 | +0.0293 | 1971 | 17/3/5 |
| oracle_ext_candidate_good | +0.1955 | +1.0204 | +0.5073 | +0.6152 | +0.0191 | +0.0321 | 1574 | 17/3/5 |

## Feature-set ablation

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| feature_base_logreg_event_max_fixed_0.05 | +0.0695 | +0.9735 | +0.0273 | 547/110/166/65/248 | 15/5/5 |
| feature_anchor_logreg_event_max_fixed_0.05 | +0.0834 | +0.9724 | +0.0275 | 470/99/143/59/195 | 17/3/5 |
| feature_all_logreg_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.0294 | 481/109/141/61/190 | 17/3/5 |

## Controller-mode ablation

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| mode_all_logreg_frame_fixed_0.05 | +0.0907 | +0.9711 | +0.0278 | 445/90/131/55/185 | 17/3/5 |
| mode_all_logreg_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| mode_all_logreg_event_mean_fixed_0.05 | +0.0883 | +0.9798 | +0.0277 | 453/93/134/54/189 | 17/3/5 |

## Threshold-protocol ablation

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| w16_accept_all_joint | +0.0810 | +0.9701 | +0.0286 | 557/118/168/67/248 | 15/5/5 |
| v9a2_all_logreg_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| v9a2_all_logreg_event_max_oof_f1 | +0.0975 | +0.9746 | +0.0293 | 515/116/153/62/216 | 17/3/5 |

## Anchor-group ablation

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| anchor_group_base_generic_event_max_fixed_0.05 | +0.0757 | +0.9676 | +0.0274 | 535/103/162/61/247 | 16/4/5 |
| anchor_group_base_generic_event_max_oof_f1 | +0.0891 | +0.9763 | +0.0291 | 554/118/165/64/248 | 16/4/5 |
| anchor_group_base_query_anchor_event_max_fixed_0.05 | +0.0768 | +0.9746 | +0.0278 | 545/110/165/64/247 | 16/4/5 |
| anchor_group_base_query_anchor_event_max_oof_f1 | +0.0876 | +0.9729 | +0.0291 | 556/118/167/66/248 | 16/4/5 |
| anchor_group_base_last_anchor_event_max_fixed_0.05 | +0.0672 | +0.9555 | +0.0272 | 528/96/162/61/247 | 16/4/5 |
| anchor_group_base_last_anchor_event_max_oof_f1 | +0.0824 | +0.9735 | +0.0286 | 555/118/166/65/248 | 15/5/5 |
| anchor_group_base_preocc_anchor_event_max_fixed_0.05 | +0.0945 | +0.9798 | +0.0280 | 484/108/140/57/198 | 17/3/5 |
| anchor_group_base_preocc_anchor_event_max_oof_f1 | +0.0891 | +0.9763 | +0.0291 | 552/118/163/62/248 | 16/4/5 |
| anchor_group_base_all_anchors_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| anchor_group_base_all_anchors_event_max_oof_f1 | +0.0975 | +0.9746 | +0.0293 | 515/116/153/62/216 | 17/3/5 |

## Paired video-level uncertainty (AJ_RD_256)

| Comparison | N | Mean | Median | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p | Sign-test p |
|---|---:|---:|---:|---|---|---:|---:|
| fixed0.05_vs_w16 | 25 | +0.000665 | +0.000000 | [-0.000051, +0.001681] | 4/2/19 | 0.1562 | 0.6875 |
| fixed0.05_vs_w8 | 25 | -0.000317 | +0.000000 | [-0.004895, +0.003646] | 9/4/12 | 0.9231 | 0.2668 |

## CPU reference efficiency

| Quantity | Value |
|---|---:|
| Extension rows / events | 557 / 141 |
| Accepted extension rows | 481 (86.36%) |
| Activated events | 113 (80.14%) |
| Feature extraction total | 2.319 s for 557 rows |
| Feature extraction p50 / p95 | 4.064 / 4.312 ms per row |
| Controller batch p50 / p95 | 0.520 / 2.267 ms for 557 rows |
| Event aggregation p50 / p95 | 0.744 / 0.754 ms |

CPU reference Python/Numpy/Scikit-learn implementation; cached video/candidate tensors; not optimized online kernel latency.

## Decision

The aggregate fixed0.05 policy remains better than W16, but the paired video-level 95% CI crosses zero.
Therefore V9-A2 is retained as a positive, paper-ready diagnostic/prototype rather than a statistically established final method.
Next: V9-A2.5 semantic/internal identity features, using the frozen fixed0.05 policy and oracle gap as baselines.