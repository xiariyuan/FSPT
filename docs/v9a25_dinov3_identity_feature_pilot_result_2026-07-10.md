# V9-A2.5 DINOv3 Identity Feature Pilot Result

## Protocol

```text
Primary: v9a25_all_plus_dino_logreg_event_max_oof_f1
Preserve W8 common; control W16 extension; event_max; video-heldout OOF-F1; no dense threshold sweep.
```

## Classifier OOF

| Feature set / model | Features | AP | AUC | Brier | OOF-F1 threshold |
|---|---:|---:|---:|---:|---:|
| dino_only / logreg | 57 | 0.3637 | 0.6429 | 0.1990 | 0.325552 |
| dino_only / extratrees | 57 | 0.2248 | 0.5494 | 0.1873 | 0.148826 |
| base_plus_dino / logreg | 85 | 0.2641 | 0.5211 | 0.2405 | 0.000002 |
| base_plus_dino / extratrees | 85 | 0.2397 | 0.5537 | 0.1802 | 0.278317 |
| all_plus_dino / logreg | 200 | 0.2562 | 0.5495 | 0.2412 | 0.007943 |
| all_plus_dino / extratrees | 200 | 0.2385 | 0.5610 | 0.1762 | 0.193109 |

## Top single DINO features

| Feature | Best AUC | Direction |
|---|---:|---|
| candidate_native_cos | 0.6984 | low |
| candidate_native_l2 | 0.6984 | high |
| ql_mem_cand_cos | 0.6537 | low |
| ql_mem_cand_l2 | 0.6537 | high |
| ql_mem_native_cos | 0.6502 | low |
| ql_mem_native_l2 | 0.6502 | high |
| all_mem_local_neg_cos_mean | 0.6431 | low |
| last_local_neg_cos_mean | 0.6415 | low |
| anchor_native_cos_max | 0.6375 | low |
| last_native_cos | 0.6357 | low |
| last_native_l2 | 0.6357 | high |
| anchor_cand_cos_max | 0.6345 | low |

## Baselines

| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w8_preserve_common_only | +0.0964 | +0.9462 | +0.3971 | +0.5085 | +0.0153 | +0.0274 | 0/0/0/0/0 | 16/4/5 |
| w16_accept_all_joint | +0.0810 | +0.9701 | +0.4339 | +0.5593 | +0.0162 | +0.0286 | 557/118/168/67/248 | 15/5/5 |
| frozen_v9a2_fixed0.05 | +0.1025 | +0.9711 | +0.4416 | +0.5848 | +0.0175 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| frozen_v9a2_oof_f1 | +0.0975 | +0.9746 | +0.4391 | +0.5704 | +0.0169 | +0.0293 | 515/116/153/62/216 | 17/3/5 |
| oracle_ext_candidate_good | +0.1955 | +1.0204 | +0.5073 | +0.6152 | +0.0191 | +0.0321 | 118/118/0/0/4 | 17/3/5 |

## Semantic identity variants

| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v9a25_dino_only_logreg_event_max_oof_f1 | +0.1068 | +0.9967 | +0.4402 | +0.5697 | +0.0170 | +0.0287 | 408/106/121/42/169 | 16/4/5 |
| v9a25_dino_only_logreg_event_max_fixed0.05 | +0.0893 | +0.9853 | +0.4339 | +0.5593 | +0.0163 | +0.0286 | 546/118/159/58/246 | 15/5/5 |
| v9a25_dino_only_extratrees_event_max_oof_f1 | +0.0810 | +0.9701 | +0.4339 | +0.5593 | +0.0162 | +0.0286 | 553/118/166/65/248 | 15/5/5 |
| v9a25_base_plus_dino_logreg_event_max_oof_f1 | +0.0846 | +0.9788 | +0.4339 | +0.5593 | +0.0162 | +0.0287 | 552/118/163/62/248 | 15/5/5 |
| v9a25_base_plus_dino_logreg_event_max_fixed0.05 | +0.0744 | +0.9631 | +0.4193 | +0.5434 | +0.0147 | +0.0271 | 525/103/159/61/237 | 15/5/5 |
| v9a25_base_plus_dino_extratrees_event_max_oof_f1 | +0.0736 | +0.9788 | +0.4244 | +0.5515 | +0.0151 | +0.0276 | 431/98/142/53/193 | 15/5/5 |
| v9a25_all_plus_dino_logreg_event_max_oof_f1 | +0.0996 | +0.9781 | +0.4395 | +0.5721 | +0.0173 | +0.0294 | 508/117/151/58/212 | 17/3/5 |
| v9a25_all_plus_dino_logreg_event_max_fixed0.05 | +0.0937 | +0.9798 | +0.4319 | +0.5616 | +0.0158 | +0.0282 | 463/106/138/52/196 | 17/3/5 |
| v9a25_all_plus_dino_extratrees_event_max_oof_f1 | +0.0830 | +0.9701 | +0.4356 | +0.5689 | +0.0160 | +0.0286 | 490/112/161/67/219 | 15/5/5 |

## Paired video-level AJ_RD_256

| Comparison | N | Mean | Median | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |
|---|---:|---:|---:|---|---|---:|
| primary_vs_frozen_v9a2_fixed0.05 | 25 | +0.000035 | +0.000000 | [-0.000227, +0.000277] | 3/1/21 | 0.8750 |
| primary_vs_w16 | 25 | +0.000700 | +0.000000 | [+0.000030, +0.001696] | 5/0/20 | 0.0625 |

## Decision

Semantic identity improves the aggregate frozen V9-A2 policy, but paired evidence is not yet conclusive.