# V9-A2.5 DINOv3 Late-Fusion Audit

predeclared late fusion weights [0.25, 0.50, 0.75]; video-heldout OOF scores; event_max; OOF-F1; no trajectory threshold sweep

Row-score Pearson correlation: 0.0029
Row-score Spearman correlation: 0.02754898067993743

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| frozen_v9a2_fixed0.05 | +0.1025 | +0.9711 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| dino_only_oof_f1 | +0.1068 | +0.9967 | +0.0287 | 408/106/121/42/169 | 16/4/5 |
| late_fusion_frozen0.25_dino0.75_event_max_oof_f1 | +0.1136 | +1.0013 | +0.0289 | 320/88/95/32/141 | 16/4/5 |
| late_fusion_frozen0.50_dino0.50_event_max_oof_f1 | +0.0920 | +0.9788 | +0.0288 | 526/118/157/62/224 | 16/4/5 |
| late_fusion_frozen0.75_dino0.25_event_max_oof_f1 | +0.0863 | +0.9735 | +0.0287 | 543/118/163/65/236 | 15/5/5 |

## Paired against frozen V9-A2 fixed0.05

| Fusion | Mean | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |
|---|---:|---|---|---:|
| late_fusion_frozen0.25_dino0.75_event_max_oof_f1 | -0.000223 | [-0.000641, +0.000122] | 2/5/18 | 0.296875 |
| late_fusion_frozen0.50_dino0.50_event_max_oof_f1 | -0.000520 | [-0.001543, +0.000176] | 2/3/20 | 0.375 |
| late_fusion_frozen0.75_dino0.25_event_max_oof_f1 | -0.000590 | [-0.001615, +0.000111] | 2/4/19 | 0.25 |