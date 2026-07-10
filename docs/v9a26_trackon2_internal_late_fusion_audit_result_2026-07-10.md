# V9-A2.6 TrackOn2 Internal Proxy Late-Fusion Audit

predeclared weights [0.25,0.50,0.75]; video-heldout OOF; event_max; OOF-F1; no trajectory threshold sweep

Pearson correlation: 0.1272
Spearman correlation: 0.10505432634932757

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| frozen_v9a2_fixed0.05 | +0.1025 | +0.9711 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| internal_only_oof_f1 | +0.0664 | +0.9711 | +0.0271 | 394/91/149/56/169 | 16/4/5 |
| late_fusion_frozen0.25_internal0.75_event_max_oof_f1 | +0.0707 | +0.9729 | +0.0273 | 488/103/161/65/214 | 16/4/5 |
| late_fusion_frozen0.50_internal0.50_event_max_oof_f1 | +0.0810 | +0.9701 | +0.0286 | 553/118/168/67/244 | 15/5/5 |
| late_fusion_frozen0.75_internal0.25_event_max_oof_f1 | +0.0810 | +0.9701 | +0.0286 | 553/118/168/67/244 | 15/5/5 |

## Paired against frozen V9-A2 fixed0.05

| Fusion | Mean | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |
|---|---:|---|---|---:|
| late_fusion_frozen0.25_internal0.75_event_max_oof_f1 | -0.001591 | [-0.004440, +0.000081] | 3/6/16 | 0.1484375 |
| late_fusion_frozen0.50_internal0.50_event_max_oof_f1 | -0.000665 | [-0.001681, +0.000051] | 2/4/19 | 0.15625 |
| late_fusion_frozen0.75_internal0.25_event_max_oof_f1 | -0.000665 | [-0.001681, +0.000051] | 2/4/19 | 0.15625 |