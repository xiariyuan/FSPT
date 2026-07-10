# V9-A2.3c-R Robust Threshold and Stability Audit

Protocol: predeclared compact threshold audit; event_max only; no dense trajectory sweep.

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---|
| oracle_ext_candidate_good | +0.1955 | +1.0204 | +0.0321 | 1574 | 118/118/0/0/4 | 17/3/5 |
| oracle_ext_good_not_worse | +0.1966 | +1.0106 | +0.0321 | 1570 | 114/114/0/0/0 | 17/3/5 |
| all_logreg_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.0294 | 1937 | 481/109/141/61/190 | 17/3/5 |
| all_logreg_event_max_oof_f1 | +0.0975 | +0.9746 | +0.0293 | 1971 | 515/116/153/62/216 | 17/3/5 |
| all_logreg_event_max_fixed_0.01 | +0.0975 | +0.9746 | +0.0293 | 1971 | 515/116/153/62/216 | 17/3/5 |
| all_logreg_event_max_fixed_0.02 | +0.0970 | +0.9711 | +0.0293 | 1954 | 498/112/149/61/204 | 17/3/5 |
| anchor_logreg_event_max_fixed_0.01 | +0.0986 | +0.9746 | +0.0293 | 1972 | 516/118/152/60/216 | 17/3/5 |
| anchor_logreg_event_max_oof_f1 | +0.0979 | +0.9746 | +0.0292 | 1976 | 520/118/153/60/220 | 17/3/5 |
| anchor_logreg_event_max_fixed_0.02 | +0.0996 | +0.9746 | +0.0291 | 1962 | 506/115/149/60/209 | 17/3/5 |
| w16_accept_all_joint | +0.0810 | +0.9701 | +0.0286 | 2013 | 557/118/168/67/248 | 15/5/5 |
| base_extratrees_event_max_fixed_0.01 | +0.0810 | +0.9701 | +0.0286 | 2013 | 557/118/168/67/248 | 15/5/5 |
| base_extratrees_event_max_fixed_0.02 | +0.0810 | +0.9701 | +0.0286 | 2013 | 557/118/168/67/248 | 15/5/5 |
| base_extratrees_event_max_fixed_0.05 | +0.0810 | +0.9701 | +0.0286 | 2013 | 557/118/168/67/248 | 15/5/5 |
| base_extratrees_event_max_fixed_0.10 | +0.0810 | +0.9701 | +0.0286 | 2013 | 557/118/168/67/248 | 15/5/5 |
| base_extratrees_event_max_oof_f1 | +0.1018 | +0.9648 | +0.0285 | 1722 | 266/87/104/60/90 | 15/5/5 |
| all_logreg_event_max_fixed_0.10 | +0.0890 | +0.9759 | +0.0277 | 1892 | 436/85/125/54/180 | 17/3/5 |
| anchor_logreg_event_max_fixed_0.10 | +0.0873 | +0.9742 | +0.0276 | 1908 | 452/91/136/58/189 | 16/4/5 |
| anchor_logreg_event_max_fixed_0.05 | +0.0834 | +0.9724 | +0.0275 | 1926 | 470/99/143/59/195 | 17/3/5 |
| w8_preserve_common_only | +0.0964 | +0.9462 | +0.0274 | 1456 | 0/0/0/0/0 | 16/4/5 |

Interpretation notes:

- W8 common rows are preserved by default for learned variants.
- Only W16-extension rows are controlled.
- This audit is stricter than the earlier dense trajectory threshold sweep.