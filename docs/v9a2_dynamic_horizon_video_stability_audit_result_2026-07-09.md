# V9-A2.3c Video Stability Audit

## Variant summary

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted ext/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| w8_preserve_common_only | +0.0964 | +0.9462 | +0.0274 | 0/0/0/0/0 | 16/4/5 |
| w16_accept_all_joint | +0.0810 | +0.9701 | +0.0286 | 557/118/168/67/248 | 15/5/5 |
| oracle_ext_candidate_good | +0.1955 | +1.0204 | +0.0321 | 118/118/0/0/4 | 17/3/5 |
| learned_all_logreg_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| learned_all_logreg_event_max_oof_f1 | +0.0975 | +0.9746 | +0.0293 | 515/116/153/62/216 | 17/3/5 |
| learned_anchor_logreg_event_max_fixed_0.01 | +0.0986 | +0.9746 | +0.0293 | 516/118/152/60/216 | 17/3/5 |

## Learned fixed 0.05 vs W16 by video

Defined AJ_RD_256 videos=25; skipped undefined=5. Summary: better=4, worse=2, equal=19, mean_diff=+0.000665

| Worst videos learned-minus-W16 | Δ |
|---|---:|
| bmx-trees | -0.001871 |
| parkour | -0.000389 |
| breakdance | +0.000000 |
| camel | +0.000000 |
| car-roundabout | +0.000000 |
| cows | +0.000000 |
| dogs-jump | +0.000000 |
| drift-chicane | +0.000000 |
| gold-fish | +0.000000 |
| horsejump-high | +0.000000 |

| Best videos learned-minus-W16 | Δ |
|---|---:|
| drift-straight | +0.010417 |
| bike-packing | +0.004325 |
| dance-twirl | +0.002278 |
| car-shadow | +0.001873 |
| soapbox | +0.000000 |
| shooting | +0.000000 |
| pigs | +0.000000 |
| paragliding-launch | +0.000000 |
| motocross-jump | +0.000000 |
| mbike-trick | +0.000000 |