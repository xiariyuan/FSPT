# V9-A2.3c Dynamic Horizon Controller Result

## Extension classifier OOF summary

| Feature set / Model | AP | AUC | Brier | Risk@50 |
|---|---:|---:|---:|---:|
| base / logreg | 0.2307 | 0.4011 | 0.2805 | 0.8423 |
| base / hgb | 0.2725 | 0.5949 | 0.2125 | 0.7204 |
| base / extratrees | 0.3233 | 0.6420 | 0.1729 | 0.7240 |
| anchor / logreg | 0.2160 | 0.4928 | 0.2990 | 0.7921 |
| anchor / hgb | 0.2296 | 0.5430 | 0.2128 | 0.7491 |
| anchor / extratrees | 0.2435 | 0.5773 | 0.1802 | 0.7348 |
| all / logreg | 0.1990 | 0.4460 | 0.3282 | 0.8208 |
| all / hgb | 0.2045 | 0.4885 | 0.2307 | 0.7921 |
| all / extratrees | 0.2528 | 0.5855 | 0.1768 | 0.7240 |

## Apply-back baselines

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| native_reject_all | +0.0000 | +0.0000 | +0.0000 | 0 | 0/0/25 |
| w8_preserve_common_only | +0.0964 | +0.9462 | +0.0274 | 1456 | 16/4/5 |
| w16_accept_all_joint | +0.0810 | +0.9701 | +0.0286 | 2013 | 15/5/5 |
| oracle_w8_plus_ext_candidate_good | +0.1955 | +1.0204 | +0.0321 | 1574 | 17/3/5 |
| oracle_w8_plus_ext_good_not_worse | +0.1966 | +1.0106 | +0.0321 | 1570 | 17/3/5 |

## Best learned dynamic-horizon rows

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---|
| learned_all_logreg_event_max_thr_0.0484 | +0.1025 | +0.9711 | +0.0294 | 1937 | 17/3/5 |
| learned_all_logreg_event_max_thr_0.0189 | +0.0993 | +0.9746 | +0.0294 | 1956 | 17/3/5 |
| learned_all_logreg_event_max_thr_0.0069 | +0.0976 | +0.9746 | +0.0293 | 1975 | 17/3/5 |
| learned_all_logreg_event_mean_thr_0.0037 | +0.0976 | +0.9746 | +0.0293 | 1977 | 17/3/5 |
| learned_all_logreg_event_max_thr_0.0120 | +0.0975 | +0.9746 | +0.0293 | 1965 | 17/3/5 |
| learned_all_logreg_event_mean_thr_0.0069 | +0.0975 | +0.9746 | +0.0293 | 1971 | 17/3/5 |
| learned_anchor_logreg_event_max_thr_0.0101 | +0.0986 | +0.9746 | +0.0293 | 1972 | 17/3/5 |
| learned_anchor_logreg_event_mean_thr_0.0067 | +0.0986 | +0.9746 | +0.0293 | 1980 | 17/3/5 |
| learned_anchor_logreg_event_max_thr_0.0067 | +0.0966 | +0.9746 | +0.0292 | 1986 | 16/4/5 |
| learned_anchor_logreg_event_mean_thr_0.0043 | +0.0966 | +0.9746 | +0.0292 | 1990 | 16/4/5 |
| learned_all_logreg_event_max_thr_0.0025 | +0.0907 | +0.9746 | +0.0292 | 1998 | 16/4/5 |
| learned_all_logreg_event_max_thr_0.0037 | +0.0907 | +0.9746 | +0.0292 | 1998 | 16/4/5 |
| learned_all_logreg_event_mean_thr_0.0025 | +0.0907 | +0.9746 | +0.0292 | 1998 | 16/4/5 |
| learned_all_logreg_event_max_thr_0.0277 | +0.0980 | +0.9711 | +0.0291 | 1948 | 17/3/5 |
| learned_anchor_logreg_event_max_thr_0.0043 | +0.0921 | +0.9746 | +0.0291 | 1996 | 16/4/5 |