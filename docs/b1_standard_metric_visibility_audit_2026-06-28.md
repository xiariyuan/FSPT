# B1 Standard Metric and Visibility Audit — 2026-06-28

## Decision

The current B1 re-entry rule `vis4_gated288` should remain the AJ_RD mainline, but it should not be claimed as a global standard TAP improvement.

Key result:

```text
vis4_gated288 improves re-entry visibility recall / missed-reentry behavior,
but it has a standard TAP AJ tradeoff compared with the old3 gated baseline and fixed best teacher.
```

## Summary table

| method | AJ_256 | OA_256 | delta_avg_256 | vis precision | vis recall | false visible | missed visible | reentry pred visible | reentry missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_cotracker3_offline | 70.0510 | 92.1543 | 82.3206 | 0.966747 | 0.910696 | 0.021087 | 0.057369 | 0.623049 | 0.376951 |
| old_masked_median_majority | 50.5340 | 69.6137 | 81.5800 | 0.949982 | 0.633547 | 0.023019 | 0.280845 | 0.664998 | 0.335002 |
| old3_all_median_gated144 | 49.6122 | 68.3640 | 68.6665 | 0.904869 | 0.656725 | 0.050216 | 0.266144 | 0.825117 | 0.174883 |
| b1_all_median4_gated192 | 48.8572 | 72.6301 | 62.6903 | 0.906673 | 0.716494 | 0.053394 | 0.220304 | 0.873828 | 0.126172 |
| vis4_gated288 | 47.4046 | 72.6762 | 76.4913 | 0.906145 | 0.717654 | 0.053842 | 0.219396 | 0.877995 | 0.122005 |
| safe_router_ridge_m0p02 | 47.5491 | 73.3553 | 75.7517 | 0.908700 | 0.728257 | 0.053736 | 0.212711 | 0.867117 | 0.132883 |

## Important comparisons

### vis4_gated288 vs old3_all_median_gated144

- AJ_256 changes from `49.6122` to `47.4046`: `-2.2076`.
- OA_256 changes from `68.3640` to `72.6762`: `+4.3122`.
- delta_avg_256 changes from `68.6665` to `76.4913`: `+7.8248`.
- reentry missed visibility changes from `0.174883` to `0.122005`: `-0.052878`.

Interpretation: the B1 rule improves re-entry recall and delta_avg, but the more aggressive visibility behavior lowers standard AJ.

### vis4_gated288 vs fixed best

- Fixed best standard AJ_256 is `70.0510`, higher than `vis4_gated288` `47.4046`.
- Fixed best reentry missed visibility is `0.376951`, much worse than `vis4_gated288` `0.122005`.

Interpretation: fixed best remains stronger on global standard TAP AJ, while B1 is specifically better for re-entry visibility / AJ_RD.

### safe router

The first safe router has AJ_256 `47.5491` and improves OA/visibility recall slightly, but its AJ_RD result remains below the current rule baseline. It is not promoted as the main method.

## Working rule update

Use careful wording in any paper or report:

```text
B1 improves re-entry / AJ_RD and reduces missed re-entry visibility,
but it trades off global standard TAP AJ.
```

Do not claim that `vis4_gated288` is a globally better TAP tracker than the fixed best teacher. Claim it as a re-entry reliability method.

## Next step

The next useful step is cross-dataset validation, starting with RGB-Stacking or a small Kinetics subset, to test whether the re-entry reliability pattern generalizes beyond DAVIS.
