# Final method audit — visibility-aware multi-teacher fusion
## Decision
The best deployable rule found so far is **all-median coordinates + disagreement-gated union visibility**.
Canonical best: `all_median + gated_mean144`, `true_AJ_RD_256 = 0.6189`.

## Main table
| Method | AJ_RD_256 | Δ vs fixed | Δ vs old ensemble | Oracle gap used | AJ_RD | proxy | dmin1 | dmin4 | dmin16 | long20 <4px | long20 <8px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_best_teacher_cotracker3_offline | 0.5546 | 0.0000 | -0.0325 | 0.0000 |  |  |  |  |  |  |  |
| old_masked_median_majority | 0.5871 | 0.0325 | 0.0000 | 0.3375 | 0.4107 | 0.4315 | 0.5847 | 0.5648 | 0.4994 | 0.2945 | 0.5959 |
| all_median_majority | 0.5884 | 0.0338 | 0.0013 | 0.3510 | 0.4106 | 0.4322 | 0.5860 | 0.5668 | 0.5019 | 0.3425 | 0.6301 |
| all_median_union | 0.6171 | 0.0625 | 0.0300 | 0.6490 | 0.4222 | 0.4891 | 0.6136 | 0.6011 | 0.5594 | 0.3425 | 0.6301 |
| reentry_gap1_w4 | 0.6180 | 0.0634 | 0.0309 | 0.6584 | 0.4229 | 0.4849 | 0.6146 | 0.6027 | 0.5571 | 0.3425 | 0.6301 |
| all_median_gated_mean144 | 0.6189 | 0.0643 | 0.0318 | 0.6677 | 0.4235 | 0.4884 | 0.6148 | 0.6039 | 0.5578 | 0.3425 | 0.6301 |
| all_median_gated_mean192 | 0.6189 | 0.0643 | 0.0318 | 0.6677 | 0.4235 | 0.4887 | 0.6149 | 0.6040 | 0.5579 | 0.3425 | 0.6301 |
| oracle_teacher_selection_upper_bound | 0.6509 | 0.0963 | 0.0638 | 1.0000 |  |  |  |  |  |  |  |

## Per-video stability
- Videos improved vs old masked-median ensemble: 15 / 25
- Videos hurt vs old masked-median ensemble: 4 / 25
- Videos improved vs all-median union: 2 / 25
- Videos hurt vs all-median union: 3 / 25

## Interpretation
- The large gain is from changing visibility from majority to union-like visibility.
- The disagreement gate gives only a small additional gain over plain union, which means the route is close to saturation without adding a new teacher or learned reliability model.
- The remaining gap to oracle is about `0.0320` AJ_RD_256. This is the next target for a learned reliability head or an additional re-detection-specific teacher.

## Current method formula
```text
tracks = median(all teacher coordinates)
if mean_pairwise_teacher_disagreement <= tau:
    visibility = union(teacher visibility)
else:
    visibility = majority(teacher visibility)
tau = 144 px in the current best sweep
```

## Next go/no-go
Do not continue broad rule sweeps. Next useful work is either: add TAPNext++/AllTracker as an extra teacher, or train a tiny reliability head to approximate the oracle routing gap.
