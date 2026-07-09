# CoTracker3 Online V7-B0 Raw Vis/Conf Verifier Result — 2026-07-07

## Context

V7-A4 exported raw CoTracker3 online visibility/confidence components with exact parity against the existing true-streaming native cache:

```text
score_max_abs_diff_vs_base = 0
visibility_diff_count_vs_base = 0
track_max_abs_diff_vs_base = 0
score_product_internal_max_abs_diff = 0
```

This made it safe to train event-level raw-component verifiers and run metric simulation.

## Inputs

Feature cache:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz
```

Native/raw-visconf cache:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt
```

Main script:

```text
scripts/train_cotracker3_online_v7b0_raw_visconf_verifier.py
```

Native metric:

```text
AJ       65.2366
OA       90.8186
AJ_RD    0.3534
AJ_RD_256 0.5333
```

Success gate:

```text
AJ delta >= -0.10
OA delta >= -0.10
AJ_RD delta >= +0.0015
AJ_RD_256 delta >= +0.004
```

## Runs

### 1. early4 + all_components

Summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b0_raw_visconf_verifier/v7b0_y_reentry_early4_all_components_summary.json
```

Classification:

```text
AP  0.1183
AUC 0.6377
```

Best constrained threshold:

```text
threshold 0.85
accept 96
precision 18.75%
recall 18.56%

AJ       -0.0833
OA       -0.0228
AJ_RD    +0.0014
AJ_RD_256 +0.0035
```

This is the strongest CoTracker3 online verifier result so far, but it does not pass the full success gate.

### 2. early4 + all

Summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b0_raw_visconf_verifier/v7b0_y_reentry_early4_all_summary.json
```

Classification:

```text
AP  0.1183
AUC 0.6737
```

Best constrained threshold:

```text
threshold 0.80 / 0.85 region
best AJ_RD_256 delta only about +0.0014 to +0.0020 depending threshold
```

The `all` family worsens metric gains despite higher AUC. It should not replace `all_components`.

### 3. early8 + all_components

Summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b0_raw_visconf_verifier/v7b0_y_reentry_early8_all_components_summary.json
```

Classification:

```text
AP  0.1208
AUC 0.6050
```

Best constrained threshold:

```text
threshold 0.85
accept 90
precision 17.78%
recall 11.43%

AJ       -0.0886
OA       -0.0298
AJ_RD    +0.0004
AJ_RD_256 +0.0015
```

Aggressive thresholds can increase AJ_RD_256 to about +0.0062, but AJ/OA damage is too large:

```text
threshold 0.1808
AJ       -0.4782
OA       -0.2966
AJ_RD    +0.0026
AJ_RD_256 +0.0062
```

### 4. early8 + all

Summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b0_raw_visconf_verifier/v7b0_y_reentry_early8_all_summary.json
```

Classification:

```text
AP  0.1262
AUC 0.6393
```

Best constrained threshold:

```text
threshold 0.80
accept 87
precision 18.39%
recall 11.43%

AJ       -0.0830
OA       -0.0219
AJ_RD    +0.0009
AJ_RD_256 +0.0020
```

Again, aggressive thresholds increase AJ_RD_256 but violate AJ/OA constraints.

## Refined threshold sweep for best branch

The strongest branch, `early4 + all_components`, was refined over threshold 0.78–0.90.

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b0_raw_visconf_verifier/refined_early4_all_components/refined_early4_all_components_threshold_report.json
```

Best constrained rows:

```text
threshold 0.850
accept 96
precision 18.75%
recall 18.56%
AJ       -0.0833
OA       -0.0228
AJ_RD    +0.0014
AJ_RD_256 +0.0035

threshold 0.835 / 0.840
accept 101
precision 18.81%
recall 19.59%
AJ       -0.0901
OA       -0.0287
AJ_RD    +0.0014
AJ_RD_256 +0.0035

threshold 0.825
accept 104
AJ       -0.0982
OA       -0.0376
AJ_RD    +0.0014
AJ_RD_256 +0.0035
```

No threshold passes the full success gate:

```text
success_count = 0
```

## Interpretation

V7-A4/V7-B0 is the best CoTracker3 online direction tested so far:

```text
V6-A1 best safe AJ_RD_256 ≈ +0.0024
V7-B0 early4/all_components best safe AJ_RD_256 = +0.0035
```

But it remains just below the success threshold:

```text
AJ_RD target +0.0015, achieved +0.0014
AJ_RD_256 target +0.0040, achieved +0.0035
```

The remaining failure mode is a precision/recall trade-off:

```text
High threshold: safe but too few useful openings.
Low threshold: enough AJ_RD gain but too much AJ/OA damage.
```

## Decision

Do not claim V7-B0 as a successful method.

However, it is a strong near-miss and should be kept as the main CoTracker3 online branch.

Next step should not be another broad feature/model sweep. The appropriate next step is targeted risk control:

```text
V7-B1: damage-aware raw vis/conf verifier.
```

V7-B1 should keep the V7-B0 early4/all_components signal but add an explicit damage model/gate:

```text
1. Benefit model: predicts early4 raw-visconf opening probability.
2. Damage model: predicts whether opening will harm AJ/OA, i.e. GT-occluded or coordinate-unsafe/redundant openings.
3. Decision score: benefit_probability - lambda * damage_probability.
```

Evaluation should still use the same metric gate:

```text
AJ >= native - 0.10
OA >= native - 0.10
AJ_RD >= native + 0.0015
AJ_RD_256 >= native + 0.004
```

If V7-B1 cannot cross the gate, frozen CoTracker3 online lightweight visibility calibration should be considered exhausted.
