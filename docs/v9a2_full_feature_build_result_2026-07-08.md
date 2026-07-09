# V9-A2.3b Full W8+W16 Feature Build Result

Date: 2026-07-08

---

## 1. Purpose

V9-A2.3b builds full anchor/uncertainty feature datasets for W8 and W16, audits common-row consistency, and then creates a canonical joint action dataset for dynamic horizon learning.

---

## 2. Artifacts

Updated scripts:

```text
scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py
scripts/v9a2_build_joint_w8_w16_dataset.py
```

Full feature NPZs:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w8_v3.npz
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w16_v3.npz
```

Reports:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w8_v3.report.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w16_v3.report.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_w8_w16_feature_consistency_audit.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_w16_dataset_report.json
```

Canonical joint dataset:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz
```

---

## 3. Full W8 feature build

```text
n_rows             = 1456
feature_dim_base   = 28
feature_dim_anchor = 115
feature_dim_all    = 143
finite_rate_X_all  = 1.0
row_type_counts    = common_w8: 1456
```

---

## 4. Full W16 feature build

```text
n_rows             = 2013
feature_dim_base   = 28
feature_dim_anchor = 115
feature_dim_all    = 143
finite_rate_X_all  = 1.0
row_type_counts    = common_w8: 1456, w16_extension: 557
```

---

## 5. Common-row feature consistency audit

The audit found that common W8/W16 rows are not feature-identical:

```text
n_common                = 1456
anchor_max_abs_diff     = 38.0
base_max_abs_diff       = 16.0
anchor_mismatched_rows  = 662
base_mismatched_rows    = 662
```

Root cause:

```text
The same (video_id, query_idx, frame_tau) row can be associated with different first_event_t / max_event_age / touch_source_count between W8 and W16 datasets.
W16 can bind the same touched frame to an earlier event because its longer window reaches farther.
```

Important interpretation:

```text
This is not a patch extraction failure.
It means W16 common-row context is not canonical for W8-preserve policy learning.
```

Decision:

```text
For dynamic horizon learning, use W8 features as canonical for common rows.
Use W16 features only for W16-extension rows.
```

---

## 6. Canonical joint dataset

To avoid mixing different event-context definitions for common rows, a canonical joint dataset was built:

```text
v9a2_joint_w8_common_plus_w16_extension_v3.npz
```

Composition:

```text
n_rows          = 2013
common_w8       = 1456
w16_extension   = 557
feature_dim_all = 143
finite_rate     = 1.0
```

Semantics:

```text
common_w8 rows use W8 canonical features.
w16_extension rows use W16-only extension features.
W16 common duplicate rows are skipped.
```

This is the correct input for V9-A2.3c dynamic horizon learning.

---

## 7. Key reflection

A naive W16 full dataset should not be used directly for all rows.

Reason:

```text
Common rows in W16 may carry different event context from W8, which changes event_age and preocc anchor features.
```

Therefore the controller's default policy should be:

```text
Preserve W8 common rows.
Learn whether and where to add W16 extension rows.
```

This exactly matches the dynamic horizon framing.

---

## 8. Decision

V9-A2.3b passes after correction.

Next:

```text
V9-A2.3c Event-Level / Extension-Level Dynamic Horizon Controller
```

Use the canonical joint dataset:

```text
v9a2_joint_w8_common_plus_w16_extension_v3.npz
```

Initial training target:

```text
For w16_extension rows: predict candidate_good / good_not_worse.
For common_w8 rows: preserve by default; high-risk veto only as ablation.
```
