# V9-A2.3a2 Preocc Anchor Strict-Fix + Smoke Recheck Result

Date: 2026-07-08

---

## 1. Purpose

V9-A2.3a2 fixes the pre-occlusion anchor definition before full W8+W16 feature build.

Previous risk:

```text
preocc_t could be selected from first_event_t itself.
```

This is unsafe because the event frame may already be low-confidence, invisible, or drifting.

Correct definition:

```text
preocc_t must be strictly earlier than first_event_t.
```

---

## 2. Code changes

Updated:

```text
scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py
```

Main changes:

```text
1. preocc anchor search now starts from first_event_t - 1.
2. preocc anchor never uses the event trigger frame.
3. Added explicit features:
   - preocc_anchor_found
   - preocc_anchor_age
   - preocc_anchor_score
4. Added metadata fields:
   - first_event_t
   - preocc_found
   - preocc_strictly_before_event
5. Added --smoke-stratified-v3.
```

---

## 3. Stratified smoke v3

Command:

```text
python scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py --smoke-stratified-v3 --rows-per-video 2
```

Result:

```text
n_rows                         = 49
n_videos                       = 25
feature_dim                    = 115
finite_rate                    = 1.0
valid_p05                      = 1.0
oob_any_row_rate               = 0.0
oob_distance_max               = 0.0
strict_preocc_rate_in_preview  = 1.0
```

Output:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified_v3.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified_v3.npz
```

Interpretation:

```text
The corrected feature builder remains stable across videos.
Feature dimension increases from 112 to 115 because of the three new strict-preocc features.
```

---

## 4. Full W8/W16 preocc strict audit

Output:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_preocc_strict_anchor_audit.json
```

### W8 dataset

```text
n rows          = 1456
strict count    = 1456
found count     = 1456
fallback count  = 0
pre_eq_event    = 0
pre_gt_event    = 0
```

Rates:

```text
strict_rate     = 1.0
found_rate      = 1.0
fallback_rate   = 0.0
```

### W16 dataset

```text
n rows          = 2013
strict count    = 2013
found count     = 2013
fallback count  = 0
pre_eq_event    = 0
pre_gt_event    = 0
```

Rates:

```text
strict_rate     = 1.0
found_rate      = 1.0
fallback_rate   = 0.0
```

---

## 5. Additional finding: last anchor and preocc anchor differ often

W8:

```text
last_t == preocc_t: 860 rows
last_t != preocc_t: 596 rows
last_ne_pre_rate: 0.4093
```

W16:

```text
last_t == preocc_t: 850 rows
last_t != preocc_t: 1163 rows
last_ne_pre_rate: 0.5777
```

Interpretation:

```text
last_anchor and preocc_anchor are meaningfully different for many rows, especially in W16.
This supports keeping both features.

last_anchor = recent causal evidence, possibly after the event.
preocc_anchor = strict pre-event identity anchor.
```

---

## 6. Decision

V9-A2.3a2 passes.

The identity-anchor semantics are now clean enough for full feature build.

Next:

```text
V9-A2.3b Full W8+W16 Feature Build
```

Expected outputs:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w8_v3.npz
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w16_v3.npz
```
