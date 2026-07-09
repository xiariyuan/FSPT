# V9-A2.1 Stratified Feature Smoke + W8/W16 Action-Space Audit Result

Date: 2026-07-08

---

## 1. Purpose

V9-A2.1 checks two prerequisites before full V9-A2 training:

```text
1. Can anchor / uncertainty features be extracted reliably across DAVIS videos?
2. Does W16 extension provide useful action space beyond CVRRM W8?
```

This is necessary because V9-A1 showed that filtering only within W8 touched frames cannot beat CVRRM accept-all on AJ_RD_256.

---

## 2. Artifacts

Script:

```text
scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py
```

Stratified feature smoke:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified.npz
```

Action-space audit:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_w8_w16_action_space_audit.json
```

---

## 3. Stratified feature smoke result

Command:

```text
python scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py --smoke-stratified --rows-per-video 2
```

Result:

```text
n_rows       = 49
n_videos     = 25
feature_dim  = 69
finite_rate  = 1.0
valid_p05    = 1.0
```

Interpretation:

```text
Anchor/candidate patch extraction is valid across DAVIS videos.
Query / last / pre-occlusion / candidate coordinate mapping is working.
The current deterministic RGB patch feature pipeline is ready for full feature build.
```

Caution:

```text
This validates data alignment, not predictive quality.
```

---

## 4. W8/W16 action-space audit

Command:

```text
python scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py --audit-action-space
```

Counts:

```text
W8 rows        = 1456
W16 rows       = 2013
Common rows    = 1456
W8-only rows   = 0
W16-only rows  = 557
```

This means:

```text
W16 strictly extends W8 by 557 additional candidate-visible touched frames.
```

---

## 5. W16-only label statistics

For the 557 W16-only extension rows:

```text
y_candidate_good_mean     = 0.2118
y_candidate_good_sum      = 118
y_candidate_bad_mean      = 0.3016
y_false_visible_mean      = 0.1203
y_candidate_worse_px_mean = 0.4452
y_repair16_mean           = 0.0377
y_damage16_mean           = 0.0018
y_utility_mean            = -0.0244
```

Interpretation:

```text
W16-only frames contain real useful recovery signal: 118 candidate-good frames.
Damage16 is extremely rare: only about 0.18% of W16-only rows.
However, many W16-only frames are still candidate-worse or low utility, so naive accept-all W16 is not optimal.
```

---

## 6. Videos contributing useful W16-only frames

Top W16-only candidate-good contributors include:

```text
bmx-trees       27 good / 108 W16-only rows
parkour         21 good / 32 W16-only rows
bike-packing    16 good / 55 W16-only rows
india           14 good / 61 W16-only rows
dance-twirl     12 good / 84 W16-only rows
scooter-black    8 good / 8 W16-only rows
```

Only one W16-only damage16 event appears in this audit, from:

```text
dance-twirl
```

---

## 7. Main conclusion

The audit strongly supports dynamic horizon / action-space expansion.

```text
If V9-A2 only filters W8 rows, it is likely upper-bounded by CVRRM W8 accept-all.
W16-only rows add 557 additional decisions and 118 candidate-good opportunities.
Therefore V9-A2 should become W8 + W16 joint action selection, not W8-only filtering.
```

The correct next formulation is:

```text
V9-A2.2 = full anchor/uncertainty feature build for W8 and W16 rows.
V9-A2.3 = video-heldout dynamic horizon controller.
```

---

## 8. Revised next step

Do not jump directly to W8-only full training.

Next:

```text
V9-A2.2 Full W8+W16 Anchor/Uncertainty Feature Build
```

Expected outputs:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w8.npz
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w16.npz
```

After that:

```text
V9-A2.3 Dynamic horizon controller:
- choose W8-only;
- extend to W16;
- veto risky frames;
- preserve W8 recall where uncertain.
```
