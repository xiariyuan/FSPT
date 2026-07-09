# V9-A2.3a Feature Builder Correction + Label Overlap Audit Result

Date: 2026-07-08

---

## 1. Purpose

Before building the full W8+W16 feature dataset, V9-A2.3a fixes feature-builder issues and audits oracle label overlap.

The goal is to avoid training the next controller on misleading features or redundant labels.

---

## 2. Code changes

Updated script:

```text
scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py
```

Main corrections:

```text
1. norm_yx_to_pixel no longer clips normalized coordinates into [0, 1].
2. extract_patch now handles fully out-of-bound coordinates safely.
3. Explicit out-of-bound features are added for candidate, native, query anchor, last anchor, and pre-occlusion anchor.
4. Anchor reliability features are added:
   - native visibility at last/preocc anchor;
   - vis score at last/preocc anchor;
   - confidence probability at last/preocc anchor;
   - native-candidate disagreement at anchors;
   - anchor motion jump;
   - last/preocc anchor distance;
   - temporal gap features.
5. Label overlap audit is added for W16, W16-only, and W8-common subsets.
```

---

## 3. Stratified feature smoke v2

Command:

```text
python scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py --smoke-stratified-v2 --rows-per-video 2
```

Output:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified_v2.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified_v2.npz
```

Result:

```text
n_rows              = 49
n_videos            = 25
feature_dim         = 112
finite_rate         = 1.0
valid_feature_min   = 0.6000
valid_feature_mean  = 0.9921
valid_feature_p05   = 1.0
oob_any_row_rate    = 0.0
oob_distance_max    = 0.0
normalized_dist_min = 0.0056
normalized_dist_max = 2.2139
normalized_dist_mean= 0.3517
```

Interpretation:

```text
The corrected feature builder is stable across stratified DAVIS rows.
The stratified sample did not contain out-of-bound coordinates, but the feature builder now preserves and exposes them instead of hiding them by clipping.
```

---

## 4. Label overlap audit

Command:

```text
python scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py --audit-label-overlap
```

Output:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_label_overlap_audit.json
```

### 4.1 W16-only subset

```text
n = 557
candidate_good      = 118
utility_positive    = 118
good_not_worse      = 114
safe_no_damage      = 389
non_false_visible   = 490
non_damage16        = 556
```

Pairwise findings:

```text
candidate_good vs utility_positive:
Jaccard  = 1.0000
Precision= 1.0000
Recall   = 1.0000

candidate_good vs good_not_worse:
Jaccard  = 0.9661
Recall   = 0.9661

candidate_good vs safe_no_damage:
Jaccard  = 0.3033
Recall   = 1.0000
Precision of safe_no_damage for candidate_good = 0.3033
```

Interpretation:

```text
On W16-only rows, candidate_good and utility_positive are identical under the current label construction.
They should not be treated as independent supervision targets.
```

Also:

```text
safe_no_damage is broad and high-recall, but low-precision for candidate_good.
This explains why avoiding damage alone is insufficient for dynamic horizon learning.
```

### 4.2 Common W8 subset

```text
n = 1456
candidate_good      = 647
utility_positive    = 647
good_not_worse      = 482
safe_no_damage      = 965
```

Again:

```text
candidate_good vs utility_positive Jaccard = 1.0
```

So this is not W16-only accidental behavior; it is true across the current label definition.

---

## 5. Consequences for V9-A2.3 controller design

### 5.1 Primary target

Use:

```text
candidate_good
```

as the main extension target.

Do not separately train on `utility_positive` as if it provided independent information.

### 5.2 Secondary target

Use:

```text
good_not_worse
```

as a stricter quality target.

It is close to candidate_good on W16-only rows but not identical.

### 5.3 Risk target

Use:

```text
candidate_bad
false_visible
candidate_worse_px
```

as risk labels.

Do not rely on `damage16` alone, because W16-only damage16 is extremely rare and misses fine degradation.

### 5.4 Policy implication

The first learned dynamic-horizon policy should be:

```text
Preserve W8 common rows by default.
Predict which W16-only rows/events are candidate_good or good_not_worse.
Optionally veto only very high-risk W8 rows as an ablation, not the default.
```

---

## 6. Decision

V9-A2.3a passes.

The corrected feature builder is stable, and the label audit clarifies target choice.

Next:

```text
V9-A2.3b Full W8+W16 Feature Build
```

Expected outputs:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w8_v2.npz
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w16_v2.npz
```

After that:

```text
V9-A2.3c Event-level / extension-level dynamic horizon controller.
```
