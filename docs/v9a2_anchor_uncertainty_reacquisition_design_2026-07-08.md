# V9-A2 Anchor- and Uncertainty-Aware Reacquisition Prototype Design

Date: 2026-07-08

Purpose:

```text
V9-A2 tests whether the missing ingredients after V9-A1 are anchor identity evidence and uncertainty-aware consistency, rather than another larger learned gate.
```

---

## 1. Why V9-A2 is needed

V9-A1 showed that candidate-good / risk is learnable, but a tabular learned gate over existing V8-C0.2 features did not beat CVRRM accept-all on trajectory-level AJ_RD_256.

Key V9-A1 result:

```text
CVRRM accept-all touched frames:
AJ_RD_256 Δ = +0.0274

V9-A1 ExtraTrees OOF-F1:
AJ_RD_256 Δ = +0.0212
```

Threshold sweep showed:

```text
The best AJ_RD_256 rows are still accept-all thresholds, i.e. CVRRM itself.
Once the controller actually filters frames, AJ_RD_256 decreases.
```

Interpretation:

```text
The failure is not only model capacity.
The existing V8-C0.2 feature set lacks identity evidence and calibrated uncertainty.
```

Therefore V9-A2 should not be a bigger MLP gate. It should add two specific structures:

```text
1. anchor identity similarity;
2. uncertainty-aware candidate consistency.
```

---

## 2. Asset audit before design

### 2.1 Existing usable assets

The project already contains scripts related to:

```text
build_reentry_patch_similarity_dataset.py
analyze_reentry_patch_similarity.py
audit_reentry_identity_patch_signal.py
audit_reentry_identity_signal_fast.py
build_online_recovery_anchor_dataset_v3.py
build_reentry_frame_keep_dataset.py
train_reentry_frame_keep_decoder.py
build_b2wa_dinov3_dense_features.py
audit_b2wa_dinov3_signal.py
```

There are also earlier 2026-06-27 outputs for:

```text
B2WA RGB appearance pilots
CLIP patch pilots
DINOv3 dense pilots
B2WV counterfactual window policies
DINOv3 reentry appearance patches on RGB dev subsets
```

These prove that identity / appearance signals have been explored before.

### 2.2 Critical limitation

For the current V8-C0.2 / CVRRM DAVIS full30 touched-frame dataset:

```text
There is no ready-to-use aligned anchor identity feature matrix in outputs/paper_discovery_2026-07-05.
```

Therefore V9-A2 must first build a new aligned feature dataset for the current touched-frame rows.

### 2.3 Design consequence

V9-A2 should start with cheap deterministic patch features on DAVIS, not immediately DINO/CLIP.

Reason:

```text
Patch features are fast, reproducible, and can be computed directly at query / last reliable / pre-occlusion / candidate coordinates.
DINO/CLIP can be added later after the coordinate and metadata alignment is proven.
```

---

## 3. V9-A2 research question

Primary question:

```text
Can anchor identity similarity + uncertainty-aware distance preserve CVRRM's re-entry gain while reducing false candidate acceptance?
```

Secondary question:

```text
Can these features produce a better risk-coverage curve than V9-A1 and raw CVRRM confidence proxies?
```

This is a stop/go prototype for the ReEntryBeliefTrack route.

---

## 4. Sample unit

The sample unit remains:

```text
candidate-visible touched frame inside a CVRRM dist<=64, W=8 recovery window.
```

This matches V8-C0.2 and V9-A1, so results remain directly comparable.

Each row is identified by:

```text
video_id
query_idx
frame_tau
first_event_t
last_native_visible_t
```

---

## 5. Anchor definitions

V9-A2 defines three positive anchors and one optional negative memory proxy.

### 5.1 Query anchor

```text
query_anchor = patch / feature at the original query point and query frame.
```

Purpose:

```text
Long-term identity reference.
```

Risk:

```text
Query appearance may differ after deformation or view change.
```

### 5.2 Last reliable anchor

```text
last_reliable_anchor = patch / feature at last_native_visible_t before frame_tau.
```

Use metadata:

```text
last_native_visible_t from V8-C0.2 meta_json.
```

Purpose:

```text
Most recent reliable appearance before recovery.
```

Risk:

```text
If native was already drifting, this anchor may be contaminated.
```

### 5.3 Pre-occlusion anchor

```text
pre_occ_anchor = patch / feature at the last reliable visible frame before first_event_t.
```

Fallback:

```text
If unavailable, use last_reliable_anchor.
```

Purpose:

```text
Cleaner identity reference before native failure.
```

### 5.4 Negative memory proxy

Do not build full negative memory yet. Use a proxy:

```text
hard_negative_anchor = candidate patches from rows where candidate is visible but y_false_visible=1 or y_candidate_bad=1.
```

For each candidate row, compute:

```text
max similarity to same-video negative anchors from earlier frames only.
```

Causal constraint:

```text
A row can only use negative anchors from earlier tau within the same video/query if evaluating online.
For offline analysis, a non-causal negative pool may be separately marked diagnostic-only.
```

V9-A2 default should be causal negative proxy only.

---

## 6. Candidate feature extraction

### 6.1 Default feature type: RGB patch descriptor

Compute small patch descriptors at:

```text
candidate coordinate at frame_tau
query anchor coordinate
last reliable anchor coordinate
pre-occlusion anchor coordinate
```

Patch radii:

```text
r = 5, 9, 17
```

Feature families:

```text
normalized cross correlation
mean absolute difference
L2 distance
gradient magnitude stats
color histogram similarity
patch validity / border fraction
```

This follows the existing patch-similarity scripts and keeps V9-A2 light.

### 6.2 Later optional feature types

Only after RGB patch alignment works:

```text
DINOv3 patch descriptors
CLIP patch descriptors
TrackOn2 internal descriptors if accessible
CoTracker hidden features if accessible
```

These are not required for V9-A2 pass/fail.

---

## 7. Uncertainty-aware consistency

V8/CVRRM uses:

```text
dist(native, candidate) <= 64 px
```

V9-A2 should add an uncertainty proxy:

```text
sigma_proxy = f(event_age, low_run_length, native_score, native_recent_motion, native-candidate disagreement)
```

Minimal deterministic proxy:

```text
sigma_proxy = sqrt( sigma0^2 + a * event_age + b * low_run + c * motion_disagreement )
```

Then compute:

```text
normalized_distance = dist(native, candidate) / max(sigma_proxy, eps)
```

Purpose:

```text
A far candidate may be plausible after long uncertainty.
A near candidate may still be suspicious if native state was highly confident.
```

This is the first step toward Mahalanobis-style consistency, without requiring a learned covariance model.

---

## 8. Temporal evidence

V9-A1 operated mostly at touched-frame level. V9-A2 should add short temporal features:

```text
candidate_visible_run_length
candidate_coord_stability over tau-2:tau
anchor_similarity_stability over tau-2:tau
normalized_distance_trend
trust_score_smoothness
```

Causal constraint:

```text
At frame tau, only frames <= tau may be used.
```

This moves the controller closer to Reacquire evidence accumulation instead of single-frame gate.

---

## 9. Labels

Reuse V8-C0.2 labels:

```text
y_candidate_good
y_candidate_bad
y_false_visible
y_candidate_worse_px
y_repair16
y_damage16
y_utility
```

Add V9-A2 derived labels:

```text
identity_good_proxy = y_candidate_good and not y_false_visible
identity_bad_proxy = y_false_visible or y_candidate_bad
safe_accept_proxy = y_candidate_good and not y_candidate_worse_px
```

Do not use GT-derived labels as inference inputs.

---

## 10. Validation protocol

Use the same strict protocol as V9-A1:

```text
GroupKFold by video.
No random frame split.
Threshold selection on training/validation folds only.
Apply predictions back to full trajectories.
```

This prevents repeating the common mistake:

```text
high AP/AUC but no trajectory-level gain.
```

---

## 11. Apply-back policies

Evaluate multiple policies:

### 11.1 CVRRM baseline

```text
accept all candidate-visible touched frames.
```

### 11.2 V9-A1 baseline

```text
existing OOF tabular controller.
```

### 11.3 V9-A2 anchor-trust policy

```text
accept if anchor_trust >= threshold.
```

### 11.4 V9-A2 uncertainty-normalized policy

```text
accept if normalized_distance <= threshold and candidate_visible.
```

### 11.5 Combined conservative policy

```text
accept if anchor_trust high OR uncertainty-normalized distance plausible.
```

Rationale:

```text
Pure filtering may lose re-entry recall, as V9-A1 showed.
The combined policy must preserve recall while vetoing only high-risk false reacquisitions.
```

---

## 12. Metrics

### Standard metrics

```text
AJ
OA
delta_avg
delta_4px
```

### Re-entry metrics

```text
AJ_RD
AJ_RD_256
repair16_rate
damage16_rate
positive / negative / zero videos
```

### Risk / calibration metrics

```text
AP / AUC for identity_good_proxy
Brier score
risk-coverage
false-visible accepted rate
candidate-worse accepted rate
```

### Qualitative diagnostics

Report worst affected videos:

```text
shooting
car-shadow
camel
breakdance
libby
```

---

## 13. Success criteria

### Strong pass

```text
AJ_RD_256 > CVRRM +0.0274
AJ/OA do not materially drop
negative videos <= CVRRM negative videos
```

### Useful pass

```text
AJ_RD_256 >= +0.025
negative videos are clearly reduced
AJ/OA improve over CVRRM
risk-coverage improves over V9-A1
```

### Fail

```text
AJ_RD_256 < +0.023
or improvements only appear in AP/AUC but not apply-back
or gains are dominated by one video
or thresholds are unstable across folds
```

If V9-A2 fails:

```text
Stop learned-controller route.
Move to internal candidate head / multi-hypothesis belief design directly.
```

---

## 14. Expected artifacts

Script:

```text
scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py
```

Feature dataset:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w8.npz
```

Reports:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_report.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_threshold_sweep.json
```

Result doc:

```text
docs/v9a2_anchor_uncertainty_reacquisition_result_2026-07-08.md
```

---

## 15. Implementation stages

### Stage 1: feature builder smoke

```text
Build RGB patch anchor/candidate features for 20 rows.
Verify coordinate alignment, patch validity, and metadata mapping.
```

### Stage 2: full V8-C0.2 touched-frame feature build

```text
Build RGB patch similarity + uncertainty proxy for all 1456 rows.
```

### Stage 3: video-heldout controller

```text
Train tabular controller with V9-A1 features + V9-A2 anchor/uncertainty features.
```

### Stage 4: apply-back

```text
Apply policies back to full trajectories exactly as V9-A1/V8-C0.4.
```

### Stage 5: stop/go judgement

```text
Decide whether to continue to V9-A3 internal candidate head.
```

---

## 16. Important cautions

1. Do not use GT patch at frame_tau as an input.
2. Do not use future frames for anchor similarity unless explicitly marked diagnostic-only.
3. Do not use non-causal negative memory in the default result.
4. Do not claim full ReEntryBeliefTrack from this prototype.
5. Do not treat DINO/CLIP pilot results from RGB dev as automatically valid on DAVIS full30.
6. Do not scale to full model unless apply-back beats or usefully matches CVRRM.

---

## 17. Self-review

Placeholder scan:

```text
No unresolved placeholder remains.
```

Scope check:

```text
V9-A2 is still a prototype. It adds anchor identity and uncertainty proxy, but not full internal candidate generation or multi-hypothesis belief.
```

Core risk:

```text
RGB patch identity may be too weak under appearance changes. If so, V9-A2 should report that as evidence that semantic anchors or internal candidate heads are necessary.
```
