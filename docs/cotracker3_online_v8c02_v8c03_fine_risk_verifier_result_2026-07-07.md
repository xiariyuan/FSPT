# CoTracker3 Online V8-C0.2 / V8-C0.3 Fine-Risk Verifier Dataset and Learnability Result

Date: 2026-07-07

Dataset script:

```text
scripts/build_cotracker3_online_v8c02_fine_risk_verifier_dataset.py
```

Verifier learnability script:

```text
scripts/train_cotracker3_online_v8c03_fine_risk_verifier.py
```

Dataset output used for verifier:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz
```

Verifier report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c03_fine_risk_verifier/v8c03_fine_risk_verifier_report.json
```

---

## 1. Purpose

V8-C0.1 showed that the current best default is:

```text
dist_nc_le64_w8_candidate_visible
AJ +0.0964, OA +0.9462, AJ_RD +0.0153, AJ_RD_256 +0.0274
positive/negative/zero videos = 16/4/5
```

The remaining problem is not catastrophic 16px damage. Negative videos are mostly caused by:

```text
1. false candidate visibility on GT-invisible frames,
2. candidate being slightly worse than native at fine thresholds,
3. candidate visible frames that are not actually useful.
```

V8-C0.2 builds a touched-frame dataset for these fine-risk cases. V8-C0.3 tests whether causal features can predict useful/damaging candidate-visible touched frames under video-level leave-one-out validation.

---

## 2. Dataset definition

Base recovery mode:

```text
event filter: dist_nc_le64
window: w8
apply mode: candidate_visible
```

A sample is a unique touched frame:

```text
(video_id, query_idx, frame_tau)
```

A frame enters the dataset if it would be touched by the current default recovery policy and TrackOn2 candidate visibility is true at that frame.

Important design choice:

```text
Overlapping event windows are deduplicated. One video/query/frame appears only once.
```

Feature rule:

```text
Only causal native/candidate/relation features at frame <= tau are used.
No X/X_aug, no V7 OOF, no oracle labels.
```

Label rule:

```text
GT is used only to label whether replacing native by candidate was useful or harmful.
```

---

## 3. Dataset summary

For `dist_nc_le64_w8`:

```text
selected_events = 1617
events_with_touch = 659
unique_touched_frames = 1456
feature_dim = 28
```

Label rates:

```text
gt_visible_rate = 0.7507
false_visible_rate = 0.2493
candidate_better_px_rate = 0.4327
candidate_worse_px_rate = 0.3180
candidate_good_rate = 0.4444
candidate_bad_rate = 0.3221
bad_strict_false_or_worse_rate = 0.5673
```

Error / threshold stats:

```text
native_err_px_mean = 9.7071
candidate_err_px_mean = 6.6880
err_delta_candidate_minus_native_mean = -3.0191

native_safe4_rate = 0.3963
candidate_safe4_rate = 0.4801
repair4_rate = 0.1394
damage4_rate = 0.0556

native_safe8_rate = 0.5185
candidate_safe8_rate = 0.6133
repair8_rate = 0.1312
damage8_rate = 0.0364

native_safe16_rate = 0.6374
candidate_safe16_rate = 0.6957
repair16_rate = 0.0762
damage16_rate = 0.0179
```

Interpretation:

```text
Candidate-visible touched frames are beneficial on average, but not clean.
About 25% are false-visible, and about 32% are worse than native in pixel error.
This justifies a fine-risk verifier.
```

---

## 4. Negative-video label audit

### breakdance

```text
n = 131
false_visible_rate = 0.4427
candidate_better_px_rate = 0.2137
candidate_worse_px_rate = 0.3435
candidate_good_rate = 0.1374
candidate_bad_rate = 0.5725
```

Main issue:

```text
Many candidate-visible frames are GT-invisible, and useful candidate frames are rare.
```

### camel

```text
n = 9
false_visible_rate = 0.4444
candidate_better_px_rate = 0.0000
candidate_worse_px_rate = 0.5556
candidate_good_rate = 0.0000
candidate_bad_rate = 1.0000
```

Main issue:

```text
Candidate is never useful in this touched set; it is either false-visible or worse than native.
```

### shooting

```text
n = 8
false_visible_rate = 0.3750
candidate_better_px_rate = 0.1250
candidate_worse_px_rate = 0.5000
candidate_good_rate = 0.0000
candidate_bad_rate = 0.5000
```

Main issue:

```text
Candidate is mostly false-visible or worse.
```

### car-shadow

```text
n = 23
false_visible_rate = 0.0000
candidate_better_px_rate = 0.2609
candidate_worse_px_rate = 0.7391
candidate_good_rate = 0.0435
candidate_bad_rate = 0.1304
```

Main issue:

```text
No false visible. Candidate is simply more often slightly worse than native at fine thresholds.
```

---

## 5. LOOV learnability result

Validation protocol:

```text
Leave-one-video-out.
No random event split.
```

### 5.1 Good-frame prediction

`candidate_good` base rate:

```text
0.4444
```

Best models:

```text
RandomForest: AP 0.7763, AUC 0.7829
ExtraTrees:   AP 0.7396, AUC 0.7669
LogReg:       AP 0.7229, AUC 0.7583
```

Interpretation:

```text
Useful candidate-visible frames are learnable from causal features.
```

### 5.2 False-visible prediction

`false_visible` base rate:

```text
0.2493
```

Best models:

```text
LogReg:     AP 0.5182, AUC 0.7979
RandomForest: AP 0.4353, AUC 0.7680
ExtraTrees:   AP 0.3912, AUC 0.7334
```

Interpretation:

```text
False-visible risk is also meaningfully learnable, especially linearly.
```

### 5.3 Candidate-worse / strict-bad prediction

`candidate_worse_px`:

```text
Best AUC about 0.6825 with LogisticRegression.
```

`bad_strict_false_or_worse`:

```text
Best AUC only about 0.5770.
```

Interpretation:

```text
The model can detect some false-visible and useful-frame signals, but it is much harder to jointly detect all bad cases.
```

### 5.4 Utility regression

Utility regression is weak:

```text
Ridge R2 = -0.0061
ExtraTrees R2 = 0.0532
```

Interpretation:

```text
Direct utility regression is not reliable yet.
A classifier-style verifier is more promising than utility regression.
```

---

## 6. Combined decision audit

Using ExtraTrees good-score minus strict-bad-score, the best threshold rows look like:

```text
accept = 389 frames, good_rate = 0.7481, bad_rate = 0.4807
accept = 437 frames, good_rate = 0.7162, bad_rate = 0.4874
accept = 486 frames, good_rate = 0.6975, bad_rate = 0.4918
```

Interpretation:

```text
The verifier can enrich for useful frames, but strict-bad rate remains high.
This cannot yet be trusted without trajectory-level application.
```

---

## 7. Reflection

### What is promising

```text
Good-frame prediction is real under video-heldout validation.
False-visible prediction is also real.
```

This means a lightweight fine-risk verifier may help remove camel/shooting/breakdance-style failures.

### What is not solved

```text
Strict bad cases are hard to predict.
Utility regression is weak.
Threshold tables still keep many bad frames.
```

Therefore:

```text
Do not claim learned verifier improves the tracker yet.
The next step must apply OOF verifier scores back to the trajectory and evaluate actual AJ/OA/AJ_RD/AJ_RD_256.
```

---

## 8. Next step

Run:

```text
V8-C0.4 OOF verifier application
```

Script target:

```text
scripts/eval_cotracker3_online_v8c04_apply_fine_risk_verifier.py
```

Use OOF predictions from:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c03_fine_risk_verifier/oof_good_rf.npy
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c03_fine_risk_verifier/oof_false_visible_logreg.npy
```

Evaluate frame acceptance rules such as:

```text
good_score >= threshold
good_score - false_visible_score >= threshold
good_score high and false_visible_score low
```

Compare against:

```text
dist_nc_le64_w8_candidate_visible baseline:
AJ +0.0964, OA +0.9462, AJ_RD +0.0153, AJ_RD_256 +0.0274.
```

Decision rule:

```text
If OOF verifier improves per-video robustness or reduces negative videos without losing too much AJ_RD_256, keep learned verifier.
If it does not, use simple dist_nc_le64_w8_candidate_visible as default and report learned verifier as non-helpful.
```
