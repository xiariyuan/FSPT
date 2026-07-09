# V9-A1 Controller-Only + Calibration-Aware Prototype Design

Date: 2026-07-08

Purpose:

```text
Test whether a learned, calibration-aware recovery controller can outperform the hard CVRRM rule before building a full ReEntryBeliefTrack model.
```

This is a stop/go experiment. It intentionally does not build the full tracker yet.

---

## 1. Research question

Current hard rule:

```text
CVRRM = native failure trigger + dist<=64 + finite W + candidate-visible confirmation.
```

Question:

```text
Can a learned controller use the same causal inputs plus simple risk features to improve candidate acceptance, reduce false reacquisition, and provide useful uncertainty/risk scores?
```

If yes, continue toward ReEntryBeliefTrack. If no, keep CVRRM as the paper route and do not scale the model.

---

## 2. Scope

V9-A1 is not a full tracker.

It uses existing caches:

```text
native CoTracker3 online predictions
TrackOn2 bridge candidate predictions
optional TAPNext++ candidate diagnostics
GT only for offline training/evaluation labels
```

It learns only:

```text
candidate trust / accept probability
phase proxy: Track / Coast / Reacquire
risk or uncertainty score
safe-write proxy
```

It does not yet train:

```text
new image backbone
internal candidate generator
multi-hypothesis belief state
full memory bank
```

---

## 3. Sample unit

Primary sample unit:

```text
touched candidate-visible frame inside a CVRRM-style recovery window
```

Secondary sample unit:

```text
re-entry event with a finite window summary
```

Reason:

```text
Frame-level samples directly map to candidate acceptance.
Event-level summaries are useful for phase/risk analysis and robustness.
```

---

## 4. Candidate frame features

All features must be causal at frame tau.

### 4.1 Native state

```text
native_visible_tau
native_score_tau
native_score_delta_recent
native_low_run_length
native_coord_tau
native_motion_recent
```

### 4.2 Candidate state

```text
candidate_visible_tau
candidate_coord_tau
candidate_visible_run_length
candidate_recent_stability
```

### 4.3 Native-candidate consistency

```text
dist_native_candidate_tau
dist_to_last_reliable_native
direction_consistency
velocity_consistency
```

### 4.4 Recovery context

```text
event_age
window_remaining
trigger_type
frames_since_last_reliable_visible
cooldown_context
```

### 4.5 Simple uncertainty proxy

No learned covariance yet. Use proxies:

```text
native_score_inverse
low-run duration
motion disagreement
candidate instability
native-candidate distance normalized by event age
```

### 4.6 Optional anchor similarity

If cached image/patch features are available later:

```text
candidate-to-query feature similarity
candidate-to-last-reliable feature similarity
candidate-to-pre-occlusion feature similarity
```

V9-A1 should not block on this optional feature.

---

## 5. Labels

Labels are generated offline using GT, but the model inputs remain causal.

### 5.1 Accept label

```text
accept_good = candidate is better than native by threshold or candidate error <= threshold.
```

Use multiple variants:

```text
good_4px
good_8px
good_16px
candidate_better_than_native
bad_false_visible_or_worse
```

Reason:

```text
A single threshold may overfit; multiple targets reveal what is learnable.
```

### 5.2 Phase proxy label

```text
Track: native reliable and close to GT.
Coast: native unreliable and candidate not trustworthy.
Reacquire: native unreliable and candidate trustworthy.
```

This is an offline proxy for training/evaluation. In deployment, phase is predicted from causal features.

### 5.3 Risk label

```text
risk = candidate false-visible or candidate worse than native.
```

### 5.4 Safe-write proxy

```text
safe_write = candidate accepted and remains stable/correct for next k frames.
```

This uses future GT only for label generation. It is not used as inference input.

---

## 6. Models to compare

Start simple and interpretable.

```text
1. Logistic regression
2. Gradient boosted trees / histogram gradient boosting
3. Random forest / ExtraTrees
4. Small MLP only if tabular models show signal
```

Why not start with a large network:

```text
If simple tabular models cannot beat CVRRM, scaling a full tracker is unjustified.
```

---

## 7. Validation protocol

Use video-heldout validation, not random frame split.

Recommended splits:

```text
GroupKFold by video
leave-video-group-out
report mean and variance across folds
```

Reason:

```text
Random frame splits leak video-specific dynamics and overestimate controller quality.
```

---

## 8. Applying controller back to trajectories

Evaluation must apply predictions back to full trajectories, not only report AP/AUC.

Procedure:

```text
1. Rebuild CVRRM recovery windows from native predictions.
2. For each candidate-visible frame, compute V9-A1 features.
3. Predict candidate trust / risk.
4. Replace native with candidate only if trust passes threshold and risk is low.
5. Recompute full standard and re-entry metrics.
```

Threshold selection:

```text
Select threshold on validation videos only.
Evaluate on heldout videos.
```

---

## 9. Metrics

### 9.1 Standard TAP-like metrics

```text
AJ
OA
delta_avg
delta_4px
```

### 9.2 Re-entry metrics

```text
AJ_RD
AJ_RD_256
repair16_rate
damage16_rate
positive / negative / zero videos
```

### 9.3 Event-level metrics

```text
re-entry success rate
false reacquisition rate
time-to-reacquire proxy
candidate accept precision / recall
```

### 9.4 Calibration / risk metrics

```text
risk-coverage
ECE for accept probability
Brier score
NLL for accept/risk classification
```

V9-A1 is not yet a full coordinate uncertainty model, so use acceptance-risk calibration first.

---

## 10. Success criteria

V9-A1 passes only if all are true:

```text
1. AJ_RD_256 > CVRRM w8 default on heldout videos.
2. AJ and OA do not materially drop.
3. false reacquisition / damage decreases.
4. risk-coverage improves over raw CVRRM confidence proxies.
5. gains survive video-heldout validation.
```

Soft pass:

```text
If it reduces negative videos substantially but loses small AJ_RD_256, keep as diagnostic but do not scale full model yet.
```

Fail:

```text
If it only improves AP/AUC but not trajectory metrics, do not scale.
```

---

## 11. Stop criteria

Stop V9-A model expansion if:

```text
controller cannot beat CVRRM rule after heldout trajectory application;
thresholds are unstable across folds;
heldout gains are only from one video;
standard metrics degrade more than the re-entry gain justifies;
risk scores are uncalibrated and post-hoc calibration does not help.
```

---

## 12. Expected outputs

Script:

```text
scripts/v9a1_controller_calibration_aware_prototype.py
```

Reports:

```text
outputs/paper_discovery_2026-07-05/v9a1_controller_calibration_aware_prototype/v9a1_report.json
outputs/paper_discovery_2026-07-05/v9a1_controller_calibration_aware_prototype/v9a1_fold_tables.csv
```

Docs:

```text
docs/v9a1_controller_calibration_aware_prototype_result_2026-07-08.md
```

---

## 13. Implementation outline

```text
1. Load native and candidate caches.
2. Rebuild CVRRM events / windows exactly as V8-C0.1 sanity check did.
3. Build candidate-visible touched-frame dataset.
4. Add GT-derived labels for accept/risk/safe-write.
5. Train models with GroupKFold by video.
6. Select thresholds on validation folds.
7. Apply predictions back to trajectories.
8. Compute standard, re-entry, event-level, and risk-calibration metrics.
9. Compare against CVRRM w8, CVRRM w16, and learned verifier V8-C0.4.
```

---

## 14. Design self-review

Placeholder scan:

```text
No unresolved placeholders remain. Optional anchor similarity is explicitly non-blocking.
```

Scope check:

```text
Focused on controller-only prototype. Does not include internal candidate head or full belief tracker.
```

Ambiguity check:

```text
The pass/fail criteria are explicit. The validation split must be video-heldout.
```

Main risk:

```text
If dataset size is too small, learned controller may overfit. Therefore trajectory-level heldout application is mandatory.
```
