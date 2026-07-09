# CoTracker3 Online V8-C0.1 Recovery Risk Audit Result

Date: 2026-07-07  
Script: `scripts/eval_cotracker3_online_v8c01_recovery_risk_audit.py`  
Report: `outputs/paper_discovery_2026-07-05/cotracker3_online_v8c01_recovery_risk_audit/v8c01_recovery_risk_audit_report.json`

---

## 1. Purpose

V8-C0 showed that a strict-causal simple recovery policy already reaches the target range:

```text
all_events_w16_candidate_visible:
AJ +0.0753, OA +0.9598, AJ_RD +0.0162, AJ_RD_256 +0.0286.
```

V8-C0.1 audits whether this result is robust and genuinely causal:

```text
1. Rebuild event proposals directly from native cache and compare to old meta event list.
2. Re-run w8/w16 recovery on both meta and rebuilt event lists.
3. Test simple risk filters.
4. Inspect negative videos.
```

No X/X_aug, V7 OOF scores, or oracle labels are used for policy selection.

---

## 2. Event proposal rebuild check

The causal event proposal was rebuilt directly from native cache using the historical event-builder rules:

```text
low_mask = score <= 0.60 or native invisible
support_score = 0.80
min_score = 0.05
trend_min = -0.10
min_low_run = 1
trigger_stride = 2
event_cooldown = 2
```

Comparison:

```text
n_meta = 1686
n_rebuilt = 1686
n_intersection = 1686
n_only_meta = 0
n_only_rebuilt = 0
Jaccard = 1.0
```

Conclusion:

```text
The old meta event list is exactly reproducible from causal native-cache history.
This removes the main concern that V8-C0 relied on a non-causal/oracle event proposal.
```

---

## 3. Key policy results

### 3.1 Strongest high-gain policy

Best high-gain rows remain effectively the same after rebuilt-event verification:

```text
dist_nc_le64_w16_candidate_visible:
AJ        +0.0810
OA        +0.9701
AJ_RD     +0.0162
AJ_RD_256 +0.0286
positive videos = 15
negative videos = 5
zero videos = 5
```

This is nearly identical to:

```text
all_w16_candidate_visible:
AJ        +0.0753
OA        +0.9598
AJ_RD     +0.0162
AJ_RD_256 +0.0286
positive videos = 15
negative videos = 5
zero videos = 5
```

The `dist_nc_le64` filter removes 69 event proposals but does not change AJ_RD_256, while slightly improving AJ/OA and false-visible rate.

Damage / risk stats for `dist_nc_le64_w16_candidate_visible`:

```text
selected_events = 1617
event_count_with_touch = 408
touched_frames = 2013
false_visible_rate_on_touched = 0.2136
damage16_rate = 0.0171
```

### 3.2 Safer default candidate

For w8:

```text
dist_nc_le64_w8_candidate_visible:
AJ        +0.0964
OA        +0.9462
AJ_RD     +0.0153
AJ_RD_256 +0.0274
positive videos = 16
negative videos = 4
zero videos = 5
```

Compared with w16, w8 gives slightly lower full30 AJ_RD_256 but better per-video robustness:

```text
w16 AJ_RD_256 delta: +0.0286, positive/negative/zero = 15/5/5
w8  AJ_RD_256 delta: +0.0274, positive/negative/zero = 16/4/5
```

Current default recommendation:

```text
Use dist_nc_le64_w8_candidate_visible as the safer default.
Use dist_nc_le64_w16_candidate_visible as the high-gain ablation.
```

---

## 4. Risk-filter audit

### 4.1 Distance/speed filters

Simple distance/speed filters do not materially improve re-entry robustness.

Examples:

```text
dist_nc_le64_w16_candidate_visible:
AJ_RD_256 +0.0286, positive/negative/zero = 15/5/5

cand_speed_le32_w16_candidate_visible:
AJ_RD_256 +0.0282, positive/negative/zero = 15/5/5

safe_motion_combo_w16_candidate_visible:
AJ_RD_256 +0.0277, positive/negative/zero = 14/6/5
```

Interpretation:

```text
Most wrong recoveries are not caused by absurd native-candidate distance or obvious candidate speed outliers.
Simple geometric filters do not solve negative videos.
```

### 4.2 Native score filters

Score filters reduce coordinate damage but sacrifice overall AJ/OA and do not remove negative videos.

For w16:

```text
score_lt_0p50_w16_candidate_visible:
AJ        +0.0576
OA        +0.6896
AJ_RD_256 +0.0280
damage16_rate = 0.0082
positive/negative/zero = 15/5/5

score_lt_0p50_or_longrun8_w16_candidate_visible:
AJ        +0.0581
OA        +0.7164
AJ_RD_256 +0.0281
damage16_rate = 0.0079
positive/negative/zero = 15/5/5
```

For w8:

```text
score_lt_0p50_or_longrun8_w8_candidate_visible:
AJ        +0.0597
OA        +0.6734
AJ_RD_256 +0.0259
positive/negative/zero = 16/4/5
```

Interpretation:

```text
Score filters reduce damage16_rate, but they also remove many useful recoveries and lower AJ/OA.
They are not a clear default.
```

### 4.3 Event-frame candidate-visible filters

Filtering events by candidate visibility at event_t loses too much recall:

```text
cand_vis_t_w16_candidate_visible:
AJ_RD_256 +0.0193
positive/negative/zero = 9/4/12
```

Interpretation:

```text
Do not require candidate visibility at the event frame.
The strong policy works because all native failure events enter recovery mode and candidate observations can confirm later within the horizon.
```

This supports the current mechanism:

```text
broad native-failure event proposal + causal candidate-visible confirmation inside horizon
```

---

## 5. Negative-video audit

Negative videos for `all_w16_candidate_visible`:

```text
camel
shooting
breakdance
drift-straight
car-shadow
```

### 5.1 breakdance

Totals:

```text
touched = 494
gt_visible_touched = 191
gt_invisible_touched = 303
false_visible = 303
repair16 = 0
damage16 = 0
improve = 65
worse = 126
```

Interpretation:

```text
Main failure is false visible during GT-invisible frames and many candidate-worse frames.
TrackOn2 candidate visibility is over-optimistic for this video.
```

### 5.2 camel

Totals:

```text
touched = 31
gt_visible_touched = 24
gt_invisible_touched = 7
false_visible = 7
repair16 = 0
damage16 = 0
improve = 0
worse = 24
```

Interpretation:

```text
Candidate replacement is consistently worse than native on visible touched frames, but not enough to cross the 16px damage threshold.
AJ_RD drops because candidate worsens fine-grained localization / threshold averages.
```

### 5.3 shooting

Totals:

```text
touched = 16
gt_visible_touched = 9
gt_invisible_touched = 7
false_visible = 7
repair16 = 0
damage16 = 0
improve = 1
worse = 8
```

Interpretation:

```text
Small number of candidate-visible frames, but mostly false-visible or worse than native.
```

### 5.4 car-shadow / drift-straight

These are mostly fine-threshold degradation rather than catastrophic 16px damage:

```text
car-shadow: repair16 = 0, damage16 = 0, improve = 35, worse = 89
drift-straight: repair16 = 0, damage16 = 0, improve = 11, worse = 9
```

Interpretation:

```text
The candidate is often within 16px but worse than native at tighter thresholds.
Current damage16-only filters cannot catch these cases.
```

---

## 6. Main reflection

### 6.1 What is now solid

```text
1. Event proposal is strictly causal and exactly reproducible.
2. Simple candidate-visible recovery mode is not redundant with TrackOn2 standalone.
3. AJ_RD_256 +0.027 to +0.029 is achievable without learned selector or oracle labels.
```

### 6.2 What is not solved

```text
1. Negative videos remain.
2. Simple motion/score/distance filters do not eliminate them.
3. False-positive candidate visibility is a real problem.
4. Some negative cases are fine-threshold degradation, not 16px catastrophic damage.
```

### 6.3 Implication for CRRP

Do not train a large learned policy yet.

The next useful model, if needed, should not learn a generic replace gate. It should specifically learn:

```text
candidate trust under fine-threshold localization and false-visible risk.
```

The label should capture:

```text
- candidate better than native across fine thresholds,
- false-visible probability,
- whether candidate-visible is actually trustworthy in this video/query context.
```

---

## 7. Current recommended method variant

Recommended default:

```text
dist_nc_le64_w8_candidate_visible
```

Reason:

```text
AJ_RD_256 +0.0274,
positive/negative/zero = 16/4/5,
AJ +0.0964, OA +0.9462,
slightly safer than w16 with only -0.0012 AJ_RD_256 loss.
```

High-gain ablation:

```text
dist_nc_le64_w16_candidate_visible
```

Reason:

```text
AJ_RD_256 +0.0286,
AJ +0.0810, OA +0.9701,
but positive/negative/zero = 15/5/5.
```

---

## 8. Next step

The next experiment should not be another broad threshold sweep.

Run:

```text
V8-C0.2 fine-risk verifier dataset
```

Purpose:

```text
Learn or audit whether candidate-visible touched frames are truly useful at fine thresholds.
```

Construct per-event/per-frame causal samples for frames touched by the current recovery mode:

```text
state = causal native/candidate/relation features at tau
label_good = candidate improves native over fine thresholds or AJ_RD proxy
label_bad = candidate false-visible or worse than native across thresholds
```

Initial goal:

```text
Determine whether a lightweight verifier can remove camel/shooting/breakdance-style false recoveries without losing major positive videos.
```

If a simple verifier helps, proceed to learned fine-risk CRRP.
If not, keep the simple w8 candidate-visible recovery as the main method and report w16 as high-gain ablation.
