# CoTracker3 Online V8-C0 Strict-Causal Recovery Baseline Result

Date: 2026-07-07  
Script: `scripts/eval_cotracker3_online_v8c0_causal_recovery_baselines.py`  
Report: `outputs/paper_discovery_2026-07-05/cotracker3_online_v8c0_causal_recovery_baselines/v8c0_causal_baseline_report.json`

---

## 1. Purpose

V8-C0 tests whether the CRRP-v2 idea is redundant with simple causal policies.

Strict causal rule:

```text
Policy selection may only use native/candidate outputs at frame <= t and causal event metadata.
No X/X_aug features.
No V7 benefit/damage OOF scores.
No early4 / early8 / useful labels.
No GT-visible forcing.
```

This is still an output-level cached replay, not yet a state-level true online intervention.

---

## 2. Inputs

Native cache:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt
```

Candidate cache:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
```

Event metadata source:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz
```

Leakage audit:

```text
The event feature package contains leaky future fields such as dt1/dt2/dt3/dt4 and post_*.
V8-C0 does not use X/X_aug or V7 OOF scores.
Only meta history fields and native/candidate cache outputs are used.
```

---

## 3. Baselines

Native CoTracker3 online:

```text
AJ        65.2366
OA        90.8186
AJ_RD      0.3534
AJ_RD_256  0.5333
```

TrackOn2 standalone delta vs native:

```text
AJ        +1.8041
OA        +1.2730
AJ_RD     +0.0180
AJ_RD_256 +0.0111
```

Event proposal summary:

```text
n_events = 1686
native_visible_t_rate = 0.0000
TrackOn2 candidate_visible_t_rate = 0.2319
mean score_t = 0.2718
mean low_run = 18.8778
mean invis_run = 18.8778
mean native-candidate distance at event_t = 16.49 px
```

---

## 4. Key result

The strongest strict-causal core policy is:

```text
all_events_w16_candidate_visible
```

Meaning:

```text
Every native low/invisible event enters a 16-frame recovery mode.
At each frame inside the recovery horizon, copy TrackOn2 coordinates only if TrackOn2 predicts candidate visibility at that frame.
No GT labels or future candidate statistics are used for selection.
```

Delta vs native:

```text
AJ        +0.0753
OA        +0.9598
AJ_RD     +0.0162
AJ_RD_256 +0.0286
```

Damage / recovery stats:

```text
selected_events = 1686
event_count_with_touch = 413
touched_frames = 2019
gt_visible_touched = 1583
gt_invisible_touched = 436
false_visible_rate_on_touched = 0.2159
candidate_safe16_rate_on_touched_gtvis = 0.9387
candidate_safe8_rate_on_touched_gtvis = 0.8509
repair16_count = 132
damage16_count = 27
repair16_rate = 0.0834
damage16_rate = 0.0171
mean_native_err_px_on_touched_gtvis = 8.0660
mean_candidate_err_px_on_touched_gtvis = 5.7340
```

This clears the previous target:

```text
AJ_RD_256 target >= +0.020
observed +0.0286
```

It also beats TrackOn2 standalone on the re-entry-focused AJ_RD_256 delta:

```text
TrackOn2 standalone AJ_RD_256 delta: +0.0111
V8-C0 best AJ_RD_256 delta:          +0.0286
```

However, it does not beat TrackOn2 standalone on overall AJ:

```text
TrackOn2 standalone AJ delta: +1.8041
V8-C0 best AJ delta:          +0.0753
```

So the method is a re-entry recovery layer, not a globally stronger tracker.

---

## 5. Other important rows

```text
all_events_w8_candidate_visible:
AJ +0.0840, OA +0.9242, AJ_RD +0.0153, AJ_RD_256 +0.0274

all_events_w4_candidate_visible:
AJ +0.0403, OA +0.8762, AJ_RD +0.0107, AJ_RD_256 +0.0212

all_events_w8_copy_candidate_visibility:
AJ +0.1559, OA +0.9846, AJ_RD +0.0110, AJ_RD_256 +0.0198

candidate_visible_t_w16_candidate_visible:
AJ +0.1096, OA +0.9594, AJ_RD +0.0094, AJ_RD_256 +0.0193
```

Interpretation:

```text
Selecting only events where candidate is visible at the event frame is worse than letting all native low/invisible events enter recovery mode and checking candidate visibility frame-by-frame.
This supports a ByteTrack-like insight: do not prematurely discard low-confidence or currently invisible candidate opportunities; keep the event alive and let future causal observations confirm recovery.
```

---

## 6. Per-video robustness check

For `all_events_w16_candidate_visible`, among videos with valid AJ_RD_256:

```text
positive videos: 15
negative videos: 5
unchanged videos: 5
mean per-video AJ_RD_256 delta: about +0.0281
```

Worst AJ_RD_256 deltas:

```text
camel          -0.0710
shooting       -0.0533
breakdance     -0.0090
drift-straight -0.0071
car-shadow     -0.0018
```

Best AJ_RD_256 deltas:

```text
car-roundabout +0.2400
motocross-jump +0.1640
parkour        +0.1099
bike-packing   +0.0757
loading        +0.0590
judo           +0.0565
bmx-trees      +0.0554
dogs-jump      +0.0491
```

Important caution:

```text
The policy passes full30 and target threshold, but it is not uniformly robust.
Risk control or learned utility may still be useful to avoid negative videos such as camel/shooting/breakdance.
```

---

## 7. Reflection

### 7.1 Main positive finding

The simplest strong causal policy already reaches the paper-level AJ_RD_256 target:

```text
native low/invisible event proposal + candidate-visible causal recovery mode
```

This means the core mechanism may be simpler than the full CRRP-v2 design.

A strong immediate framing is:

```text
Causal Candidate-Visible Re-entry Recovery Mode
```

or, within CRRP language:

```text
CRRP-V0: event-proposal + candidate-observation-confirmed recovery.
```

### 7.2 Main negative finding

More selective event filtering by current candidate visibility, low_run, invis_run, anchor consistency, or low-confidence mining did not beat the all-events candidate-visible recovery mode in this core run.

This implies:

```text
The main bottleneck may not be event selection; the current native low/invisible event proposal already has enough recall.
The main useful signal is candidate visibility during the future causal recovery horizon.
```

### 7.3 What this means for CRRP

Do not immediately build a complex utility model.

First, we should strengthen and validate the simple causal recovery mode:

```text
- test with online-reproducible event proposal rather than precomputed meta only,
- test per-video / heldout stability,
- test a risk filter to remove negative videos/events,
- test whether w8 is a safer default than w16 despite slightly lower AJ_RD_256.
```

---

## 8. Next step

Immediate next experiment should be:

```text
V8-C0.1: Robustness and risk-control audit for all_events_w8/w16_candidate_visible.
```

Required checks:

```text
1. Rebuild event proposals directly from native cache with explicit causal trigger rules.
2. Verify the result is identical or close to the meta-based event list.
3. Add per-video delta report into the JSON output.
4. Run 5-fold video-heldout style threshold-free validation.
5. Audit negative videos: camel, shooting, breakdance.
6. Test simple risk filters:
   - candidate visible run >= 2,
   - candidate safe motion band,
   - no recovery when candidate/native disagreement is absurd,
   - shorter w8 instead of w16 for risky cases.
```

Only after this should we decide whether V8-C1 learned counterfactual utility is necessary.

Decision:

```text
Do not proceed directly to a complex learned CRRP model.
Promote V8-C0 simple causal recovery as the current main candidate, then run robustness/risk audit.
```
