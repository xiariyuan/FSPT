# CoTracker3 Online V8-C1.2 TAPNext++ Metadata-Bridge Candidate Evaluation

Date: 2026-07-07

Script:

```text
scripts/v8c12_tapnextpp_bridge_eval.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c12_tapnextpp_bridge_eval/v8c12_tapnextpp_bridge_eval_report.json
```

---

## 1. Purpose

V8-C1 showed that CVRRM is strong with TrackOn2 bridge but weak with old CoTracker candidates. V8-C1.1 found TAPNext++ caches that are close to alignment but differ by about 1/255 in query/GT coordinates.

V8-C1.2 formalizes the TAPNext++ diagnostic using a native-metadata bridge:

```text
Use TAPNext++ pred_tracks / pred_visibility as candidate outputs.
Use native query_points / gt_tracks / gt_visibility / metadata for evaluation.
```

This is marked as a diagnostic bridge, not a strict raw-cache alignment result.

---

## 2. Online-style TAPNext++ caches

The following online-style TAPNext++ caches give identical or equivalent results:

```text
tapnextpp_davis_baseline_cache.pt
tapnextpp_davis_baseline_w8_cache.pt
tapnextpp_davis_first_input_cache_v4.pt
tapnextpp_davis_first_input_cache_v5_conf.pt
```

Standalone bridge:

```text
AJ        -1.4554
OA        +1.5031
AJ_RD     -0.0052
AJ_RD_256 +0.0035
```

CVRRM default, w8:

```text
dist_nc_le64_w8_candidate_visible
AJ        -0.6909
OA        +0.5097
AJ_RD     +0.0063
AJ_RD_256 +0.0204
positive/negative/zero = 16/6/3
false_visible_rate_on_touched = 0.3043
damage16_rate = 0.0456
```

CVRRM high-gain, w16:

```text
dist_nc_le64_w16_candidate_visible
AJ        -0.9359
OA        +0.4455
AJ_RD     +0.0045
AJ_RD_256 +0.0209
positive/negative/zero = 16/6/3
false_visible_rate_on_touched = 0.2752
damage16_rate = 0.0338
```

Interpretation:

```text
Online-style TAPNext++ candidates cross the +0.020 AJ_RD_256 target under CVRRM.
This supports that CVRRM is not only TrackOn2-specific.
However, overall AJ is negative and risk rates are worse than TrackOn2 bridge.
```

---

## 3. Offline TAPNext++ caches

Offline TAPNext++ candidates are harmful:

```text
tapnextpp_davis_offline_cache.pt:
CVRRM w8 AJ_RD_256 -0.0695, AJ -5.5996

tapnextpp_davis_offline_w8_cache.pt:
CVRRM w8 AJ_RD_256 -0.0616, AJ -5.3874
```

They have high false-visible and damage rates:

```text
false_visible_rate_on_touched ≈ 0.65
damage16_rate ≈ 0.65 - 0.69
```

Conclusion:

```text
Offline TAPNext++ should not be used as a CVRRM candidate source in the current setup.
```

---

## 4. Updated candidate-source ranking

CVRRM default w8 ranking:

```text
TrackOn2 bridge:          AJ_RD_256 +0.0274, AJ +0.0964
TAPNext++ online-style:   AJ_RD_256 +0.0204, AJ -0.6909
old CoTracker online:     AJ_RD_256 +0.0065, AJ -0.3241
old CoTracker offline:    AJ_RD_256 +0.0039, AJ -0.0773
TAPNext++ offline:        negative
```

CVRRM high-gain w16 ranking:

```text
TrackOn2 bridge:          AJ_RD_256 +0.0286, AJ +0.0810
TAPNext++ online-style:   AJ_RD_256 +0.0209, AJ -0.9359
old CoTracker online:     AJ_RD_256 +0.0070, AJ -0.3118
old CoTracker offline:    AJ_RD_256 +0.0039, AJ -0.0627
TAPNext++ offline:        negative
```

---

## 5. Reflection

The previous statement that CVRRM mainly depends on TrackOn2 needs refinement.

More precise conclusion:

```text
CVRRM requires a strong online re-entry candidate provider.
TrackOn2 bridge is currently the best provider.
Online-style TAPNext++ also works at the AJ_RD_256 target level, but with worse AJ and more risk.
Weak or offline candidates do not work.
```

This is a stronger and more defensible claim than either extreme:

```text
Too broad: CVRRM works with any candidate.
Too narrow: CVRRM only works with TrackOn2.
Correct: CVRRM works when candidate-visible frames have sufficient re-entry quality; TrackOn2 bridge is currently the strongest candidate source.
```

---

## 6. Decision

```text
Keep TrackOn2 bridge as the default candidate provider.
Add TAPNext++ online-style metadata-bridge as a cross-candidate diagnostic ablation.
Do not use offline TAPNext++ as a candidate.
```

---

## 7. Next step

The next logical step is not state-level repair yet. First update the final V8-C0.5 table to include the formal TAPNext++ cross-candidate diagnostic row.

Next task:

```text
V8-C1.3 update final package / candidate-source table
```

Then decide whether the paper framing should be:

```text
CVRRM with TrackOn2 as default provider and TAPNext++ as supporting cross-candidate evidence.
```
