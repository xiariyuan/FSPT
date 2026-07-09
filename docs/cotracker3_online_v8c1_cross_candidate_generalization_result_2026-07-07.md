# CoTracker3 Online V8-C1 Cross-Candidate Generalization Audit

Date: 2026-07-07

Script:

```text
scripts/v8c1_cross_candidate.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c1_cross_candidate_generalization/v8c1_cross_candidate_generalization_report.json
```

---

## 1. Purpose

V8-C0.5 packaged the default method:

```text
CVRRM default = dist_nc_le64_w8_candidate_visible
```

V8-C1 asks whether CVRRM is candidate-source agnostic or mainly dependent on TrackOn2 bridge.

Same strict-causal rule is applied to each candidate source:

```text
native re-entry event proposal
+ native-candidate distance <= 64 px
+ w8 / w16 recovery mode
+ replace only on candidate-visible frames
```

No X/X_aug, no V7 OOF, no oracle labels.

---

## 2. Candidate alignment

Comparable candidate sources:

```text
old_cotracker3_online_bridge:  align_ok = True
old_cotracker3_offline_bridge: align_ok = True
trackon2_dinov3_bridge:        align_ok = True
```

Excluded from direct comparison:

```text
trackon2_dinov3_davis: align_ok = False
max_query_diff ≈ 0.00391 normalized units
max_gt_diff ≈ 0.00392 normalized units
```

Interpretation:

```text
trackon2_dinov3_davis is not directly aligned with the native cache under the strict check.
Only aligned bridge candidates are used for the main comparison.
```

---

## 3. Main results by candidate

### 3.1 TrackOn2 bridge

Standalone:

```text
AJ        +1.8041
OA        +1.2730
AJ_RD     +0.0180
AJ_RD_256 +0.0111
```

CVRRM default:

```text
dist_nc_le64_w8_candidate_visible
AJ        +0.0964
OA        +0.9462
AJ_RD     +0.0153
AJ_RD_256 +0.0274
positive/negative/zero = 16/4/5
```

CVRRM high-gain:

```text
dist_nc_le64_w16_candidate_visible
AJ        +0.0810
OA        +0.9701
AJ_RD     +0.0162
AJ_RD_256 +0.0286
positive/negative/zero = 15/5/5
```

Conclusion:

```text
TrackOn2 bridge remains the only candidate source that reaches the +0.020 AJ_RD_256 target.
```

---

### 3.2 Old CoTracker3 online bridge

Standalone:

```text
AJ        -0.3417
OA        +0.9797
AJ_RD     -0.0048
AJ_RD_256 -0.0087
```

CVRRM default:

```text
dist_nc_le64_w8_candidate_visible
AJ        -0.3241
OA        +0.3956
AJ_RD     +0.0020
AJ_RD_256 +0.0065
positive/negative/zero = 12/2/11
```

CVRRM high-gain:

```text
dist_nc_le64_w16_candidate_visible
AJ        -0.3118
OA        +0.4009
AJ_RD     +0.0027
AJ_RD_256 +0.0070
positive/negative/zero = 12/3/10
```

Conclusion:

```text
CVRRM can extract a small positive re-entry gain even from a weak old online CoTracker candidate, but the gain is far below the target and AJ is negative.
```

---

### 3.3 Old CoTracker3 offline bridge

Standalone:

```text
AJ        -2.5800
OA        -2.6698
AJ_RD     -0.0392
AJ_RD_256 -0.0808
```

CVRRM default:

```text
dist_nc_le64_w8_candidate_visible
AJ        -0.0773
OA        +0.4463
AJ_RD     +0.0027
AJ_RD_256 +0.0039
positive/negative/zero = 8/6/11
```

CVRRM high-gain:

```text
dist_nc_le64_w16_candidate_visible
AJ        -0.0627
OA        +0.4463
AJ_RD     +0.0036
AJ_RD_256 +0.0039
positive/negative/zero = 9/5/11
```

Conclusion:

```text
CVRRM prevents the very weak offline bridge from catastrophically damaging the full trajectory, but the recovered AJ_RD_256 gain is too small.
```

---

## 4. Interpretation

V8-C1 gives a nuanced answer.

What generalizes:

```text
The recovery-mode mechanism itself is not useless on other candidates.
Even weak candidates can give small positive AJ_RD_256 when used selectively instead of standalone.
```

What does not generalize strongly:

```text
The large +0.027 to +0.029 AJ_RD_256 gain depends on TrackOn2 bridge quality.
Old CoTracker candidates do not provide enough useful candidate-visible coordinates.
```

Therefore:

```text
CVRRM is not fully candidate-agnostic.
It is a candidate-aware recovery mode that requires a strong re-entry candidate provider.
```

This is still acceptable if framed honestly:

```text
TrackOn2 is the candidate provider; CVRRM is the causal recovery policy that selectively uses it for CoTracker3 online re-entry failures.
```

But we should not claim broad cross-candidate generalization yet.

---

## 5. Implication for next step

The previous plan said state-level repair should wait until cross-source evidence holds.
V8-C1 shows cross-source evidence is weak across old CoTracker candidates.

Therefore, do **not** immediately proceed to state-level repair as the main next step.

Recommended next step:

```text
V8-C1.1: stronger candidate-source generalization audit.
```

Specifically:

```text
1. Find or generate aligned TAPIR / LocoTrack / TAPNext-style candidate caches if available.
2. If unavailable, run TrackOn2 bridge as the declared candidate provider and proceed with paper framing as TrackOn2-assisted recovery.
3. Only after this, decide whether state-level repair is worth testing.
```

Fallback path:

```text
If no stronger aligned candidate source exists, package the contribution as:
CVRRM with TrackOn2 bridge candidate provider for CoTracker3 online re-entry recovery.
```

---

## 6. Current decision

```text
Keep CVRRM default = dist_nc_le64_w8_candidate_visible with TrackOn2 bridge.
Do not claim candidate-source agnostic generalization.
Do not start state-level repair yet.
Next: search/inspect whether aligned TAPIR/LocoTrack/TAPNext candidate caches exist; otherwise finalize TrackOn2-assisted framing.
```
