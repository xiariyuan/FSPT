# CoTracker3 Online V8-C2.1 Controlled State Writeback Full30 Result

Date: 2026-07-07

Script:

```text
scripts/v8c21_full.py
```

Raw report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c21_full_hook_eval/v8c21_full_report.json
```

---

## 1. Purpose

This experiment answers the narrow decision question:

```text
Should state writeback replace output-level CVRRM as the main method?
```

Only two controlled state-writeback probabilities were tested:

```text
0.80
0.90
```

No broad sweep was performed.

Default recovery source:

```text
CVRRM + TrackOn2 bridge, dist<=64, W=8
```

State hook writes candidate coordinates and controlled visibility/confidence logits into:

```text
online_coords_predicted
online_vis_predicted
online_conf_predicted
```

---

## 2. Sanity note about output-level comparison

The raw V8-C2.1 script output-level row uses the all-events w8 baseline. The state writeback itself uses the correct dist<=64 touch set.

For final interpretation, compare state writeback to the already validated V8-C0.5 default:

```text
CVRRM default: dist_nc_le64_w8_candidate_visible
```

Validated default result:

```text
AJ        +0.0964
OA        +0.9462
AJ_RD     +0.0153
AJ_RD_256 +0.0274
```

High-gain output-level ablation:

```text
CVRRM high-gain: dist_nc_le64_w16_candidate_visible
AJ        +0.0810
OA        +0.9701
AJ_RD     +0.0162
AJ_RD_256 +0.0286
```

---

## 3. Full30 touch stats

```text
selected_events = 1617
events_with_touch = 659
unique_touched_frames_total = 1456
```

These match the validated CVRRM default touched-frame scale.

---

## 4. Full30 state-writeback results

### 4.1 Controlled probability 0.80

```text
AJ        +0.1310
OA        +0.9394
AJ_RD     +0.0151
AJ_RD_256 +0.0275
```

Compared with output-level default w8:

```text
AJ        +0.0346
OA        -0.0068
AJ_RD     -0.0002
AJ_RD_256 +0.0001
```

Compared with output-level high-gain w16:

```text
AJ        +0.0500
OA        -0.0307
AJ_RD     -0.0011
AJ_RD_256 -0.0011
```

Interpretation:

```text
0.80 is stable, but it essentially ties the output-level default and does not beat high-gain w16.
```

---

### 4.2 Controlled probability 0.90

```text
AJ        +0.0401
OA        +0.8578
AJ_RD     +0.0150
AJ_RD_256 +0.0285
```

Compared with output-level default w8:

```text
AJ        -0.0563
OA        -0.0885
AJ_RD     -0.0003
AJ_RD_256 +0.0011
```

Compared with output-level high-gain w16:

```text
AJ        -0.0409
OA        -0.1123
AJ_RD     -0.0012
AJ_RD_256 -0.0001
```

Interpretation:

```text
0.90 nearly matches high-gain w16 on AJ_RD_256, but it does not exceed it and it reduces AJ/OA.
```

---

## 5. Per-video stability vs output-level default

For probability 0.80:

```text
n = 25 re-entry-valid videos
positive = 9
negative = 8
zero = 8
mean state-minus-output AJ_RD_256 = -0.0001
```

For probability 0.90:

```text
n = 25 re-entry-valid videos
positive = 10
negative = 7
zero = 8
mean state-minus-output AJ_RD_256 = +0.0011
```

Largest 0.90 improvements over output-level default:

```text
loading       +0.0162
judo          +0.0120
bike-packing  +0.0099
dogs-jump     +0.0098
parkour       +0.0054
```

Largest 0.90 regressions:

```text
libby       -0.0213
gold-fish   -0.0068
india       -0.0049
car-shadow  -0.0040
soapbox     -0.0039
```

Interpretation:

```text
State writeback has real propagation effects, but the improvement is fragile and video-dependent.
```

---

## 6. Decision

Do **not** promote state writeback as the main method.

Reason:

```text
1. 0.80 only ties output-level default.
2. 0.90 gives only +0.0011 AJ_RD_256 over output-level default.
3. 0.90 does not beat output-level w16 high-gain.
4. AJ/OA are lower than output-level alternatives.
5. Per-video wins/losses are mixed.
```

Therefore the main method remains:

```text
CVRRM output-level default:
TrackOn2 bridge + dist<=64 + W=8
```

High-gain ablation remains:

```text
CVRRM output-level high-gain:
TrackOn2 bridge + dist<=64 + W=16
```

State writeback becomes:

```text
future work / engineering follow-up, not the paper main method
```

---

## 7. Reflection on the original idea

The original idea was not to blindly mutate CoTracker3 internals. The idea was re-entry recovery.

The experiments show:

```text
Output-level CVRRM captures the core re-entry recovery effect cleanly.
State writeback introduces additional propagation but does not provide enough full30 benefit to justify main-method complexity.
```

This means the project should stop state-writeback exploration for the main paper path.

---

## 8. Next step

Next task:

```text
V8-C3 final paper-style packaging
```

Required outputs:

```text
1. Final method description for CVRRM.
2. Final main table.
3. Candidate-source table.
4. Learned verifier ablation summary.
5. State writeback negative/neutral follow-up summary.
6. Clear limitation/future-work section.
```

The paper framing should be:

```text
CVRRM is an output-level, strict-causal, finite-horizon re-entry recovery mode.
It selectively uses strong candidate-visible observations after native failure events.
State writeback is not necessary for the main result.
```
