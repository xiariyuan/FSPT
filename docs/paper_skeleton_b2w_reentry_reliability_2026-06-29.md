# Paper Skeleton — B2-W Re-entry Reliability — 2026-06-29

## Working Title Options

### Recommended title

**B2-W: Windowed Re-entry Override for Reliable Tracking Any Point**

### Safer / more analysis-oriented titles

1. **Localized Re-entry Override for Reliable Point Tracking**
2. **Re-entry-Aware Local Intervention for Tracking Any Point**
3. **Improving Point Re-detection by Windowed Local Override**
4. **When Re-detection Helps and Hurts: Localized Override for Point Tracking**

Recommended framing:

```text
Not a new universal SOTA tracker.
A re-entry reliability / standard-tracking tradeoff intervention.
```

---

## One-sentence thesis

Strong standard trackers can fail after long occlusion, while re-entry-strong branches can recover reappearing points but damage ordinary tracking if used globally; B2-W activates the re-entry branch only within predicted re-entry windows, improving re-entry reliability while preserving most standard tracking performance.

---

## Abstract Draft

Tracking Any Point methods have improved substantially on standard short- and medium-term tracking benchmarks, yet re-detecting points after occlusion or disappearance remains unreliable. We observe a practical tradeoff: branches or trackers that are strong at re-entry can substantially improve re-detection metrics, but using them globally causes severe degradation in standard tracking accuracy. To address this, we propose **B2-W**, a windowed local override framework that keeps a standard-strong base tracker during normal tracking and activates a re-entry-strong override branch only around predicted re-entry windows. A lightweight persistence guard, **B2-W16-P2**, further reduces false-trigger cost by requiring the override branch to be visible for two consecutive frames before activation.

On DAVIS, global re-entry fusion improves AJ_RD from 0.5546 to 0.6279 but collapses standard AJ from 70.05 to 47.40. B2-W16-P2 retains most of the re-entry gain, reaching 0.6251 AJ_RD while recovering standard AJ to 69.01. On RGB-Stacking fresh videos 20–49, frozen B2-W16-P2 improves AJ_RD over the offline base by +0.0638 while losing only -0.5280 standard AJ, and it outperforms the online override by +0.0333 AJ_RD and +34.37 standard AJ. Per-video and occlusion-length analyses show that gains are concentrated in long-occlusion and re-entry regimes. We further analyze false-trigger costs and show that remaining failures are dominated by a small set of harmful local overrides, suggesting learned window-level verification as a promising future direction.

---

## Contributions

### Contribution 1 — Re-entry / standard tracking tradeoff diagnosis

We identify and quantify a tradeoff in point tracking:

```text
re-entry-strong global fusion/online branches improve AJ_RD,
but severely damage standard AJ when used globally.
```

Evidence:

```text
DAVIS:
offline base:       AJ_RD_256=0.5546, AJ_256=70.0510
global B1:          AJ_RD_256=0.6279, AJ_256=47.4046

RGB fresh20-49:
offline base:       AJ_RD_256=0.3816, AJ_256=79.5944
online override:    AJ_RD_256=0.4121, AJ_256=44.6933
```

### Contribution 2 — B2-W localized/windowed override framework

We propose B2-W:

```text
base = standard-strong tracker
override = re-entry-strong branch
trigger = base invisible run >= 1 and override visible
action = use override only from t-pre to t+W, then return to base
```

B2-W unifies:

```text
B2 full-post = W = infinity
B2-W16      = finite-window local override
B2-W16-P2   = W16 + 2-frame override visibility persistence
```

### Contribution 3 — Cross-domain validation

We validate B2-W16-P2 on:

```text
DAVIS
RGB-Stacking fresh20-49
```

Main RGB result:

```text
B2-W16-P2 vs offline:
AJ_RD_256 +0.0638
AJ_256    -0.5280

B2-W16-P2 vs online:
AJ_RD_256 +0.0333
AJ_256    +34.3731
```

### Contribution 4 — Mechanistic analysis

We provide:

```text
per-video stability
occlusion-length bucket analysis
trigger precision / recall
false-trigger taxonomy
qualitative success and failure cases
```

Most important DAVIS occlusion result:

```text
occ 9-16:  P2-fixed = +0.1010
occ 17-32: P2-fixed = +0.1014
occ 33+:   P2-fixed = +0.1192
```

### Contribution 5 — Reliability verifier exploration

We investigate B2-RV, a possible learned reliability verifier. Development results show runtime features are predictive, but the current track-level verifier loses too much AJ_RD. This supports B2-RV as future work rather than the current main method.

---

# 1. Introduction

## 1.1 Motivation

Tracking Any Point has become a central formulation for dense motion, object interaction, robotics, video editing, and long-term temporal correspondence. Standard trackers perform well when points remain continuously visible or undergo moderate occlusion. However, practical videos often contain longer occlusions, disappearance, reappearance, and re-entry events.

Recent re-detection metrics such as AJ_RD highlight this blind spot: a tracker can achieve strong standard AJ but still fail to re-detect points after long occlusion. This motivates a question:

```text
Can we improve re-entry reliability without sacrificing standard tracking performance?
```

## 1.2 Observed tradeoff

Our experiments reveal that re-entry-strong branches exist, but they cannot be used globally.

Example on DAVIS:

```text
CoTracker3 offline:
AJ_RD_256 = 0.5546
AJ_256    = 70.0510

global B1 vis4_gated288:
AJ_RD_256 = 0.6279
AJ_256    = 47.4046
```

Interpretation:

```text
global re-entry fusion helps re-detection,
but global use damages ordinary tracking.
```

Example on RGB fresh20-49:

```text
CoTracker3 offline:
AJ_RD_256 = 0.3816
AJ_256    = 79.5944

CoTracker3 online:
AJ_RD_256 = 0.4121
AJ_256    = 44.6933
```

Again, online/override branch has re-entry advantage but disastrous standard tracking.

## 1.3 Key idea

Instead of choosing between base and override globally, use them conditionally:

```text
Use base tracker by default.
When the base has been invisible and override becomes visible,
activate override locally for a short window.
Then return control to the base.
```

This gives B2-W:

```text
Predicted Re-entry Windowed Override
```

## 1.4 Summary of results

On DAVIS:

```text
B2-W16-P2:
AJ_RD_256 = 0.6251
AJ_256    = 69.0119
```

Compared with global B1:

```text
similar AJ_RD,
+21.6073 standard AJ.
```

On RGB fresh20-49:

```text
B2-W16-P2:
AJ_RD_256 = 0.4454
AJ_256    = 79.0664
```

Compared with offline:

```text
+0.0638 AJ_RD
-0.5280 AJ
```

Compared with online:

```text
+0.0333 AJ_RD
+34.3731 AJ
```

## 1.5 Introduction paragraph conclusion

The rest of the paper formalizes B2-W, validates it on DAVIS and RGB-Stacking, and analyzes when localized override helps or hurts. We show that the method improves re-entry reliability with a controlled standard-tracking tradeoff, and we identify false-trigger cost as the main remaining limitation.

---

# 2. Related Work

## 2.1 Tracking Any Point

Discuss TAP-Vid, TAPIR, CoTracker, CoTracker3, TAPNext, TAPNext++, TrackOn2, AllTracker.

Need emphasize:

```text
These works usually focus on tracker architecture/training.
Our work studies inference-time re-entry reliability routing.
```

## 2.2 Long-term tracking and re-detection

Discuss:

```text
long occlusion
out-of-view points
reappearance
AJ_RD / re-detection metrics
```

Position:

```text
We focus specifically on when to trust a re-entry branch after base invisibility.
```

## 2.3 Ensembles, routing, and verifier methods

Discuss:

```text
multi-tracker fusion
teacher agreement
pseudo-label verifier
reliability prediction
```

Important distinction:

```text
Verifier-guided pseudo-labeling selects reliable training labels.
B2-W performs inference-time local override around predicted re-entry windows.
```

## 2.4 Difference from SOTA tracker papers

State carefully:

```text
We do not claim a new SOTA tracker architecture.
B2-W is a plug-in intervention framework for re-entry reliability.
```

---

# 3. Method

## 3.1 Problem setting

Given a query point at time `q`, a tracker predicts:

```text
coordinates p_t
visibility v_t
for all frames t
```

We evaluate:

```text
standard AJ
occlusion-aware accuracy
AJ_RD for reappearing points
```

The goal is not merely to maximize AJ_RD, but to improve AJ_RD while preserving standard AJ.

## 3.2 Base and override branches

Define:

```text
base tracker B:
standard-strong, stable for ordinary tracking

override tracker O:
re-entry-strong, can recover after occlusion but may damage ordinary tracking
```

In our experiments:

```text
DAVIS base      = CoTracker3 offline
DAVIS override  = B1 vis4_gated288

RGB base        = CoTracker3 offline
RGB override    = CoTracker3 online
```

Important limitation:

```text
RGB validates the B2-W mechanism, not the full DAVIS 4-teacher branch generalization.
```

## 3.3 Predicted re-entry trigger

For each query track, we monitor the base visibility and override visibility.

Trigger rule:

```text
base invisible run >= k
and override visible
```

Default:

```text
k = 1
```

For P2:

```text
override visible for at least 2 consecutive frames
```

## 3.4 Windowed override

When trigger occurs at frame `t`, B2-W uses override predictions within:

```text
[t - pre, t + W]
```

Then returns to base.

Default robust variant:

```text
pre = 1
W = 16
P2 = override visible persistence of 2 frames
```

So final variant:

```text
B2-W16-P2
```

## 3.5 Why finite window?

Failure analysis shows that full-post override can over-trust a weak override branch after re-entry. This was visible in RGB dev case `rgb_stacking_000008`.

Therefore:

```text
finite window harvests re-entry recovery
without allowing override to pollute the entire future trajectory
```

## 3.6 Why P2?

P2 is a conservative false-trigger-cost reducer.

On DAVIS:

```text
B2-W16 false-trigger tracks = 1183
B2-W16-P2 false-trigger tracks = 1081
```

P2 improves trigger precision:

```text
0.5300 -> 0.5486
```

with small recall reduction:

```text
0.9632 -> 0.9487
```

## 3.7 Algorithm pseudocode

```text
Input:
  base predictions B = (x_B, v_B)
  override predictions O = (x_O, v_O)
  query time q
  window W
  persistence P

Initialize output Y = B
For each track i:
  t = q_i + 1
  while t < T:
    if invisible_run(v_B[i], before=t) >= 1
       and all(v_O[i, t : t+P]) == visible:
          lo = max(0, t - pre)
          hi = min(T, t + W + 1)
          Y[i, lo:hi] = O[i, lo:hi]
          t = hi
    else:
          t = t + 1
Return Y
```

---

# 4. Experiments

## 4.1 Datasets

### DAVIS

Use existing TAP/DAVIS style evaluation. Report:

```text
AJ_RD_256
AJ_256
OA_256
delta_avg_256
```

### RGB-Stacking

Protocol:

```text
dev first-10:          rgb_stacking_000000 ... 000009
consumed diagnostic:   rgb_stacking_000010 ... 000019
fresh validation:      rgb_stacking_000020 ... 000049
```

Main RGB table uses only:

```text
fresh20-49
```

## 4.2 Metrics

Report:

```text
AJ_RD_256: re-entry / re-detection metric
AJ_256: standard tracking metric
OA_256: occlusion accuracy
average_pts_within_thresh / delta_avg_256
```

Important:

```text
AJ_RD and AJ can trade off.
A method improving AJ_RD but destroying AJ is not acceptable.
```

## 4.3 Main DAVIS result

Use Table 1 from:

```text
docs/paper_experiment_tables_consolidation_2026-06-29.md
```

Key text:

```text
global B1 improves AJ_RD but collapses AJ.
B2-W16-P2 maintains most AJ_RD while recovering standard AJ.
```

## 4.4 Main RGB result

Use Table 2 from:

```text
docs/paper_experiment_tables_consolidation_2026-06-29.md
```

Key text:

```text
B2-W16-P2 improves AJ_RD over offline by +0.0638,
while losing only -0.5280 AJ.
It also beats online by +0.0333 AJ_RD and +34.3731 AJ.
```

## 4.5 Per-video stability

RGB fresh20-49:

```text
P2 improves AJ_RD vs offline = 24 / 30
P2 improves AJ_RD by >=0.01 = 23 / 30
P2 AJ drop >2 = 3 / 30
P2 beats online AJ by >=10 = 30 / 30
```

DAVIS:

```text
P2 improves AJ_RD vs fixed = 18 / 30
P2 improves by >=0.01 = 17 / 30
P2 AJ drop >3 = 4 / 30
P2 beats global B1 AJ by >=10 = 26 / 30
```

---

# 5. Ablation Studies

## 5.1 Global vs local override

Main comparison:

```text
global B1: AJ_RD high, AJ low
B2-W: AJ_RD high, AJ recovered
```

## 5.2 Window size

DAVIS finite-window table:

```text
full-post
W16
W32
W64
W16-P2
```

Key conclusion:

```text
finite-window is not RGB-specific;
W16/W64 preserve almost all full-post benefit.
```

## 5.3 Persistence guard P2

Explain:

```text
P2 slightly reduces trigger recall,
improves precision,
reduces false-trigger tracks,
improves standard AJ slightly.
```

DAVIS:

```text
false-trigger tracks: 1183 -> 1081
precision: 0.5300 -> 0.5486
recall: 0.9632 -> 0.9487
```

RGB fresh20-49:

Need use Table 5 consolidated trigger comparison.

## 5.4 B2-RV exploratory verifier

Keep as appendix/future work, not main method.

Key result:

```text
B2-RV can recover standard AJ but loses too much AJ_RD.
```

Aggregate dev:

```text
harm_safe90:
ΔAJ_RD vs P2 = -0.0126
ΔAJ vs P2 = +0.2579
```

Conclusion:

```text
track-level verifier is too coarse;
window-level verifier may be future work.
```

---

# 6. Analysis

## 6.1 Occlusion-length bucket

Use Table 4.

Key text:

```text
B2-W16-P2 improves long occlusion buckets by about +0.10 to +0.12 AJ segment.
```

This supports the claim that B2-W improves the intended re-entry regime.

## 6.2 Trigger and false-trigger taxonomy

Use Table 5.

DAVIS B2-W16-P2:

```text
triggered_tracks = 2395
true_trigger_tracks = 1314
false_trigger_tracks = 1081
missed_reentry_tracks = 71
precision = 0.5486
recall = 0.9487
```

False-trigger cost:

```text
low_cost_rate = 0.5116
harmful_rate = 0.3941
severe_rate = 0.2479
```

Interpretation:

```text
false triggers are common but not uniformly harmful;
remaining standard AJ losses concentrate in harmful false triggers.
```

## 6.3 True trigger local benefit

DAVIS true triggers:

```text
true_delta_post_mean = +0.0251
```

This supports local intervention framing.

---

# 7. Qualitative Figures

Need select actual images from:

```text
outputs/paper_discovery_2026-06-27/teacher_expansion/b2_qualitative_cases/
```

## Figure 1 — Method diagram

Design a simple visual:

```text
base normal tracking
base invisible period
override visible at predicted re-entry
local W-frame override
return to base
```

## Figure 2 — DAVIS success

Use case from:

```text
success_reentry_recovery
```

Show:

```text
GT green
base yellow
global/override cyan
B2 magenta
```

## Figure 3 — B2 avoids global damage

Use case from:

```text
b2_avoids_b1_global_damage
```

Show global branch hurts ordinary tracking while B2 preserves base outside re-entry window.

## Figure 4 — Failure / limitation

Use case from:

```text
harmful_false_trigger
targeted_harmful_false_trigger
```

Purpose:

```text
honest limitation:
false triggers / weak override branch can still damage standard AJ
```

---

# 8. Limitations

## 8.1 Not a new SOTA tracker

B2-W is an intervention/routing framework. It depends on available base and override branches.

## 8.2 Requires a re-entry-strong override

If the override branch is not stronger than base for re-entry, B2-W has limited benefit.

## 8.3 False-trigger cost remains

P2 reduces false-trigger cost but does not eliminate it.

## 8.4 RGB does not prove full DAVIS 4-teacher branch generalization

RGB uses:

```text
base = CoTracker3 offline
override = CoTracker3 online
```

DAVIS uses:

```text
override = B1 vis4_gated288
```

So RGB validates the B2-W mechanism, not full 4-teacher B1 generalization.

## 8.5 B2-RV not mature

Track-level verifier recovers standard AJ but loses too much AJ_RD. Future work should use window-level or soft fallback verification.

---

# 9. Final paper claim boundaries

## Supported claims

```text
1. Global re-entry fusion improves AJ_RD but damages standard tracking.
2. Windowed local override improves re-entry reliability while preserving most standard AJ.
3. B2-W16-P2 works on DAVIS and RGB fresh20-49.
4. Gains concentrate in long-occlusion / re-entry regimes.
5. P2 is a conservative false-trigger-cost reducer.
```

## Unsupported claims

```text
1. New SOTA tracker.
2. Universal improvement in standard AJ.
3. Complete solution to long-term TAP.
4. Full RGB generalization of the DAVIS 4-teacher B1 branch.
5. Learned verifier already improves final method.
```

---

# 10. Next writing tasks

## Immediate next task

Select qualitative images and create a figure manifest:

```text
Figure 2 success case
Figure 3 global-damage avoidance
Figure 4 failure case
```

## After that

Write a full first draft in this order:

```text
1. Abstract
2. Introduction
3. Method
4. Experiments
5. Analysis
6. Limitations
7. Related Work
```

Related Work can be polished later.



---

# Addendum — v2 supplemental strengthening

The paper now has two supplemental strengthening tables:

```text
Table 6: Baseline parity audit for TrackOn2 / TAPNext
Table 7: Plug-in generality on reproduced TrackOn2 first-query/input-resolution protocol
```

Use Table 7 carefully: it shows a small positive B2-W gain on top of TrackOn2 under a parity-valid first-input protocol, but it should not be mixed directly with the main strided-original DAVIS/RGB tables.
