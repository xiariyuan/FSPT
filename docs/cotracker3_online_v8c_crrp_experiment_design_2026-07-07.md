# CoTracker3 Online V8-C / CRRP-v2 Experiment Design

Date: 2026-07-07  
Project: FSPT / online point tracking re-entry recovery  
Status: **design freeze before V8-C implementation**

---

## 0. Executive summary

The current mainline should move from V7 visibility calibration to V8-C **re-entry coordinate recovery**.

The proposed method is **not** a simple gate and not a `CoTracker3 + TrackOn2` ensemble. The method should be defined as:

```text
CRRP-v2: Observation-Centric Counterfactual Re-entry Recovery Policy
观察驱动的反事实重进入恢复策略
```

Core formulation:

```text
Online point re-entry recovery is modeled as a finite-option, risk-constrained, causal action-selection problem.

Given:
- a native online trajectory N, currently CoTracker3 online,
- a candidate trajectory C, currently TrackOn2 bridge,
- causal history up to current frame t,

the policy chooses one recovery action:
- keep native,
- replace by candidate for a short horizon,
- candidate-visible-only recovery,
- observation-centric bridge,
- conservative fusion,
- dual-hypothesis confirmation,
- optional state writeback after confirmation.
```

The policy is trained from **counterfactual action labels**: for each historical re-entry event, simulate multiple possible recovery actions offline, measure the recovery gain and damage using ground truth, and learn a utility/risk predictor. At inference time, the policy must be **strictly causal**: it can only use current and past native/candidate outputs, never future GT or future candidate stability.

Primary goal:

```text
Show that CRRP-v2 is not redundant:
CRRP-v2 must outperform CoTracker3 online native, V7-B2 visibility-only, TrackOn2 standalone, and strong simple routing baselines.
```

Minimum target:

```text
AJ_RD_256 improvement >= +0.020 over CoTracker3 online native,
with no meaningful AJ/OA collapse.
```

Ideal target:

```text
AJ_RD_256 improvement >= +0.030,
with stable video-heldout performance and evidence of cross-candidate or cross-native generalization.
```

---

## 1. Current evidence and why V8-C is needed

### 1.1 Native reference

Current CoTracker3 online native full30 reference:

```text
AJ        65.2366
OA        90.8186
AJ_RD      0.3534
AJ_RD_256  0.5333
```

Native cache:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt
```

Event feature cache:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz
```

### 1.2 V7 conclusion: visibility-only is too small

Best V7-B2 / damage-ceiling visibility-style policy:

```text
Full30: AJ -0.0524, OA +0.0331, AJ_RD +0.0020, AJ_RD_256 +0.0047.
5-fold mean: AJ -0.0396, OA +0.0606, AJ_RD +0.00170, AJ_RD_256 +0.00406.
```

Conclusion:

```text
V7 is a real positive local basin, but it is too small to be the main paper-level direction.
Visibility opening alone does not explain the large re-entry failure mode.
```

### 1.3 V8-A conclusion: large headroom is coordinate + visibility recovery

V8-A GT oracle results:

```text
GT coord+vis early4 w4:        AJ_RD_256 +0.0597
GT coord+vis early4 w8:        AJ_RD_256 +0.0761
GT coord+vis early8 w8:        AJ_RD_256 +0.0810
GT coord+vis useful_open_t w16: AJ_RD_256 +0.1055
```

Conclusion:

```text
The main re-entry bottleneck is coordinate recovery, not only visibility calibration.
```

### 1.4 V8-B conclusion: a non-GT candidate source exists

Best current non-GT candidate is:

```text
trackon2_dinov3_bridge
```

TrackOn2 standalone relative to native:

```text
AJ        +1.8041
OA        +1.2730
AJ_RD     +0.0180
AJ_RD_256 +0.0111
```

TrackOn2 coordinate recovery with oracle event/window selection:

```text
useful_open_t w16:     AJ_RD_256 +0.0354
useful_open_t segment: AJ_RD_256 +0.0335
early8 w16:            AJ_RD_256 +0.0323
early8 w8:             AJ_RD_256 +0.0264
early4 w8:             AJ_RD_256 +0.0254
```

Deployability caution:

```text
The strongest V8-B rows still use oracle event/window selection or GT-visible forcing.
They are upper bounds, not deployable methods.
```

Candidate-visible rows are more deployable but still oracle-event dependent:

```text
useful_open_t segment + candidate_visible: AJ_RD_256 +0.0188
useful_open_t w16 + candidate_visible:     AJ_RD_256 +0.0181
early8 segment + candidate_visible:        AJ_RD_256 +0.0179
early4 segment + candidate_visible:        AJ_RD_256 +0.0177
```

Conclusion:

```text
There is a real algorithmic gap:
TrackOn2 standalone:              +0.0111 AJ_RD_256
candidate-visible oracle events:   about +0.018
TrackOn2 oracle recovery selector: +0.025 to +0.035
GT coord+vis oracle:              +0.076 to +0.105

V8-C must learn a causal, non-oracle selector/policy that eats part of this gap.
```

---

## 2. What the contribution is and is not

### 2.1 Not the contribution

Do **not** frame the work as:

```text
- a pluggable module,
- CoTracker3 + TrackOn2,
- a confidence gate,
- a tracker ensemble,
- first discovery of re-entry/re-detection,
- first use of candidate trajectories.
```

These claims are weak or false.

### 2.2 Actual contribution candidate

The contribution should be:

```text
We formulate online TAP re-entry recovery as a finite-horizon, risk-constrained, counterfactual recovery-action selection problem.
```

More concretely:

```text
Given a native online trajectory and a candidate trajectory, learn which recovery action should be executed at a re-entry uncertainty point: keep native, switch to candidate for a short horizon, bridge to a candidate observation, conservatively fuse, or enter a confirmation buffer.
```

### 2.3 Why this is different from a gate

A gate asks:

```text
replace or not?
```

CRRP asks:

```text
Which action has the best expected utility and acceptable risk?
- keep native
- recover_w4
- recover_w8
- recover_w16
- bridge
- fuse
- confirm-buffer
- state-writeback-after-confirmation
```

A gate has one threshold. CRRP has:

```text
- state/action features,
- counterfactual utility labels,
- risk labels,
- finite-horizon action semantics,
- causal online execution,
- optional interval/action conflict resolution,
- optional state repair after confirmation.
```

---

## 3. Related-domain ideas to borrow

This section records adjacent-field mechanisms that are useful, but must be converted into simple ablations rather than piled blindly.

### 3.1 OC-SORT: observation-centric bridge

Borrowed insight:

```text
During occlusion, prediction-only state propagation accumulates error.
When a new reliable observation appears, use the observation to correct/bridge the occlusion gap.
```

CRRP conversion:

```text
Observation-Centric Re-entry Bridge:
Use last reliable native anchor and current candidate observation to construct a short recovery bridge, rather than hard replacing all frames by candidate.
```

Potential action:

```text
a_bridge_w4 / a_bridge_w8
```

### 3.2 ByteTrack: low-confidence candidate mining

Borrowed insight:

```text
Low-confidence detections may still correspond to true occluded objects and should not be discarded early.
```

CRRP conversion:

```text
Do not require candidate_visible == True as the only admissibility rule.
Low-confidence candidate points can enter recovery if motion/appearance/anchor consistency is strong.
```

Potential mechanism:

```text
Low-confidence re-entry candidate mining.
```

### 3.3 DeepSORT / BoT-SORT / StrongSORT: motion + appearance + interpolation

Borrowed insight:

```text
Long occlusion association needs motion evidence, appearance evidence, and temporal consistency.
```

CRRP conversion:

```text
Motion-Appearance Re-entry Consistency Score:
- candidate vs pre-occlusion motion direction,
- candidate vs last reliable native anchor distance,
- candidate patch vs query/anchor feature similarity,
- candidate tracklet smoothness,
- native-candidate convergence/divergence.
```

### 3.4 MHT / JPDA / particle filtering: dual-hypothesis buffer

Borrowed insight:

```text
When association is uncertain, keep multiple hypotheses briefly instead of hard-switching immediately.
```

CRRP conversion:

```text
Dual-Hypothesis Re-entry Buffer:
Keep H_native and H_candidate for a short confirmation horizon. Only commit to candidate recovery if candidate wins over several causal frames.
```

Potential action:

```text
a_enter_buffer_w4 / a_enter_buffer_w8
```

### 3.5 Covariance Intersection: conservative fusion under unknown correlation

Borrowed insight:

```text
When two estimators are correlated in unknown ways, naive averaging can be overconfident and wrong.
```

CRRP conversion:

```text
Do not blindly average native and candidate coordinates.
Only perform conservative fusion if both are trusted and their disagreement is small.
If disagreement is large, choose one or buffer, not average.
```

Potential action:

```text
a_fuse_conservative
```

### 3.6 SLAM relocalization / loop closure: reliable anchors

Borrowed insight:

```text
When local tracking is lost, relocalization uses reliable anchors/place recognition to reset the state.
```

CRRP conversion:

```text
Reliable Anchor Memory:
Store query feature, last reliable visible feature, pre-occlusion anchor feature, last reliable coordinate, and pre-occlusion velocity. Candidate re-entry must be consistent with anchors.
```

### 3.7 XMem / VOS memory: hierarchical memory and cautious write

Borrowed insight:

```text
Long video memory should be hierarchical, and memory write should be cautious to avoid contamination.
```

CRRP conversion:

```text
For state-level repair, do not write candidate into CoTracker3 online state immediately.
First require low risk or multi-frame confirmation.
```

---

## 4. Final proposed method: CRRP-v2

### 4.1 Name

```text
CRRP-v2: Observation-Centric Counterfactual Re-entry Recovery Policy
```

Shorter paper name if needed:

```text
CARE-Utility
```

Preferred technical phrasing:

```text
finite-option risk-constrained counterfactual recovery policy
```

### 4.2 Inputs

Native tracker output, currently CoTracker3 online:

```text
N_coord[t, q]
N_vis[t, q]
N_score[t, q]
N_raw_vis[t, q]
N_raw_conf[t, q]
N_history[<=t, q]
```

Candidate tracker output, initially TrackOn2 bridge:

```text
C_coord[t, q]
C_vis[t, q]
C_uncertainty/logits if available
C_history[<=t, q]
```

Optional image/feature access:

```text
query patch feature
last reliable native anchor patch feature
candidate patch feature
DINO/TrackOn/Cotracker feature similarity if cached or cheap enough
```

### 4.3 Causal state features

Feature groups:

#### A. Native failure features

```text
native visibility at t
native score at t
raw visibility logit at t
raw confidence logit at t
low-score run length up to t
time since last reliable native visible frame
native speed history
native acceleration/jitter history
native score slope
```

#### B. Candidate trust features

```text
candidate visibility at t
candidate uncertainty/logit at t if available
candidate low/high confidence run length up to t
candidate speed history
candidate acceleration/jitter history
candidate temporal smoothness up to t
```

#### C. Native-candidate relation features

```text
||C_t - N_t||
||v_C - v_N||
candidate distance to last reliable native anchor
candidate motion consistency with pre-occlusion native velocity
native-candidate convergence/divergence over past k frames
candidate vs native smoothness ratio
```

#### D. Optional appearance/anchor features

```text
candidate patch similarity to query patch
candidate patch similarity to last reliable native anchor patch
candidate patch similarity to pre-occlusion anchor patch
feature consistency slope over past k frames
```

All features used by online policy must be causal:

```text
features may use frames <= t only.
```

### 4.4 Action space

Minimal action space for V8-C1:

```text
a0 = keep_native
a1 = recover_w4
a2 = recover_w8
a3 = recover_w16
a4 = candidate_visible_only_w8
```

Extended action space for V8-C2/C3 ablation:

```text
a5 = bridge_w4
a6 = bridge_w8
a7 = fuse_conservative
a8 = enter_dual_hypothesis_buffer_w4
a9 = enter_dual_hypothesis_buffer_w8
a10 = state_writeback_after_confirmation
```

### 4.5 Online semantics of recover_w

Important:

```text
recover_w8 does not mean seeing future 8 frames.
It means entering a recovery mode for at most 8 future frames.
Every new frame re-evaluates risk and can terminate recovery early.
```

Pseudo-state:

```text
mode[q] in {NativeMode, RecoveryMode, BufferMode}
remaining[q] = number of recovery frames left
```

### 4.6 Utility and risk

For a historical event e and action a:

```text
U(e, a) = recovery_reward(e, a) - damage_penalty(e, a)
R(e, a) = false_visible + coordinate_damage + over_replacement + state_pollution_proxy
```

Initial practical utility formula:

```text
U(e, a)
  = 1.0 * safe8_gain
  + 0.5 * safe16_gain
  + 0.5 * visibility_recovery_gain
  + 0.5 * AJ_RD_proxy_gain
  - 2.0 * false_visible_count
  - 1.5 * coordinate_damage_count
  - 0.1 * action_window_length
```

This formula is not final. It is a starting point. It must be ablated.

Risk labels:

```text
risk_bad = 1 if false_visible_count > 0 or coordinate_damage_count > damage_limit
risk_score = false_visible_count + coordinate_damage_count + over_replacement_penalty
```

### 4.7 Decision rule

Train predictors:

```text
utility_model: f_u(phi(e, a)) -> predicted utility
risk_model:    f_r(phi(e, a)) -> predicted risk / bad probability
```

Inference decision:

```text
choose a* = argmax_a f_u(phi(e, a))
execute a* only if:
  f_u(phi(e, a*)) > tau_u
  f_r(phi(e, a*)) < tau_r
  a* beats keep_native by margin m
otherwise keep native
```

Optional conservative lower-bound rule:

```text
execute only if lower_confidence_bound(utility) > utility_keep + margin.
```

### 4.8 Interval/action conflict handling

If multiple recovery intervals overlap for the same point:

Initial implementation:

```text
greedy non-overlap selection by descending utility/risk-adjusted score.
```

Later implementation:

```text
weighted interval scheduling per query.
```

---

## 5. How to integrate with baseline models

### 5.1 CoTracker3 online as native tracker

Relevant code:

```text
baselines/cotracker/cotracker/models/core/cotracker/cotracker3_online.py
```

Known online state:

```text
online_ind
online_track_feat
online_track_support
online_coords_predicted
online_vis_predicted
online_conf_predicted
```

V8-C integration levels:

#### Level A: output-level recovery

```text
Do not alter CoTracker3 internal online state.
CRRP only modifies output trajectory and visibility.
```

Purpose:

```text
Validate whether recovery action selection has value without state-intervention complexity.
```

#### Level B: state-level recovery

```text
Write confirmed recovery result into:
- online_coords_predicted
- online_vis_predicted
- online_conf_predicted
```

Purpose:

```text
Allow future CoTracker3 windows to inherit repaired state.
```

Risk:

```text
Wrong candidate recovery can contaminate future windows.
Therefore Level B must be delayed until Level A is proven and must use true online replay.
```

### 5.2 TrackOn2 as initial candidate provider

Relevant code:

```text
baselines/track_on/model/trackon.py
baselines/track_on/model/trackon_predictor.py
```

Useful TrackOn2 components:

```text
point_memory
temporal_mask
multiscale_correlation
top-k reranking
prediction_head visibility/uncertainty
memory_update_policy
```

Initial V8-C usage:

```text
Freeze TrackOn2.
Use its coords/visibility/logits as candidate output.
Do not train TrackOn2.
```

### 5.3 TAPIR / LocoTrack as candidate-source generalization

Use as second-stage candidate providers if caches are aligned.

Purpose:

```text
Test whether CRRP learns general native-candidate trust, not TrackOn2-specific confidence bias.
```

### 5.4 TAPNext++ as strong comparator

TAPNext++ is closer to an end-to-end re-detection method. Use it primarily as:

```text
standalone strong baseline / comparator
```

Do not claim superiority without direct results.

---

## 6. Experimental plan

### V8-C0: causal anti-redundancy baseline ladder

Script:

```text
scripts/eval_cotracker3_online_v8c0_causal_recovery_baselines.py
```

Purpose:

```text
Before training a model, determine whether simple causal policies already solve the problem.
```

Inputs:

```text
native cache:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt

event features:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz

TrackOn2 candidate:
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt

V7 benefit/damage OOF:
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/benefit_oof_early4_all_components.npy
outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/damage_oof_bad_topk.npy
```

Policies to evaluate:

```text
1. TrackOn2 standalone
2. candidate_visible_current -> recover_w4/w8/w16
3. native_invisible_current + candidate_visible_current -> recover_w4/w8/w16
4. native_score_low_current + candidate_visible_current -> recover_w4/w8/w16
5. native-candidate distance large + candidate_visible_current -> recover_w4/w8/w16
6. V7 benefit high + candidate_visible_current
7. V7 damage low + candidate_visible_current
8. confidence routing between native and candidate
9. ByteTrack-style low-confidence candidate mining
10. OC-SORT-style observation-centric bridge
11. DeepSORT-style motion consistency filter
12. conservative fusion when native-candidate distance is small
13. dual-hypothesis confirmation with k-frame causal confirmation
```

Metrics:

```text
AJ
OA
AJ_RD
AJ_RD_256
accepted_events
recovery_frames
false_visible_count
coordinate_damage_count
per-video deltas
per-fold deltas
```

Success signal:

```text
If simple causal policies already reach AJ_RD_256 >= +0.020 with safe AJ/OA, then the method can be lightweight.
```

Failure signal:

```text
If all simple policies are close to TrackOn2 standalone or below candidate-visible oracle-ish rows, learned CRRP is needed.
```

### V8-C1: counterfactual recovery dataset

Script:

```text
scripts/build_cotracker3_online_v8c1_counterfactual_recovery_dataset.py
```

Sample unit:

```text
(video_name, query_id, event_t, action)
```

Actions:

```text
keep_native
recover_w4
recover_w8
recover_w16
candidate_visible_only_w8
bridge_w4
bridge_w8
fuse_conservative
buffer_confirm_w4
buffer_confirm_w8
```

Saved arrays:

```text
X_causal_features.npy
A_action_id.npy
Y_utility.npy
Y_risk.npy
Y_safe8_gain.npy
Y_safe16_gain.npy
Y_false_visible.npy
Y_coord_damage.npy
meta.json
feature_names.json
action_names.json
```

Strict rule:

```text
Features must be causal <= t.
Labels may use future GT, because labels represent action outcomes.
```

### V8-C2: train utility/risk policy

Script:

```text
scripts/train_cotracker3_online_v8c2_counterfactual_policy.py
```

Initial models:

```text
ExtraTreesRegressor for utility
ExtraTreesClassifier for risk_bad
RandomForestRegressor ablation
Logistic/linear ranker ablation
```

Do not start with large neural models.

Validation protocol:

```text
video-level LOOV
threshold selected on train videos only
test videos never used for threshold/model selection
```

Ablations:

```text
binary gate
multi-action utility
multi-action utility + risk
multi-action utility + risk + interval selection
multi-action utility + risk + bridge
multi-action utility + risk + dual-hypothesis buffer
```

Required comparison:

```text
CRRP must beat:
- TrackOn2 standalone
- best V8-C0 simple causal routing
- V7-B2 visibility-only
```

### V8-C3: causal online replay

Script:

```text
scripts/eval_cotracker3_online_v8c3_causal_replay.py
```

Replay protocol:

```text
for frame t in video:
    native_t = native output at t
    candidate_t = candidate output at t
    CRRP uses only history <= t
    CRRP emits output_t
```

Report three levels:

```text
1. post-hoc oracle upper bound
2. causal-feature post-hoc policy on cached outputs
3. true causal replay
```

Interpretation:

```text
If 1 is good but 2/3 fail: method relies on oracle/future information.
If 2 is good but 3 fails: state/timing integration problem.
If 3 is good: online method is credible.
```

### V8-C4: state-level repair

Only start after V8-C3 output-level success.

Script:

```text
scripts/eval_cotracker3_online_v8c4_state_repair.py
```

Intervention:

```text
when recovery is confirmed, write repaired coords/vis/conf into CoTracker3 online state.
```

Write targets:

```text
online_coords_predicted
online_vis_predicted
online_conf_predicted
```

Safety rule:

```text
No state writeback after a single weak frame.
Require low predicted risk or dual-hypothesis confirmation.
```

### V8-C5: generalization experiments

Three axes:

#### A. Cross-video

```text
video-level LOOV and 5-fold split
```

#### B. Cross-candidate

```text
train candidate: TrackOn2
candidate test: TAPIR / LocoTrack / old CoTracker bridge if aligned
```

#### C. Cross-native

```text
native train: CoTracker3 online
native test: old CoTracker online / TAPNext-style online output if available
```

#### D. Cross-dataset

If resources allow:

```text
DAVIS -> RGB-stacking / RoboTAP / Kinetics subset
```

---

## 7. Failure criteria

Stop or demote CRRP if:

```text
1. CRRP does not beat TrackOn2 standalone AJ_RD_256 +0.0111.
2. CRRP does not beat best simple causal routing.
3. CRRP only works in post-hoc oracle mode but fails causal replay.
4. CRRP gains vanish in video-heldout split.
5. CRRP gains are due only to TrackOn2-specific confidence calibration.
6. State-level repair improves AJ_RD but damages AJ/OA heavily.
```

Paper direction should continue only if:

```text
AJ_RD_256 >= +0.020 over native,
AJ/OA are stable,
CRRP beats TrackOn2 standalone and simple routing,
video-heldout performance is nontrivial,
and at least one generalization test supports candidate/native robustness.
```

---

## 8. Expected contribution if successful

A successful paper claim:

```text
We show that online TAP re-entry failures can be decomposed into candidate availability and candidate trust/action selection. Existing trackers focus on generating stronger trajectories; CRRP-v2 instead learns a finite-horizon, risk-constrained recovery policy over native and candidate point trajectories. It uses counterfactual action utility labels and causal online execution to selectively recover coordinates and visibility after occlusion.
```

Chinese summary:

```text
我们不是提出一个更强 tracker，也不是简单融合两个 tracker。
我们提出一个重进入恢复策略层：在遮挡后 native online tracker 不可靠、candidate 轨迹存在但也不完全可靠时，学习哪个恢复动作最值得执行，并控制错误替换风险。
```

---

## 9. Immediate next task

Implement first:

```text
V8-C0: scripts/eval_cotracker3_online_v8c0_causal_recovery_baselines.py
```

Reason:

```text
Before training CRRP, we must know whether the idea is redundant with simple causal routing.
```

Primary output:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c0_causal_recovery_baselines/v8c0_causal_baseline_report.json
```

Documentation to create after running:

```text
docs/cotracker3_online_v8c0_causal_baseline_result_2026-07-07.md
```

Decision after V8-C0:

```text
If simple policies reach >= +0.020 AJ_RD_256, simplify the method around the strongest mechanism.
If not, proceed to V8-C1 counterfactual utility dataset and V8-C2 policy training.
```

---

## 10. Preflight correction after causal-feature audit (2026-07-07)

A direct inspection of the current V7-A4 event feature package shows that it contains non-causal fields:

```text
feature_names include dt1, dt2, dt3, dt4 and post_* statistics.
meta_json includes future_safe16, best_safe16_offset, future_gt_visible, etc.
```

Therefore, the following correction is mandatory for V8-C:

```text
Do not use X / X_aug as-is for strict online policy training or V8-C0 causal baselines.
Do not use existing V7 benefit_oof / damage_oof as causal policy scores unless they are retrained from causal-only features.
```

Allowed causal information for V8-C0:

```text
- meta fields that are computable from history at t: low_run, invis_run, last_visible_t, native_visible_t, score_t.
- native record fields at frame <= t: pred_visibility, pred_vis_score, pred_raw_vis_logit, pred_raw_conf_logit, pred_vis_prob_component, pred_conf_prob_component, pred_tracks.
- candidate record fields at frame <= t: pred_visibility, pred_tracks, candidate motion history.
- relation features at frame <= t: native-candidate distance, velocity disagreement from past/current frames, distance to last reliable native anchor.
```

Disallowed for strict V8-C0/C1 policy input:

```text
- y_reentry_early4 / y_reentry_early8 / y_useful_open_t.
- future_safe16 / future_safe8 / best_safe offset fields.
- post_* feature aggregates.
- dt+ future features.
- existing V7 benefit/damage OOF scores trained on non-causal X_aug.
```

Preflight observation:

```text
Among the 1686 current native event proposals, native_visible_t is false for all events by construction, while TrackOn2 candidate_visible at event_t is only about 23.2%.
This means candidate_visible-only policies may have limited recall. V8-C0 should still test them, but should also test low-confidence candidate mining and motion-consistency policies.
```

Revised immediate implementation rule:

```text
V8-C0 must build its causal policy table directly from native/candidate caches and meta history fields, not from X/X_aug.
If V7-style learned benefit/damage scores are needed, create a new causal-only V8-C1 model later.
```
