# V9-A0 ReEntryBeliefTrack Design Spec

Date: 2026-07-08

Source inputs:

```text
1. Current CVRRM experimental evidence on CoTracker3 online DAVIS.
2. User-provided ExportBlock TAP paper survey and BeliefTrack proposal.
3. V8-C4 final paper-style story packaging.
```

---

## 1. Strategic decision

CVRRM should not be the final A-level model contribution.

Current CVRRM is best understood as:

```text
1. an empirical discovery;
2. a strong rule baseline;
3. a teacher for model distillation;
4. a hard-coded special case of a richer belief tracker.
```

For a stronger paper, the next-stage idea should be:

```text
ReEntryBeliefTrack:
A causal online point tracker that explicitly maintains point belief states and performs Track / Coast / Reacquire reasoning.
```

Core shift:

```text
From: external candidate + hard output rule
To: internal belief state + re-entry candidate generation + calibrated trust / memory update
```

---

## 2. Why CVRRM alone is not enough for A-level ambition

CVRRM is clean and effective, but as a top-conference model contribution it has limits:

```text
1. It is rule-based and output-level.
2. It depends on TrackOn2 bridge as a strong external candidate provider.
3. It is not a new tracker backbone.
4. It does not learn candidate generation.
5. It does not maintain calibrated uncertainty.
6. It does not explicitly solve identity under long occlusion.
```

Therefore, CVRRM should be retained as the minimal causal principle, not as the final model.

---

## 3. What to absorb from the uploaded BeliefTrack idea

The uploaded proposal is valuable because it turns re-entry tracking into a belief filtering problem.

Most important components to absorb:

```text
1. Track / Coast / Reacquire phase state.
2. Per-point belief state instead of single coordinate.
3. Uncertainty-aware prior / candidate consistency.
4. Multi-hypothesis reacquisition during long occlusion.
5. Identity memory and safe memory write.
6. Calibration-oriented evaluation under occlusion / re-entry buckets.
```

Components to defer:

```text
1. full covariance as default;
2. downstream BA / SfM;
3. full retrospective calibration system;
4. complete risk-coverage benchmark;
5. large-scale all-dataset end-to-end training from day one;
6. complex existence modeling on datasets without native out-of-frame labels.
```

Reason:

```text
First prove that a small belief-model prototype can beat CVRRM.
Only then expand calibration/downstream/multi-dataset claims.
```

---

## 4. Final model direction

Working name:

```text
ReEntryBeliefTrack
```

Long title:

```text
Causal Belief-Guided Reacquisition for Online Point Tracking
```

Main hypothesis:

```text
Online TAP re-entry failures are best handled by explicit belief states:
a point should maintain position, uncertainty, phase, identity anchors, and re-acquisition hypotheses.
```

One-sentence model claim:

```text
ReEntryBeliefTrack converts CVRRM's hard rules into a learnable causal belief-state tracker with Track / Coast / Reacquire phases, uncertainty-aware candidate consistency, and identity-aware multi-hypothesis reacquisition.
```

---

## 5. Mapping from CVRRM rules to model structure

| CVRRM rule | ReEntryBeliefTrack structure |
| --- | --- |
| native low / invisible trigger | learned phase head: Track / Coast / Reacquire |
| distance <= 64 px | uncertainty-aware Mahalanobis / learned consistency gate |
| W=8 recovery window | finite recovery option / learned phase duration |
| candidate_visible | observation likelihood / candidate trust |
| TrackOn2 candidate | internal re-detection candidate head |
| hard output replacement | posterior belief update / soft candidate-native fusion |
| no state writeback | safe memory-write gate |

This mapping is the key intellectual bridge. CVRRM becomes a hard-coded instance of the broader belief tracker.

---

## 6. Minimal belief state

Do not start with the full final mixture state. Use a progressive design.

### 6.1 MVP belief state

```text
b_t = {
  μ_t: current point mean coordinate,
  σ_t: diagonal position uncertainty,
  phase_t: Track / Coast / Reacquire,
  h_t: identity / anchor memory,
  c_t: candidate set,
  q_t: confidence / trust state
}
```

### 6.2 Later multi-hypothesis state

```text
belief_t = { (w_i, μ_i, Σ_i, e_i, existence_i, identity_i) }_{i=1..K}

Track phase: K = 1
Coast / Reacquire phase: K = 4~8
```

MVP should use diagonal uncertainty and small K. Full covariance is optional.

---

## 7. Model modules

### 7.1 Native Tracking Stream

Purpose:

```text
Handle ordinary visible tracking and short-term motion.
```

Input:

```text
current frame feature
query token
previous coordinate / memory
previous visibility / confidence
```

Output:

```text
native_coord_t
native_vis_t
native_conf_t
native_feature_t
```

This stream replaces the role of CoTracker3 native in the long-term model.

---

### 7.2 Belief Prediction / Coast Module

Purpose:

```text
Propagate point state through occlusion without hallucinating high-confidence observations.
```

Prediction:

```text
μ_t^- , σ_t^- = f(μ_{t-1}, σ_{t-1}, velocity, phase_{t-1}, context)
```

Behavior:

```text
Track phase: uncertainty stays small or shrinks.
Coast phase: uncertainty expands according to learned dynamics.
Reacquire phase: uncertainty interacts with candidate hypotheses.
```

This turns fixed `dist<=64` into uncertainty-aware candidate consistency.

---

### 7.3 Phase Head

States:

```text
Track: normal visible tracking.
Coast: native observation unreliable; propagate belief only.
Reacquire: candidate observations exist; verify identity and trust.
```

Prediction:

```text
p_track, p_coast, p_reacquire = PhaseHead(native_state, belief_state, observation_features)
```

Training:

```text
1. CVRRM pseudo labels for initial supervision.
2. GT-derived occlusion/re-entry labels where available.
3. Hysteresis or temporal smoothness to reduce phase jitter.
```

---

### 7.4 Anchor Memory

Purpose:

```text
Preserve point identity across long occlusion.
```

Stored anchors:

```text
query-frame anchor
last reliable visible anchor
pre-occlusion anchor
recent reliable anchor queue
optional negative memory
```

Each anchor stores:

```text
feature descriptor
coordinate
time index
confidence
phase tag
```

Memory write rule:

```text
Only write when safe memory gate is high.
Do not write low-confidence candidate features.
```

This directly addresses state writeback pollution observed in V8-C2.1.

---

### 7.5 Re-entry Candidate Head

Purpose:

```text
Remove external TrackOn2 dependency by generating candidates internally.
```

Two-stage design:

```text
1. coarse grid / patch candidate classification;
2. local offset refinement.
```

Search mode by phase:

```text
Track: local search.
Short Coast: uncertainty-expanded local search.
Long Coast: global or semi-global top-K re-detection.
Reacquire: multi-hypothesis scoring and temporal evidence accumulation.
```

Output:

```text
candidate_i = {
  coord_i,
  visibility_i,
  feature_i,
  raw_score_i,
  identity_score_i
}
```

---

### 7.6 Identity-Consistency Scorer

Purpose:

```text
Decide whether a candidate is the same physical point.
```

Inputs:

```text
candidate feature
positive anchor features
negative memory features
belief prior μ, σ
motion consistency
visibility evidence
```

Outputs:

```text
identity_confidence_i
candidate_trust_i
false_reacquisition_risk_i
```

This is the learned version of:

```text
candidate_visible + dist<=64
```

---

### 7.7 Multi-Hypothesis Reacquisition

Purpose:

```text
Avoid committing too early after long occlusion.
```

During Reacquire:

```text
keep top-K hypotheses for several frames;
update weights with temporal evidence;
collapse to one hypothesis only when identity/trust is stable.
```

This is essential for A-level novelty. Without multi-hypothesis, the model is only a soft version of CVRRM.

---

### 7.8 Safe Memory Write Gate

Purpose:

```text
Prevent wrong candidate observations from contaminating internal state.
```

Gate:

```text
β_t = p_safe_write(candidate, identity, uncertainty, temporal stability)
```

Write if:

```text
identity high
uncertainty decreasing
candidate stable over frames
false-visible risk low
```

This is motivated by V8-C2.1:

```text
hard state writeback had signal but was mixed and did not beat output-level high-gain.
```

---

## 8. Training plan

### Stage 0: CVRRM teacher labels

Use current CVRRM to generate:

```text
re-entry events
recovery windows
accepted candidate-visible frames
rejected frames
false-visible negatives
```

Use these to initialize:

```text
phase head
candidate trust head
safe write gate
```

---

### Stage 1: Controller-only prototype

Do not yet remove external candidates.

Input:

```text
native state
TrackOn2/TAPNext candidate state
anchor similarity
motion consistency
uncertainty estimate
```

Output:

```text
candidate trust
phase
safe write probability
```

Success criterion:

```text
learned controller must outperform CVRRM rule on AJ_RD_256 without hurting AJ/OA.
```

Stop if:

```text
controller cannot beat CVRRM rule.
```

---

### Stage 2: Uncertainty-aware gate

Replace fixed distance gate with uncertainty-aware gate.

Losses:

```text
position Huber
visibility BCE
Gaussian NLL with diagonal σ
calibration / coverage regularizer
```

Success criterion:

```text
reduces negative cases like shooting / car-shadow / camel
or improves risk-coverage under re-entry.
```

---

### Stage 3: Internal candidate head

Train internal anchor-conditioned candidate head.

Tasks:

```text
coarse heatmap classification
offset regression
candidate visibility
candidate identity score
```

Success criterion:

```text
internal candidate version approaches or exceeds CVRRM + external TrackOn2.
```

This is the point where it becomes a self-contained point tracking model.

---

### Stage 4: End-to-end ReEntryBeliefTrack

Integrate:

```text
native stream
belief prediction
phase head
candidate generation
multi-hypothesis update
safe memory write
```

Evaluate across datasets and re-entry metrics.

---

## 9. Loss design

Minimal losses:

```text
L_pos = Huber(μ_t, gt_t) on visible/evaluable frames
L_vis = BCE(visibility_t, gt_visibility_t)
L_unc = Gaussian NLL with diagonal σ
L_phase = phase classification / soft gating supervision
L_candidate = heatmap CE + offset regression
L_id = contrastive identity loss
L_write = safe memory write supervision
```

Later losses:

```text
multi-hypothesis MDN-NLL
existence BCE where reliable labels exist
retrospective calibration under re-entry
occlusion-conditioned ECE
risk-coverage loss
```

Important caution:

```text
Do not use DAVIS visibility alone to supervise existence as if it means in-frame vs out-of-frame.
```

---

## 10. Evaluation plan

### 10.1 Standard TAP metrics

```text
AJ
OA
delta_avg
delta_4px
```

### 10.2 Re-entry metrics

```text
AJ_RD
AJ_RD_256
re-entry success rate
false reacquisition rate
time-to-reacquire
identity switch rate
repair / damage
```

### 10.3 Calibration metrics

```text
NLL
ECE
coverage at 1σ / 2σ
risk-coverage
occlusion-conditioned calibration
```

### 10.4 Latency metrics

```text
average ms/frame
p95 latency
global re-detection activation rate
latency under K active reacquisitions
```

---

## 11. Baselines required for A-level positioning

Core baselines:

```text
CoTracker3 online
TrackOn2 / Track-On2
TAPNext / TAPNext++
TAPIR / BootsTAPIR
LocoTrack if feasible
CVRRM rule
ReEntryBeliefTrack controller-only
ReEntryBeliefTrack full
```

Calibration/fairness baselines:

```text
TAPNext + post-hoc calibration
ProTracker + post-hoc calibration
CoTracker3 + calibration
```

Reason:

```text
If the model claims calibrated belief, it must beat post-hoc calibration baselines, not only raw baselines.
```

---

## 12. Required datasets

Do not start with all datasets. Use stages.

### Stage A: DAVIS validation

Purpose:

```text
Does model beat CVRRM rule on the dataset where the rule was discovered?
```

### Stage B: Kinetics subset / RGB-Stacking smoke

Purpose:

```text
Does the mechanism generalize beyond DAVIS?
```

### Stage C: PointOdyssey / Dynamic Replica

Purpose:

```text
Long occlusion, coast, and calibration training/evaluation.
```

### Stage D: RoboTAP / full benchmark

Purpose:

```text
Robot/real-world generalization.
```

---

## 13. What makes the idea A-level rather than rule-level

A-level version must prove at least three things:

```text
1. Structure beats hard rule:
   ReEntryBeliefTrack > CVRRM.

2. Internal candidate generation works:
   model no longer relies on external TrackOn2 for candidate tracks.

3. Belief is calibrated and useful:
   uncertainty improves risk-coverage / occlusion-conditioned calibration, not just AJ.
```

If it only matches CVRRM, then the paper should remain a rule-based recovery paper, not an A-level model paper.

---

## 14. Stop criteria

Stop or downgrade model route if:

```text
1. learned controller cannot beat CVRRM rule;
2. uncertainty improves NLL but hurts AJ / AJ_RD badly;
3. internal candidate head is far worse than TrackOn2 candidate;
4. multi-hypothesis increases latency without re-entry gain;
5. gains disappear outside DAVIS;
6. calibration baselines close the gap.
```

---

## 15. Immediate next experiment: V9-A1

Next task:

```text
V9-A1 controller-only prototype
```

Goal:

```text
Train / evaluate a small recovery controller using existing native + external candidate caches.
```

Inputs:

```text
native visibility / score / raw logits
candidate visibility
native-candidate distance
motion disagreement
anchor distance
event age
candidate visible run
optional patch/feature similarity if available
```

Outputs:

```text
trust score
phase label
accept/reject candidate frame
```

Comparison:

```text
CVRRM hard rule vs learned controller.
```

Criterion:

```text
If controller cannot beat CVRRM on DAVIS full30 with video-heldout validation, do not scale to full ReEntryBeliefTrack yet.
```

---

## 16. Final judgement

The uploaded BeliefTrack idea is not a replacement for CVRRM. It is the model-level continuation of CVRRM.

Correct relation:

```text
CVRRM = discovered causal rule / teacher / baseline.
ReEntryBeliefTrack = structural model that internalizes that rule through belief states.
```

The next phase should not continue tuning CVRRM thresholds. It should test whether a learned belief-aware controller can surpass the hard rule.
