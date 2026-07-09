# V9-A0 Addendum: ReEntryBeliefTrack Missing Pieces After Full BeliefTrack Review

Date: 2026-07-08

This addendum revises the previous V9-A0 design after a closer reading of the uploaded BeliefTrack proposal. The key correction is:

```text
Do not frame the A-level model only as "better re-entry recovery".
Frame it as "calibrated occlusion-aware online point tracking", where re-entry is the hardest and most visible test case.
```

---

## 1. Main correction

Previous V9-A0 emphasis:

```text
CVRRM -> learned re-entry controller -> internal candidate head.
```

Revised emphasis:

```text
CVRRM is the empirical hard-rule teacher.
ReEntryBeliefTrack is a calibrated belief tracker:
Track / Coast / Reacquire phases,
uncertainty-aware belief propagation,
multi-hypothesis reacquisition,
identity-aware verification,
safe memory write,
and occlusion-stratified calibration.
```

The model should not compete only on AJ. It must prove that its uncertainty is meaningful during occlusion and re-entry.

---

## 2. Calibration must become a main axis

The uploaded proposal is not just about re-detection. It explicitly argues that point tracking should answer:

```text
Where is the point?
How certain am I?
Why am I uncertain?
Is the reacquired candidate the same physical point?
```

Therefore, V9 should add the following as main evaluation axes:

```text
NLL
ECE
1sigma / 2sigma coverage
risk-coverage
occlusion-conditioned calibration
```

This should not be optional appendix only. It is the differentiator against methods that already handle re-detection.

---

## 3. Re-entry metrics must be operationalized

Current CVRRM evidence uses:

```text
AJ_RD
AJ_RD_256
```

These remain useful, but the model route also needs event-level metrics:

```text
Re-entry success:
After GT becomes visible again, within T=5/10 frames, prediction is within delta and identity_confidence > tau_id.

Identity switch:
Prediction stays near a wrong physical candidate for M consecutive frames with high identity confidence.

False reacquisition:
GT is still invisible / absent, but the model outputs high-confidence reacquisition.

Time-to-reacquire:
Number of frames from GT re-visible to first successful reacquisition.
```

These metrics make the paper less dependent on a single aggregate and more directly tied to the physical failure mode.

---

## 4. Coast uncertainty needs weak but explicit supervision

Critical problem:

```text
During Coast, GT may be invisible, so direct NLL on hidden point position is unavailable or unreliable.
```

The model must not simply inflate uncertainty without supervision.

Adopt two weak-supervision paths:

```text
1. Re-entry retrospective calibration:
   Use pre-occlusion and re-entry anchors to weakly calibrate covariance coverage.
   Do not use pseudo hidden trajectory to supervise the mean mu.

2. Cross-trajectory statistical consistency:
   Bucket successful re-entry cases by occlusion length and match sigma growth to empirical displacement quantiles.
```

Principle:

```text
Pseudo trajectories calibrate coverage, not exact hidden positions.
```

---

## 5. Phase head must be differentiable and stable

Track / Coast / Reacquire must not become another brittle hard-rule module.

Required design:

```text
soft phase weights during training
Gumbel-softmax or continuous relaxation
temperature annealing
two-threshold hysteresis for inference
phase transition smoothness loss
```

Required logging:

```text
phase accuracy
phase switch rate
phase jitter
temperature schedule effect
```

---

## 6. Existence is not visibility

BeliefTrack separates:

```text
visibility
existence
identity_confidence
```

This is conceptually correct, but supervision must be dataset-aware.

Tiered supervision:

```text
Tier 1: explicit out-of-frame / validity / 3D labels.
Tier 2: synthetic augmentations with known absence.
Tier 3: weak rules using visibility + boundary estimates.
```

Do not train existence as a simple inverse of visibility on DAVIS.

---

## 7. Fair calibration baselines are mandatory

If the model claims calibrated uncertainty, compare against:

```text
CoTracker3 + post-hoc calibration
TAPNext / TAPNext++ + post-hoc calibration
Track-On2 + post-hoc calibration
ProTracker + post-hoc calibration if available
```

Fair protocol:

```text
same held-out calibration split
same calibration family
no test leakage
test set disjoint from calibration set
```

---

## 8. Latency and activation cost must be reported

Multi-hypothesis and global re-detection can cause online latency spikes.

Required metrics:

```text
average ms/frame
p95 latency
global re-detection activation rate
latency under K active reacquisitions
memory cost per point
```

The method can improve re-entry but fail as an online tracker if latency is uncontrolled.

---

## 9. Identity memory contamination is a central risk

Wrong memory writes cause self-reinforcing tracking drift.

Required mechanisms:

```text
positive memory write gate
negative memory for known distractors
write only after confidence and temporal stability
do not write during uncertain reacquire
stop-gradient / quarantine for doubtful candidates
```

Required ablations:

```text
without negative memory
without safe write
without temporal stability gate
```

---

## 10. Multi-hypothesis must be trainable

Multi-hypothesis cannot be only a test-time top-k trick.

Trainable formulations:

```text
Mixture Density NLL:
p(y_t | x_t) = sum_i w_i * N(y_t | mu_i, Sigma_i)

Optional DETR-style matching:
K hypotheses matched to true target + dustbin slots.
```

Collapse criterion:

```text
main-to-second hypothesis margin
or entropy below threshold
or temporal stability over several frames
```

---

## 11. Revised immediate next step

Do not jump directly to full ReEntryBeliefTrack.

Next:

```text
V9-A1: controller-only + calibration-aware prototype
```

Inputs:

```text
native state
candidate state
distance / motion / event age
anchor similarity
simple uncertainty proxy
```

Outputs:

```text
phase probability
candidate trust
safe write probability
uncertainty / risk score
```

Evaluation:

```text
standard TAP metrics
AJ_RD / AJ_RD_256
event-level re-entry success
false reacquisition
risk-coverage
latency cost
```

Success criterion:

```text
Must beat CVRRM rule in re-entry metrics without hurting standard metrics.
Must improve at least one calibration / risk-coverage metric.
```

---

## 12. Final revised positioning

The A-level path should not be:

```text
We learned a better gate than CVRRM.
```

It should be:

```text
We convert online TAP into calibrated belief tracking through explicit Track / Coast / Reacquire states.
CVRRM discovered the hard-rule version of this mechanism.
ReEntryBeliefTrack makes it learnable, uncertainty-aware, and identity-safe.
```
