# Literature Matrix for ReEntry-VisGuard Improvement Paper

Date: 2026-07-02

Purpose: position **ReEntry-VisGuard** as a plug-in re-detection / visibility-recovery improvement method for Tracking Any Point (TAP), not as a new tracker backbone and not as a generic SOTA claim.

## Core paper thesis

```text
ReEntry-VisGuard improves re-detection in Tracking Any Point by preserving reliable base coordinates and locally recovering visibility in predicted re-entry windows.
```

The paper should be framed as an **improvement method**:

```text
inference-time plug-in layer
+ coordinate-preserving visibility recovery
+ AJ_RD improvement under standard-AJ preservation
```

not as:

```text
new global TAP tracker
or universal TAP SOTA claim
```

---

## Reading template

For every related work, evaluate:

```text
1. What TAP problem does it solve?
2. Does it explicitly model visibility / occlusion?
3. Does it explicitly target re-entry / re-detection?
4. Is the improvement from architecture, training, data, memory, or inference-time correction?
5. How does ReEntry-VisGuard differ?
6. What experiment does this paper motivate for us?
```

---

## 1. TAP-Vid: task and metric origin

**Paper:** TAP-Vid: A Benchmark for Tracking Any Point in a Video  
**URL:** https://arxiv.org/abs/2211.03726

### Role in our paper

TAP-Vid is the task/benchmark origin for Tracking Any Point. It formalizes the point-tracking problem and provides the benchmark context for AJ / OA style evaluation.

### What to cite for

- TAP task definition.
- Tracking arbitrary physical surface points over video.
- Benchmark includes real-world annotated tracks and synthetic perfect ground truth.
- TAP output naturally contains coordinate and occlusion/visibility state.

### Relation to ReEntry-VisGuard

TAP-Vid motivates the channel split:

```text
TAP prediction = coordinate channel + visibility / occlusion channel
```

ReEntry-VisGuard exploits this split. Instead of replacing whole trajectories, it preserves base coordinates and repairs only visibility inside predicted re-entry windows.

### Threat / reviewer question

A reviewer may ask why standard AJ is insufficient. Our answer:

```text
standard AJ entangles coordinate error and visibility error;
AJ_RD and channel-wise diagnosis expose re-entry-specific failure modes.
```

---

## 2. TAPIR: matching + temporal refinement route

**Paper:** TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement  
**URL:** https://arxiv.org/abs/2306.08637

### Role in our paper

TAPIR is a strong classic TAP method. Its improvement route is per-frame matching followed by temporal refinement.

### What to cite for

- Strong point-tracking baseline.
- Per-frame candidate matching + local correlation refinement.
- Large improvement on TAP-Vid benchmarks.

### Relation to ReEntry-VisGuard

TAPIR-style methods improve the **coordinate / correspondence** side. ReEntry-VisGuard targets a complementary channel:

```text
visible-state recovery after re-entry
```

### Experiment motivated

If feasible, export TAPIR / BootsTAP-style predictions as a second base or visibility source:

```text
base = TAPIR/BootsTAP
visibility source = CoTracker online or TAPIR visibility
metric = AJ_RD_256 / AJ_256 / OA_256
```

---

## 3. BootsTAP: real-data self-training route

**Paper:** BootsTAP: Bootstrapped Training for Tracking-Any-Point  
**URL:** https://arxiv.org/abs/2402.00847

### Role in our paper

BootsTAP is a training/data scaling route: self-supervised student-teacher training on real unlabeled video improves TAP performance.

### Relation to ReEntry-VisGuard

BootsTAP improves the tracker by training. ReEntry-VisGuard improves re-detection at inference:

```text
BootsTAP = training-time improvement
ReEntry-VisGuard = inference-time plug-in improvement
```

### Claim boundary

We should not claim to replace training/data scaling. Instead:

```text
ReEntry-VisGuard is orthogonal to training/data improvements and could be applied to trackers trained with stronger data.
```

---

## 4. CoTracker3: current validated base/override family

**Paper:** CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos  
**URL:** https://arxiv.org/abs/2410.11831

### Role in our paper

CoTracker3 is our currently validated tracker family. It provides online/offline variants, and our main caches show strong complementarity:

```text
offline/base: strong coordinates, conservative visibility
online/override: responsive visibility, global AJ damage
```

### What to cite for

- CoTracker3 uses pseudo-labeled real videos to reduce synthetic-to-real gap.
- It has online and offline variants.
- It tracks visible and occluded points.

### Relation to ReEntry-VisGuard

Our main method extracts channel-wise complementarity from CoTracker3 variants:

```text
coordinates = offline/base
visibility = local online/override in predicted re-entry windows
```

### Risk

The paper could be criticized as CoTracker-specific. Therefore, the improvement-paper version needs at least one cross-source or cross-base smoke:

```text
LocoTrack / TAPIR / TAPNext++ / TrackOn2 visibility or base
```

---

## 5. LocoTrack: local 4D correlation / matching route

**Paper:** Local All-Pair Correspondence for Point Tracking  
**URL:** https://arxiv.org/abs/2407.15420

### Role in our paper

LocoTrack is a strong matching/correlation route. It uses local 4D correlation to reduce ambiguity in homogeneous or repetitive regions and integrates long-term temporal information.

### Why it matters most for next experiments

LocoTrack is the best first cross-tracker target because it is:

```text
strong
non-CoTracker
TAP-specific
matching-oriented rather than visibility-calibration-oriented
```

### Experiment priority

Highest-priority generality smoke:

```text
1. Export LocoTrack cache on RGB dev0 1-video.
2. Validate coordinate convention and visibility convention.
3. Evaluate AJ_RD_256 / AJ_256 / OA_256.
4. Try CoTracker offline coordinates + LocoTrack visibility in predicted re-entry windows.
5. If positive, scale to RGB dev0-9, then fresh20-29 or fresh20-49.
```

### Gate

```text
Proceed if dev0-9 AJ_RD gain >= +0.02 and AJ loss <= 1.
Stop / appendix negative if gain < +0.005 or AJ loss > 2.
```

---

## 6. TAPNext: online recurrent transformer route

**Paper:** TAPNext: Tracking Any Point (TAP) as Next Token Prediction  
**URL:** https://arxiv.org/abs/2504.05579

### Role in our paper

TAPNext represents a pure online sequential masked token decoding route. It reduces tracking-specific heuristics through end-to-end recurrent training.

### Relation to ReEntry-VisGuard

TAPNext improves architecture/training. ReEntry-VisGuard is an inference-time layer:

```text
TAPNext = learn better online tracking dynamics
ReEntry-VisGuard = locally recover visibility state from existing predictions
```

### Use in writing

Use TAPNext to explain why online TAP and low-latency tracking are active directions.

---

## 7. TAPNext++: closest problem framing, AJ_RD and re-detection

**Paper:** TAPNext++: What's Next for Tracking Any Point (TAP)?  
**URL:** https://arxiv.org/abs/2604.10582

### Role in our paper

This is the most important problem-framing citation. TAPNext++ explicitly states that re-detection of reappearing points is a blind spot and introduces AJ_RD.

### What to cite for

- Re-detection / reappearing points are an explicit blind spot in TAP literature.
- AJ_RD is introduced to evaluate reappearing points.
- TAPNext++ improves re-detection through long-sequence training, periodic roll augmentation, and supervising occluded points.

### Relation to ReEntry-VisGuard

This creates the cleanest contrast:

```text
TAPNext++: training-time re-detection improvement
ReEntry-VisGuard: inference-time plug-in visibility recovery
```

### Experiment priority

Second external target after LocoTrack:

```text
1. Check official checkpoint / code availability.
2. Run 1-video parity smoke only.
3. Do not use weak / non-parity adapter results as main baseline.
```

---

## 8. Track-On2: online memory / long-term tracking route

**Paper:** Track-On2: Enhancing Online Point Tracking with Memory  
**URL:** https://arxiv.org/abs/2509.19115

### Role in our paper

Track-On2 is relevant for long-term online tracking, memory, occlusion, and identity consistency.

### What to cite for

- Long-term point tracking requires consistent identity under appearance change, motion, and occlusion.
- Online causal memory is a major direction for handling drift and occlusions.

### Current project status

We have TrackOn2 DINOv3 stress smoke, but current stress results indicate protocol / coordinate / visibility mismatch. Do not use as main strong baseline yet.

### Correct role

```text
Main paper: cite as related long-term online/memory tracker.
Appendix: TrackOn2 feasibility / protocol mismatch diagnostic.
Main table: only if parity is validated.
```

---

## 9. PointOdyssey and SynthVerse: data-scaling route

**PointOdyssey URL:** https://arxiv.org/abs/2307.15055  
**SynthVerse URL:** https://arxiv.org/abs/2602.04441

### Role in our paper

These papers represent large-scale synthetic long-term point-tracking data routes.

### Relation to ReEntry-VisGuard

They improve trackers through data diversity and long videos. ReEntry-VisGuard is orthogonal:

```text
data scaling can make trackers better;
ReEntry-VisGuard can still act as an inference-time visibility recovery layer.
```

---

## 10. OmniMotion and DOT: dense / occlusion-aware long-term tracking context

**OmniMotion URL:** https://arxiv.org/abs/2306.05422  
**DOT URL:** https://arxiv.org/abs/2312.00786

### Role in our paper

Use these as broader context for long-term tracking through occlusion and visibility masks.

### Relation to ReEntry-VisGuard

They show that occlusion / visibility is central in dense and long-term tracking, not merely an auxiliary output.

---

## Related Work organization for paper

Recommended structure:

```text
1. TAP formulation and benchmarks
   TAP-Vid

2. Matching and trajectory refinement
   TAPIR, LocoTrack

3. Training and data scaling
   BootsTAP, CoTracker3, PointOdyssey, SynthVerse

4. Online memory and re-detection
   TAPNext, TAPNext++, Track-On2

5. Our distinction
   inference-time coordinate-preserving visibility recovery
```

---

## Experiments demanded by the literature

To make the improvement paper strong, we need:

```text
A. Channel-wise factorial intervention
   base_coord/base_vis
   override_coord/override_vis
   override_coord/base_vis
   base_coord/override_vis
   base_coord/local_override_vis = ReEntry-VisGuard

B. No-leak / sanity audit
   coord1/2/4/8/16
   coordinate perturbation
   visibility shuffle
   cache alignment
   qualitative panels

C. Cross-source / cross-base smoke
   first target: LocoTrack
   second target: TAPNext++
   TrackOn2 only after parity repair
```

---

## Final positioning sentence

```text
Existing TAP improvements primarily come from stronger matching, recurrent memory, training recipes, or larger data. ReEntry-VisGuard instead shows that a substantial part of re-detection loss can be recovered at inference time by decomposing TAP outputs into coordinate and visibility channels and locally calibrating only the visibility state in predicted re-entry windows.
```
