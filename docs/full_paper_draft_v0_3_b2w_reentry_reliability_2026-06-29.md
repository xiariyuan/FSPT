# B2-W: Windowed Re-entry Override for Reliable Tracking Any Point

**Draft v0.3 — 2026-06-29**

> Status: claim-boundary and protocol-consistency pass added; main method remains B2-W16-P2.  
> Main method: **B2-W16-P2**.  
> Main claim: re-entry reliability / standard tracking tradeoff, not new universal SOTA tracker.

---

## Abstract

Re-detecting points after disappearance or long occlusion remains a persistent weakness in Tracking Any Point. We study a practical but underexplored tradeoff: predictions that are strong for re-entry can improve re-detection, yet using them globally may severely degrade ordinary tracking. On DAVIS, for example, a global re-entry fusion branch improves AJ_RD from 0.5546 to 0.6279, but standard AJ drops from 70.05 to 47.40. This suggests that the key problem is not whether a re-entry-strong branch exists, but when it should be trusted.

We propose **B2-W**, a predicted re-entry windowed override framework. B2-W keeps a standard-strong base tracker as the default prediction source and activates a re-entry-strong override branch only inside short, predicted re-entry windows. Our robust variant, **B2-W16-P2**, uses a 16-frame local override window and a two-frame override-visibility persistence guard. This simple routing design preserves the base tracker outside suspected re-entry periods while still exploiting the override branch when the base is likely to have lost the point.

On DAVIS, B2-W16-P2 achieves 0.6251 AJ_RD while recovering standard AJ to 69.01, retaining most of the global branch's re-entry benefit without its standard-tracking collapse. On RGB-Stacking fresh videos 20--49, frozen B2-W16-P2 improves AJ_RD over the offline base by +0.0638 while losing only -0.5280 standard AJ, and it outperforms the online override by +0.0333 AJ_RD and +34.37 standard AJ. Occlusion-length analysis shows that gains concentrate in long-occlusion and re-entry regimes. We further provide trigger and false-trigger analysis, qualitative success and failure cases, a parity audit for TrackOn2/TAPNext artifacts, and a supplementary plug-in experiment showing small positive gains on a reproduced TrackOn2 first-query protocol. These results support B2-W as a focused inference-time intervention for improving re-entry reliability while preserving most standard tracking performance.

---

## 1. Introduction

Tracking Any Point asks a model to estimate the trajectory and visibility of arbitrary query points throughout a video. This formulation has become useful for dense motion understanding, video editing, robotics, interaction analysis, and long-term correspondence. While recent point trackers have improved standard tracking performance, long-term re-detection remains difficult: a tracker may follow visible points accurately but fail to recover them after disappearance, occlusion, or re-entry.

This gap is especially visible when standard tracking metrics are separated from re-detection metrics. Standard AJ rewards ordinary trajectory accuracy over the full video, while AJ_RD focuses on points that reappear after becoming invisible. A tracker can score well on standard AJ yet miss reappearing points; conversely, a branch that is aggressive at re-entry can improve AJ_RD but corrupt ordinary tracking if used everywhere. Therefore, the central question is not merely how to improve re-detection, but how to improve it without sacrificing standard tracking reliability.

Our experiments reveal a strong empirical tradeoff. On DAVIS, a CoTracker3 offline base reaches 0.5546 AJ_RD and 70.05 standard AJ. A global B1 re-entry fusion branch raises AJ_RD to 0.6279, but collapses standard AJ to 47.40. A similar pattern appears on RGB-Stacking fresh videos 20--49: the online branch improves AJ_RD over the offline base, but standard AJ drops from 79.59 to 44.69. These results indicate that re-entry-strong predictions can be valuable, but their global use is unsafe.

We address this tradeoff with **B2-W**, a predicted re-entry windowed override framework. B2-W uses a standard-strong base tracker during normal tracking. It monitors the base visibility and the override visibility, and when the base has been invisible while the override becomes visible, B2-W activates the override only inside a short temporal window. After this window, control returns to the base tracker. The method is deliberately local: it is designed to harvest re-entry recovery while limiting the damage that global override can cause to ordinary tracking.

Our final robust variant is **B2-W16-P2**. The suffix W16 denotes a 16-frame local override window, and P2 denotes a two-frame override-visibility persistence guard. P2 reduces spurious activations by requiring the override branch to remain visible for two consecutive frames before triggering. It is a conservative reliability guard rather than the main source of performance.

The resulting method gives a favorable re-entry / standard-tracking operating point. On DAVIS, B2-W16-P2 achieves 0.6251 AJ_RD and 69.01 standard AJ, closely matching the re-entry gain of global B1 while avoiding its standard-AJ collapse. On RGB-Stacking fresh20-49, B2-W16-P2 improves AJ_RD over the offline base by +0.0638 with only -0.5280 standard AJ loss. Per-video analysis shows AJ_RD improvement on 24/30 fresh RGB videos. Occlusion-length buckets further show that the largest gains occur in long-occlusion regimes, with roughly +0.10 to +0.12 segment-AJ improvements for longer occlusion ranges.

We also analyze when B2-W fails. The main failure mode is false-trigger cost: if the base is still reliable but the override becomes visible, local override can damage standard tracking. P2 reduces false-trigger count and improves trigger precision, but does not eliminate this risk. We explore a learned verifier, B2-RV, and find that runtime features are predictive, but a track-level verifier rejects too much useful re-entry and loses too much AJ_RD. We therefore keep B2-W16-P2 as the main method and treat learned verification as future work.

Finally, we include two safeguards for fair interpretation. First, we do not claim that B2-W is a new universal SOTA tracker. It is an inference-time reliability intervention that depends on compatible base and override branches. Second, we audit TrackOn2/TAPNext artifacts and show that TrackOn2 is strong under a reproduced first-query/input-resolution protocol, but current TrackOn2/TAPNext strided-original bridge caches are not parity-established. We therefore use same-protocol CoTracker3 offline/online and global B1 as the main baselines, and report TrackOn2 plug-in results only as supplementary evidence.

### Contributions

1. **Tradeoff diagnosis.** We quantify a re-entry / standard-tracking tradeoff: re-entry-strong branches improve AJ_RD but can severely damage standard AJ when used globally.
2. **B2-W framework.** We propose predicted re-entry windowed override, an inference-time local intervention that activates a re-entry branch only within predicted temporal windows.
3. **Robust operating point.** We introduce B2-W16-P2, a finite-window variant with a two-frame persistence guard that reduces false-trigger cost while retaining most re-entry gain.
4. **Cross-domain validation.** We validate B2-W16-P2 on DAVIS and RGB-Stacking fresh20-49, with per-video, occlusion-length, and trigger analyses.
5. **Failure and protocol analysis.** We provide qualitative success/failure cases, a false-trigger taxonomy, a TrackOn2/TAPNext baseline parity audit, and a supplementary TrackOn2 plug-in experiment.

---

## 2. Related Work

### Tracking Any Point

Tracking Any Point benchmarks and methods formulate point tracking as predicting coordinates and visibility for arbitrary query points. Recent trackers such as TAPIR, CoTracker/CoTracker3, TAPNext/TAPNext++, TrackOn2, and AllTracker improve different aspects of point correspondence, including dense tracking, online tracking, long-term memory, and training scale. These works primarily focus on tracker architectures, training data, and inference pipelines.

Our work is complementary. We do not propose a new tracker architecture. Instead, we study an inference-time reliability problem: when should a system trust a re-entry-strong branch after the base tracker becomes invisible?

### Long-term Tracking and Re-detection

Long-term point tracking requires recovering points after occlusion, disappearance, or re-entry. Standard AJ can hide this failure mode because a tracker can perform well on continuously visible regions while missing reappearing points. AJ_RD focuses on re-detection quality for reappearing points and exposes this blind spot.

B2-W is designed specifically for this regime. Its goal is not to globally replace the base tracker, but to intervene locally around re-entry events.

### Routing, Fusion, and Reliability Verification

Multi-tracker fusion and verifier-guided reliability methods have been used in tracking and pseudo-label generation. These approaches often choose among multiple predictions or identify reliable labels. B2-W differs in two ways. First, it operates at inference time rather than as a training-label selection mechanism. Second, it uses local temporal windows, returning control to the base tracker after a predicted re-entry window.

We also explore a learned verifier, B2-RV, but our current track-level verifier is not yet strong enough to replace B2-W16-P2. This suggests that future reliability verification should be window-level and optimized directly for the AJ_RD / AJ tradeoff.

---

## 3. Method

### 3.1 Problem Setup

Given a video and a query point at frame `q`, a point tracker predicts a coordinate sequence and a visibility sequence:

```text
p_t = predicted point coordinate at frame t
v_t = predicted visibility at frame t
```

We evaluate two complementary goals. Standard tracking quality is measured by AJ, OA, and average threshold accuracy. Re-entry quality is measured by AJ_RD, which focuses on points that reappear after being invisible. The objective of this work is not to maximize AJ_RD alone; instead, we seek an operating point that improves AJ_RD while preserving standard AJ.

### 3.2 Base and Override Branches

B2-W assumes two prediction sources:

```text
Base tracker B:
  stable and standard-strong during ordinary tracking.

Override branch O:
  useful for re-entry, but potentially harmful if applied globally.
```

The base provides the default prediction. The override is used only when the base is likely to have lost the point and the override branch provides a plausible re-entry prediction. This separation is important: B2-W does not assume the override is globally better than the base. It assumes only that the override may be more useful near re-entry events.

In our main DAVIS experiments, the base is CoTracker3 offline and the override is a global B1 re-entry fusion branch. In RGB-Stacking, the base is CoTracker3 offline and the override is CoTracker3 online. In the supplementary TrackOn2 experiment, we use TrackOn2 as a strong base and CoTracker3 offline as a local override under the parity-valid first-query/input-resolution protocol.

### 3.3 Predicted Re-entry Trigger

B2-W monitors base and override visibility over time. For a query track, a trigger occurs at frame `t` if the base has been invisible immediately before `t` and the override branch is visible at `t`:

```text
invisible_run(v_B, before=t) >= k
and v_O(t) = visible
```

We use `k = 1` by default. This high-recall trigger is intentionally simple: it detects situations in which the base may have lost the point and the override claims to have a visible prediction.

The robust P2 variant adds an override persistence requirement:

```text
v_O(t) = visible and v_O(t+1) = visible
```

This reduces one-frame spurious activations. P2 improves trigger precision and reduces false-trigger count, at the cost of a small recall decrease.

### 3.4 Windowed Local Override

When a trigger occurs at frame `t`, B2-W copies override predictions only inside a finite local window:

```text
[t - pre, t + W]
```

Outside this window, the output remains the base prediction. In the final variant:

```text
pre = 1
W = 16
P = 2 visibility-persistence frames
```

This gives the main method:

```text
B2-W16-P2
```

The finite window is essential. Full-post override can over-trust the override branch after the re-entry moment and propagate errors into later ordinary tracking. A finite window limits this contamination while still allowing the override to recover the point around re-entry.

### 3.5 Algorithm

```text
Input:
  base predictions B = (x_B, v_B)
  override predictions O = (x_O, v_O)
  query time q
  window size W
  pre-window size pre
  persistence P

Initialize output Y = B

For each query track i:
  t = q_i + 1
  while t < T:
    if invisible_run(v_B[i], before=t) >= 1
       and all(v_O[i, t:t+P]) is visible:
          lo = max(0, t - pre)
          hi = min(T, t + W + 1)
          Y[i, lo:hi] = O[i, lo:hi]
          t = hi
    else:
          t = t + 1

Return Y
```

### 3.6 Design Rationale

**Why not global fusion?** Global fusion can improve re-entry, but it damages standard tracking. DAVIS global B1 raises AJ_RD to 0.6279 but drops standard AJ to 47.40. B2-W avoids this by preserving the base outside predicted re-entry windows.

**Why finite windows?** A re-entry branch may be useful near the reappearance moment but unreliable later. Windowed override allows local recovery without committing to the override for the remainder of the video.

**Why P2?** The trigger is high-recall and can produce false positives. Requiring two consecutive visible override frames filters short-lived visibility spikes. On DAVIS, P2 reduces false-trigger tracks from 1183 to 1081 and improves trigger precision from 0.5300 to 0.5486.

**Why not a learned verifier as the main method?** We explored B2-RV and found that runtime features can predict helpful and harmful triggers, but a track-level learned verifier loses too much AJ_RD. This suggests that learned verification is promising but should be window-level or soft-fallback in future work.

---

## 4. Experiments

### 4.1 Datasets and Protocols

We evaluate on DAVIS and RGB-Stacking, and we explicitly separate controlled diagnosis, method development, and frozen validation.

**DAVIS controlled protocol.** We use the unified strided-original protocol. DAVIS is used for controlled diagnosis and main mechanism analysis: CoTracker3 offline provides the standard-strong base, global B1 demonstrates the re-entry / standard-tracking tradeoff, and B2-W variants test whether localized override can recover re-entry without global damage. We do not present DAVIS as a fully untouched validation split; instead, it is the controlled setting where the tradeoff is diagnosed and ablated.

**RGB-Stacking protocol.** RGB-Stacking is used to test whether the B2-W mechanism holds beyond the DAVIS teacher setting. We separate development and validation:

```text
dev first-10:          rgb_stacking_000000 ... 000009
consumed diagnostic:   rgb_stacking_000010 ... 000019
fresh validation:      rgb_stacking_000020 ... 000049
```

The main RGB table uses only fresh20-49. The B2-W16 / B2-W16-P2 choices are frozen before reporting fresh20-49. We do not tune on fresh20-49.

**TrackOn2 supplementary protocol.** TrackOn2 parity is validated under first-query/input-resolution. We therefore report TrackOn2 plug-in results only as a supplementary table and do not mix them with the main strided-original DAVIS/RGB tables.

**B2-WV exploratory protocol.** Learned verifier experiments, including B2-RV, B2-WV-Lite, and P2-safe policies, are development-only. They use DAVIS and RGB dev0-9 and are reported as exploratory evidence, not as main validation results. RGB fresh20-49 is not used for these learned-policy experiments.

### 4.2 Metrics

We report:

```text
AJ_RD_256: re-entry / re-detection accuracy
AJ_256: standard tracking accuracy
OA_256: occlusion accuracy
delta_avg_256: average threshold accuracy
```

Main result tables use query-weighted evaluation from the prediction caches. Per-video counts, bootstrap confidence intervals, and sign tests are explicitly labeled as video-level robustness analysis and are not mixed with the query-weighted main table values. AJ_RD and AJ can trade off. A method that improves AJ_RD but collapses AJ is not considered a good operating point.

---

## 5. Main Results

### 5.1 DAVIS Main Result

| method | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ | role |
|---|---:|---:|---:|---:|---|
| CoTracker3 offline | 0.5546 | 70.0510 | 92.1544 | 82.3206 | standard-strong base |
| global B1 vis4_gated288 | 0.6279 | 47.4046 | 72.6762 | 76.4913 | re-entry strong but globally damaging |
| B2 full-post | 0.6278 | 68.9702 | 91.1851 | 82.4604 | localized trigger, W=∞ |
| B2-W16 | 0.6266 | 68.9512 | 91.1924 | 82.4480 | finite-window local override |
| **B2-W16-P2** | **0.6251** | **69.0119** | **91.2382** | **82.4425** | main robust variant |
| B2 GT-window oracle | 0.6290 | 70.1821 | 92.6439 | 82.3494 | diagnostic upper bound |

The global B1 branch improves re-entry but destroys standard AJ. B2-W16-P2 retains most of the re-entry gain and recovers standard tracking. Compared with global B1, B2-W16-P2 has nearly the same AJ_RD but gains more than 21 standard AJ points.

### 5.2 RGB Fresh20-49 Validation

| method | videos | queries | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 30 | 37221 | 0.3816 | 79.5944 | 91.4636 | 88.4854 |
| CoTracker3 online | 30 | 37221 | 0.4121 | 44.6933 | 55.7585 | 72.8684 |
| B2-W16 | 30 | 37221 | 0.4456 | 79.0634 | 92.8891 | 88.4346 |
| **B2-W16-P2** | 30 | 37221 | **0.4454** | **79.0664** | **92.8804** | **88.4385** |

B2-W16-P2 improves over the offline base by:

```text
AJ_RD_256 +0.0638
AJ_256    -0.5280
```

Compared with the online override:

```text
AJ_RD_256 +0.0333
AJ_256    +34.3731
```

Per-video stability on RGB fresh20-49:

```text
n_videos = 30
p2_improves_AJRD_vs_offline = 24 / 30
p2_improves_AJRD_vs_offline_ge_0p01 = 23 / 30
p2_AJ_drop_vs_offline_le_1 = 20 / 30
p2_AJ_drop_vs_offline_gt_2 = 3 / 30
p2_beats_online_AJRD = 23 / 30
p2_beats_online_AJ_by_10 = 30 / 30
```

This supports the claim that B2-W is not overfit to DAVIS and that the same local override principle works on RGB-Stacking.

---

## 6. Ablations and Analysis

### 6.1 Window and Persistence Ablation

| variant | AJ_RD_256 ↑ | AJ_256 ↑ | ΔAJ_RD vs full-post | ΔAJ vs full-post | interpretation |
|---|---:|---:|---:|---:|---|
| B2 full-post / W=∞ | 0.6278 | 68.9702 | +0.0000 | +0.0000 | W=∞ reference |
| B2-W16 | 0.6266 | 68.9512 | -0.0012 | -0.0190 | finite window |
| B2-W32 | 0.6265 | 68.9596 | -0.0013 | -0.0106 | finite window |
| B2-W64 | 0.6277 | 68.9706 | -0.0001 | +0.0004 | upper finite-window limit |
| **B2-W16-P2** | **0.6251** | **69.0119** | **-0.0027** | **+0.0417** | main robust variant |

Finite-window override preserves most full-post re-entry benefit. P2 sacrifices a small amount of AJ_RD but improves standard AJ and trigger precision.

### 6.2 Occlusion-length Bucket Analysis

| bucket | fixed | global B1 | B2 full-post | B2-W16 | B2-W16-P2 | P2-fixed | events |
|---|---:|---:|---:|---:|---:|---:|---:|
| occ_1_4 | 0.6246 | 0.6624 | 0.6619 | 0.6606 | 0.6590 | +0.0344 | 810 |
| occ_5_8 | 0.6261 | 0.6397 | 0.6396 | 0.6375 | 0.6385 | +0.0124 | 340 |
| occ_9_16 | 0.4500 | 0.5529 | 0.5533 | 0.5517 | 0.5509 | +0.1010 | 351 |
| occ_17_32 | 0.4926 | 0.5980 | 0.5980 | 0.5975 | 0.5940 | +0.1014 | 330 |
| occ_33_plus | 0.2914 | 0.4097 | 0.4097 | 0.4103 | 0.4106 | +0.1192 | 32 |

The largest gains occur in longer occlusion buckets, which supports the intended mechanism: B2-W helps re-entry and long-occlusion recovery rather than merely improving ordinary tracking.

### 6.3 Trigger and False-trigger Analysis

| method | triggered tracks | true-trigger tracks | false-trigger tracks | missed re-entry tracks | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| B2-W16 | 2517 | 1334 | 1183 | 51 | 0.5300 | 0.9632 |
| B2-W16-P2 | 2395 | 1314 | 1081 | 71 | 0.5486 | 0.9487 |

B2-W16-P2 false-trigger cost:

```text
false_triggers = 1081
low_cost_rate = 0.511563
harmful_rate = 0.394080
severe_rate = 0.247919
true_trigger_delta_post_mean = +0.025076
```

P2 reduces trigger count and improves precision, but false-trigger cost remains the main limitation.

---

## 7. Supplementary Baseline and Plug-in Experiments

### 7.1 Baseline Parity Audit for TrackOn2 / TAPNext

| artifact | protocol | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | status |
|---|---|---:|---:|---:|---:|---|
| TrackOn2 first-input bridge | first-query/input-resolution | 0.5444 | 67.0406 | 93.0615 | 79.8418 | parity-valid, strong but different protocol |
| TrackOn2 existing strided-original cache | strided/original | 0.5383 | 37.5303 | 58.3304 | 42.7415 | not main-table-ready |
| TAPNext existing strided-original cache | strided/original | 0.5199 | 34.5573 | 65.9795 | 43.4950 | parity unresolved / not main-table-ready |

TrackOn2 is strong under its reproduced first-query/input-resolution protocol, but current strided-original bridge artifacts are not parity-established. Therefore we do not claim that B2-W generally beats TrackOn2 or TAPNext based on those bridge caches.

### 7.2 Plug-in Generality on Reproduced TrackOn2 Protocol

| method | base | override | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ | trigger precision | trigger recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline first-input | - | - | 0.4525 | 62.1660 | 89.0107 | 77.2244 | - | - |
| TrackOn2 first-input | - | - | 0.5444 | 67.0406 | 93.0615 | 79.8418 | - | - |
| B2-W16 | TrackOn2 | CoTracker3 offline | 0.5513 | 67.1750 | 93.2513 | 79.8543 | 0.7700 | 0.6332 |
| **B2-W16-P2** | **TrackOn2** | **CoTracker3 offline** | **0.5509** | **67.1260** | **93.2224** | **79.8202** | **0.7892** | **0.6216** |
| B2-W16-P2 | CoTracker3 offline | TrackOn2 | 0.5495 | 64.4295 | 92.7140 | 77.8145 | 0.6132 | 0.8263 |

Using TrackOn2 as the strong base, B2-W16-P2 improves TrackOn2 by:

```text
AJ_RD_256 +0.0065
AJ_256    +0.0854
```

This is a small but positive plug-in generality result under a parity-valid TrackOn2 protocol. It should be treated as supplementary evidence, not as a main cross-protocol comparison.

---

## 8. Qualitative Results

We select three main qualitative figures.


### Figure 1: Method Diagram

Generated files:

```text
outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.svg
outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.png
outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.pdf
```

Caption draft: B2-W keeps a standard-strong base tracker by default and activates the re-entry-strong override branch only when the base has been invisible and the override becomes persistently visible. The override is copied only inside a short local window `[t-pre, t+W]`, after which the output returns to the base tracker. This local intervention improves re-entry reliability while avoiding the standard-tracking damage caused by global override.

### Figure 2: Re-entry Recovery

Case:

```text
success_reentry_recovery__bmx-trees__q0.png
fixed AJ = 0.1554
B1 AJ    = 0.7101
B2 AJ    = 0.7076
```

Caption draft: B2-W recovers a point after re-entry. The fixed offline base loses the point after occlusion, while the re-entry branch re-localizes it; B2-W activates the override only around the predicted re-entry window and recovers the trajectory. This illustrates why the method improves AJ_RD.

### Figure 3: Avoiding Global Damage

Case:

```text
b2_avoids_b1_global_damage__drift-straight__q83.png
fixed AJ = 1.0000
global B1 AJ = 0.0105
B2 AJ = 1.0000
```

Caption draft: B2-W avoids the standard-tracking damage caused by global re-entry fusion. The global branch drifts badly on a normally tracked point, whereas B2-W keeps the base prediction because no re-entry trigger is activated. This explains why B2-W preserves standard AJ while global B1 collapses.

### Figure 4: Failure Case

Case:

```text
harmful_false_trigger__dog__q15.png
fixed AJ = 0.9200
B1 AJ    = 0.2963
B2 AJ    = 0.2784
```

Caption draft: A harmful false trigger. The base tracker remains reliable, but the override branch becomes visible and is activated despite no true re-entry event. B2-W therefore inherits the override error and loses standard tracking accuracy. This illustrates the remaining false-trigger-cost limitation.

Color legend:

```text
GT = green
fixed offline = yellow
global B1 / override = cyan
B2 = magenta
```

---

## 9. Exploratory Learned Verification

We explored learned reliability verification to test whether the fixed B2-W16-P2 rule could be replaced by a stronger learned policy. These experiments are development-only and are not promoted to main results.

### 9.1 Track-level B2-RV

B2-RV uses runtime-visible features such as base/override disagreement, visibility persistence, base invisible run length, visibility agreement, and trigger timing.

Feature separability was promising:

```text
harmful_full best single-feature AUC ≈ 0.664
helpful_post best single-feature AUC ≈ 0.789
```

However, cache-level development validation showed that a track-level verifier is too coarse. It recovers standard AJ but loses too much AJ_RD:

```text
Aggregate dev, harm_safe90:
ΔAJ_RD vs B2-W16-P2 = -0.0126
ΔAJ vs B2-W16-P2    = +0.2579
```

Therefore B2-RV is not used as the main method.

### 9.2 Window-level B2-WV

We further tested B2-WV, a counterfactual-supervised window-level verifier. A GT best-action oracle shows substantial diagnostic headroom on DAVIS:

```text
DAVIS B2-WV oracle λ=1 vs B2-W16-P2:
ΔAJ_RD_256 = +0.0204
ΔAJ_256    = +2.1043
```

This oracle is not deployable because it uses counterfactual labels. Learned runtime-feature policies were then evaluated with grouped cross-validation. They recovered some standard AJ but lost too much AJ_RD. For example:

```text
B2-WV-Lite preserve95_dyn, DAVIS:
ΔAJ_RD_256 vs B2-W16-P2 = -0.0190
ΔAJ_256 vs B2-W16-P2    = +0.4777

B2-WV P2-safe p2_short_re99, RGB dev10:
ΔAJ_RD_256 vs B2-W16-P2 = -0.0413
ΔAJ_256 vs B2-W16-P2    = +0.0907
```

Thus, runtime-only learned policies are not ready to replace B2-W16-P2. The oracle suggests that better verification is possible, but the learned policy likely requires appearance-level evidence to decide whether the override has re-localized the same physical point.

---

## 10. Limitations

1. **Not a new tracker architecture.** B2-W is an inference-time intervention framework. It depends on the availability of a useful base and override branch.
2. **False triggers remain.** P2 reduces false triggers but does not eliminate them. Harmful false triggers still cause standard AJ loss.
3. **A real AJ tradeoff remains.** Bootstrap analysis confirms that the standard-AJ cost versus the offline base is measurable, even though it is much smaller than the collapse caused by global/online override.
4. **Override quality matters.** If the override branch is not better than the base in re-entry windows, B2-W may not help.
5. **Protocol specificity.** RGB validates the B2-W mechanism, but does not prove full DAVIS 4-teacher B1 branch generalization. TrackOn2 plug-in evidence is under first-query/input-resolution, not the main strided-original protocol.
6. **Learned verification is not mature.** B2-RV and B2-WV show oracle headroom but current runtime-feature policies lose too much AJ_RD. They are exploratory, not main methods.
7. **Appearance verification is future work.** The current prediction caches do not contain video frames. Appearance-grounded verification would require additional frame loading and feature extraction, and has not yet been validated as a main result.

---

## 11. Conclusion

We studied the re-entry reliability problem in Tracking Any Point and identified a practical tradeoff: re-entry-strong branches improve re-detection metrics but can damage standard tracking when used globally. We proposed B2-W, a predicted re-entry windowed override framework that uses a standard-strong base tracker by default and activates a re-entry-strong override only within local windows. The robust B2-W16-P2 variant improves AJ_RD on DAVIS and RGB-Stacking while preserving most standard AJ.

Our analysis shows that gains are concentrated in long-occlusion and re-entry regimes, while remaining failures are dominated by false-trigger cost. Supplementary TrackOn2 experiments provide small positive plug-in evidence under a reproduced strong baseline protocol. Overall, B2-W offers a simple and effective inference-time intervention for improving re-entry reliability without the severe standard-tracking degradation of global re-entry fusion.

---

## Appendix Notes / TODO

### Paper figures

```text
Figure 1: method diagram generated at `outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.svg` / `outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.png` / `outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.pdf`.
Figure 2: use bmx-trees re-entry recovery.
Figure 3: use drift-straight global-damage avoidance.
Figure 4: use dog false-trigger failure.
```

### Tables to keep in main paper

```text
Table 1: DAVIS main result
Table 2: RGB fresh20-49 validation
Table 3: window/persistence ablation
Table 4: occlusion-length buckets
Table 5: trigger/false-trigger analysis
```

### Tables to place in appendix

```text
Table 6: TrackOn2/TAPNext baseline parity audit
Table 7: TrackOn2 plug-in generality
B2-RV exploratory verifier table
```

### Claim boundaries

Supported:

```text
B2-W improves re-entry reliability while preserving most standard tracking.
B2-W16-P2 works on DAVIS controlled analysis and RGB fresh20-49 frozen validation.
Gains concentrate in long-occlusion/re-entry regimes.
Per-video bootstrap confirms robust AJ_RD gains on DAVIS and RGB fresh20-49.
B2-W gives a small positive aggregate plug-in gain on reproduced TrackOn2 first-input protocol.
B2-WV oracle shows diagnostic headroom for future learned verification.
```

Unsupported:

```text
B2-W is a new universal SOTA tracker.
B2-W improves all standard tracking metrics.
B2-W generally beats TrackOn2/TAPNext.
B2-WV oracle is a deployable method result.
B2-RV or B2-WV learned policy is ready to replace B2-W16-P2.
Appearance verification is already validated as a contribution.
```


---

## Appendix: Statistical Robustness Audit

We additionally perform a per-video paired bootstrap/sign-test analysis using existing per-video metrics. No additional model inference is used. The audit confirms that the main re-entry gains are robust across videos.

Key results:

```text
DAVIS B2-W16-P2 vs fixed, AJ_RD_256:
mean delta = +0.0795, 95% bootstrap CI = [+0.0445, +0.1176]
positive videos = 18 / 25 non-empty re-entry videos
sign-test p = 0.0227

RGB fresh20-49 B2-W16-P2 vs offline, AJ_RD_256:
mean delta = +0.0613, 95% bootstrap CI = [+0.0423, +0.0804]
positive videos = 24 / 30
sign-test p = 0.00055

RGB fresh20-49 B2-W16-P2 vs online, AJ_256:
mean delta = +34.3731, 95% bootstrap CI = [+32.9499, +35.7175]
positive videos = 30 / 30
```

The same audit confirms that standard-AJ cost versus the offline base is real, so the paper should maintain the tradeoff framing rather than claim universal standard-AJ improvement. The TrackOn2 plug-in result is positive in aggregate but has confidence intervals crossing zero, so it remains supplementary.
