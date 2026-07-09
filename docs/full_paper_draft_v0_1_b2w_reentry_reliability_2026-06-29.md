# B2-W: Windowed Re-entry Override for Reliable Tracking Any Point

**Draft v0.1 — 2026-06-29**

> Status: complete first draft for internal iteration.  
> Main method: **B2-W16-P2**.  
> Main claim: re-entry reliability / standard tracking tradeoff, not new universal SOTA tracker.

---

## Abstract

Tracking Any Point methods have improved substantially on standard tracking benchmarks, yet re-detecting points after disappearance or long occlusion remains unreliable. We observe a practical tradeoff: branches that are strong at re-entry can improve re-detection metrics, but using them globally can severely damage ordinary tracking. This paper studies that tradeoff and proposes **B2-W**, a windowed local override framework for re-entry reliability. B2-W keeps a standard-strong base tracker during normal tracking and activates a re-entry-strong override branch only around predicted re-entry windows. Our robust variant, **B2-W16-P2**, uses a 16-frame override window and a two-frame override-visibility persistence guard to reduce false-trigger cost.

On DAVIS, a global re-entry fusion branch improves AJ_RD from 0.5546 to 0.6279 but collapses standard AJ from 70.05 to 47.40. B2-W16-P2 retains most of the re-entry gain, reaching 0.6251 AJ_RD while recovering standard AJ to 69.01. On RGB-Stacking fresh videos 20–49, frozen B2-W16-P2 improves AJ_RD over the offline base by +0.0638 while losing only -0.5280 standard AJ; it also outperforms the online override by +0.0333 AJ_RD and +34.37 standard AJ. Occlusion-length analysis shows that the gains are concentrated in long-occlusion and re-entry regimes. We further provide trigger and false-trigger analysis, qualitative success/failure cases, a baseline parity audit for TrackOn2/TAPNext artifacts, and a supplementary plug-in experiment showing small positive gains on top of a reproduced TrackOn2 first-query protocol. These results support B2-W as a focused inference-time intervention for improving re-entry reliability while preserving most standard tracking performance.

---

## 1. Introduction

Tracking Any Point aims to estimate the trajectory and visibility of arbitrary query points across a video. This formulation is useful for dense motion analysis, video editing, robotics, object interaction, and long-term temporal correspondence. Modern point trackers can perform well when points remain visible or undergo moderate occlusion, but re-detecting points after disappearance, long occlusion, or re-entry is still difficult. A tracker may achieve strong standard tracking accuracy while failing to recover reappearing points.

Recent re-detection-oriented evaluation highlights this weakness. In this work we use **AJ_RD** as the main re-entry metric and standard **AJ** as the main ordinary tracking metric. The key challenge is not simply to maximize re-entry accuracy. A useful method should improve re-entry reliability without collapsing normal tracking performance.

Our first observation is that re-entry-strong predictions exist, but they are unsafe when used globally. On DAVIS, a global B1 re-entry fusion branch improves AJ_RD from 0.5546 to 0.6279, but standard AJ drops from 70.05 to 47.40. On RGB-Stacking fresh20-49, the online tracker has higher AJ_RD than the offline base, but its standard AJ collapses from 79.59 to 44.69. These results reveal a clear tradeoff: re-entry-strong branches help after disappearance, but global use can damage ordinary tracking.

This motivates a different strategy. Instead of choosing a single tracker globally, we propose to use a standard-strong base tracker by default and activate a re-entry-strong override only around predicted re-entry windows. The resulting method, **B2-W**, performs localized windowed override. When the base tracker has been invisible and the override branch becomes visible, B2-W copies the override prediction for a short window and then returns to the base tracker. Our final robust variant, **B2-W16-P2**, uses a 16-frame override window and requires the override branch to be visible for two consecutive frames before triggering.

The central thesis of this paper is:

```text
Global re-entry fusion improves re-detection but damages standard tracking.
Predicted localized/windowed override can recover re-entry benefit while preserving most standard tracking performance.
```

We validate this thesis on DAVIS and RGB-Stacking. On DAVIS, B2-W16-P2 achieves 0.6251 AJ_RD and 69.01 standard AJ, closely matching the re-entry performance of global fusion while avoiding its standard-tracking collapse. On RGB fresh20-49, frozen B2-W16-P2 improves AJ_RD over the offline base by +0.0638 while losing only -0.5280 AJ. Per-video analysis shows that 24/30 fresh RGB videos improve AJ_RD over the offline base. On DAVIS, occlusion-length bucket analysis shows gains of roughly +0.10 to +0.12 segment AJ in longer occlusion buckets.

We also analyze limitations. False triggers remain the main failure mode: when the base is still reliable but the override becomes visible, local override can damage standard tracking. P2 reduces false-trigger count and improves trigger precision, but it does not eliminate false-trigger cost. We explore a learned reliability verifier, B2-RV, but find that a track-level verifier recovers standard AJ at the cost of too much AJ_RD. We therefore keep B2-W16-P2 as the main method and treat B2-RV as future work.

### Contributions

1. **Tradeoff diagnosis.** We quantify a re-entry / standard-tracking tradeoff: re-entry-strong branches improve AJ_RD but can severely damage standard AJ when used globally.
2. **B2-W framework.** We propose predicted re-entry windowed override, an inference-time local intervention that activates a re-entry branch only within predicted windows.
3. **Robust variant.** We introduce B2-W16-P2, a finite-window variant with a two-frame persistence guard that reduces false-trigger cost.
4. **Cross-domain validation.** We validate B2-W16-P2 on DAVIS and RGB-Stacking fresh20-49, with per-video and occlusion-length analyses.
5. **Failure and parity analysis.** We provide qualitative cases, trigger/false-trigger taxonomy, a baseline parity audit for TrackOn2/TAPNext artifacts, and a supplementary TrackOn2 plug-in experiment.

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

For a query point at frame `q`, a tracker predicts a trajectory and visibility sequence:

```text
p_t = predicted point coordinates
v_t = predicted visibility
for t = 1 ... T
```

We evaluate both standard tracking and re-entry tracking. Standard tracking is measured by AJ, OA, and average threshold accuracy. Re-entry tracking is measured by AJ_RD, which focuses on reappearing points after disappearance or occlusion.

The desired operating point is a tradeoff:

```text
increase AJ_RD
without collapsing standard AJ
```

### 3.2 Base and Override Branches

B2-W assumes two prediction sources:

```text
Base tracker B:
  standard-strong and stable for ordinary tracking.

Override branch O:
  stronger at re-entry, but potentially harmful if used globally.
```

In our main DAVIS experiments:

```text
base = CoTracker3 offline
override = global B1 vis4_gated288
```

In RGB-Stacking experiments:

```text
base = CoTracker3 offline
override = CoTracker3 online
```

In supplementary TrackOn2 experiments:

```text
base = TrackOn2 first-input bridge
override = CoTracker3 offline first-input bridge
```

The RGB and TrackOn2 experiments validate the B2-W mechanism under additional base/override settings; they do not imply that the DAVIS B1 branch fully generalizes to all RGB or TrackOn2 settings.

### 3.3 Predicted Re-entry Trigger

For each query track, B2-W monitors base visibility and override visibility. A trigger occurs at frame `t` if:

```text
base invisible run before t >= k
and override is visible at t
```

The default is:

```text
k = 1
```

The robust P2 variant additionally requires:

```text
override visible for 2 consecutive frames
```

This reduces short spurious override activations.

### 3.4 Windowed Override

After a trigger at frame `t`, B2-W copies override predictions into the output only within a local window:

```text
[t - pre, t + W]
```

Then it returns to the base tracker. The final robust configuration is:

```text
pre = 1
W = 16
P = 2 override-visible persistence frames
```

We refer to this as:

```text
B2-W16-P2
```

### 3.5 Algorithm

```text
Input:
  base predictions B = (x_B, v_B)
  override predictions O = (x_O, v_O)
  query time q
  window W
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

### 3.6 Why Local Windows?

Global use of a re-entry branch can damage standard tracking. Full-post override also risks over-trusting the override after the re-entry moment. A finite window harvests the recovery benefit while limiting future contamination. DAVIS finite-window ablations show that W16/W32/W64 preserve most full-post AJ_RD while controlling standard AJ.

### 3.7 Why P2?

P2 is a conservative false-trigger-cost reducer. On DAVIS, it reduces false-trigger tracks from 1183 to 1081 and improves trigger precision from 0.5300 to 0.5486, while recall decreases only from 0.9632 to 0.9487. It is not the main performance source; it is a robustness guard.

---

## 4. Experiments

### 4.1 Datasets and Protocols

We evaluate on DAVIS and RGB-Stacking.

**DAVIS main protocol.** We use the unified strided-original protocol. Main methods include CoTracker3 offline, global B1, B2 full-post, B2-W16, B2-W16-P2, and a GT-window oracle.

**RGB-Stacking protocol.** We separate development and validation:

```text
dev first-10:          rgb_stacking_000000 ... 000009
consumed diagnostic:   rgb_stacking_000010 ... 000019
fresh validation:      rgb_stacking_000020 ... 000049
```

The main RGB table uses only fresh20-49.

**TrackOn2 supplementary protocol.** TrackOn2 parity is validated under first-query/input-resolution. We therefore report TrackOn2 plug-in results only as a supplementary table and do not mix them with the main strided-original DAVIS/RGB tables.

### 4.2 Metrics

We report:

```text
AJ_RD_256: re-entry / re-detection accuracy
AJ_256: standard tracking accuracy
OA_256: occlusion accuracy
delta_avg_256: average threshold accuracy
```

AJ_RD and AJ can trade off. A method that improves AJ_RD but collapses AJ is not considered a good operating point.

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

## 9. Exploratory B2-RV Verifier

We explored a learned reliability verifier, B2-RV, using runtime-visible features such as base/override disagreement, visibility persistence, base invisible run length, visibility agreement, and trigger timing.

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

Therefore B2-RV is not used as the main method. Future work should use window-level verification or soft fallback rather than rejecting entire tracks.

---

## 10. Limitations

1. **Not a new tracker architecture.** B2-W is an inference-time intervention framework. It depends on the availability of a useful base and override branch.
2. **False triggers remain.** P2 reduces false triggers but does not eliminate them. Harmful false triggers still cause standard AJ loss.
3. **Override quality matters.** If the override branch is not better than the base in re-entry windows, B2-W may not help.
4. **Protocol specificity.** RGB validates the B2-W mechanism, but does not prove full DAVIS 4-teacher B1 branch generalization. TrackOn2 plug-in evidence is under first-query/input-resolution, not the main strided-original protocol.
5. **B2-RV is not mature.** The current track-level learned verifier loses too much re-entry benefit.

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
B2-W16-P2 works on DAVIS and RGB fresh20-49.
Gains concentrate in long-occlusion/re-entry regimes.
B2-W gives a small positive plug-in gain on reproduced TrackOn2 first-input protocol.
```

Unsupported:

```text
B2-W is a new universal SOTA tracker.
B2-W generally beats TrackOn2/TAPNext.
B2-RV is ready to replace B2-W16-P2.
```
