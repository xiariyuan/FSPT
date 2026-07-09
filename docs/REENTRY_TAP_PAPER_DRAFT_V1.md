# Improving Re-entry Recovery in Tracking Any Point with Selective Local Override

> Draft V1 — 2026-07-01  
> Working title.  
> This draft frames the work as an **indicator-improvement method paper**: improving AJ_RD for re-entry recovery while preserving standard AJ.

---

## Abstract

Tracking Any Point (TAP) models have made strong progress on standard point tracking metrics, but a point that disappears due to occlusion or leaving the frame and later reappears remains difficult to recover reliably. This failure is not always reflected by standard average tracking metrics, yet it is critical for long-horizon point tracking. In this work, we target the re-entry recovery problem in TAP and study the tradeoff between re-entry reliability and standard tracking accuracy. We observe that a re-entry-strong online branch can improve re-detection, but applying it globally severely damages standard tracking performance. To address this, we propose **B2-W16-P2**, a training-free **Selective Local Re-entry Override** module that combines a standard-strong base tracker with a re-entry-strong override branch. The module activates the override branch only inside short predicted re-entry windows, using a persistent visibility trigger and otherwise preserving the base tracker output. On RGB fresh20-49, B2-W16-P2 improves AJ_RD_256 from **0.3816 to 0.4454** while reducing AJ_256 by only **0.528** points. On DAVIS, it improves AJ_RD_256 from **0.5546 to 0.6251** with a **1.039** AJ cost. We further introduce ReEntry-TAP, a controlled stress validation protocol with out-of-frame translate and moving-occluder re-entry, and validate the method on RGB fresh20-49 frozen stress with 30 videos, 70,984 query instances, and 32,676 re-entry events. Under this frozen validation, B2-W16-P2 improves AJ_RD_256 by **+0.0522** on translate stress and **+0.0277** on occluder stress while avoiding the 32--34 point AJ collapse of the global online branch. In addition, we explore **ReEntry-Guard**, an oracle-guided runtime reliability gate trained only on dev stress labels. When frozen on RGB fresh20-49, ReEntry-Guard further improves over B2-W16-P2 on natural, translate, and occluder validation, closing 20--32% of the oracle routing gap while preserving or improving AJ. These results show that selective local override is an effective and extensible way to improve TAP re-entry recovery without sacrificing standard tracking.

---

## 1. Introduction

Tracking Any Point (TAP) requires estimating the trajectory and visibility of arbitrary query points throughout a video. While recent TAP trackers achieve strong performance on standard benchmarks, average tracking metrics can hide an important failure mode: a point may disappear due to occlusion, object motion, or leaving the frame, and then reappear later. In such cases, the tracker must not only continue local motion estimation but also recover the same point identity after a visibility gap. We refer to this as **re-entry recovery**.

Re-entry recovery is important because many real tracking applications require long-term point identity preservation. A tracker that is accurate before occlusion but fails after reappearance is unreliable for long-horizon correspondence, manipulation, egocentric video understanding, and object-centric motion analysis. However, a tracker can still obtain a high standard AJ score if most visible frames are easy and the re-entry segment is a relatively small part of the trajectory. This motivates using **AJ_RD**, a re-entry / re-detection-oriented metric, as the primary metric for this paper.

A straightforward solution is to use a branch that is better at re-detection or online recovery. We find, however, that using such a branch globally creates a severe tradeoff. On RGB fresh20-49, a global online branch improves AJ_RD_256 from 0.3816 to 0.4121, but standard AJ_256 collapses from 79.5944 to 44.6933. In other words, global re-entry strength comes at the cost of destroying standard tracking performance.

This observation leads to our core question:

```text
Can we improve TAP re-entry recovery while preserving standard tracking accuracy?
```

We answer this with **B2-W16-P2**, a simple but effective inference-time module for selective local override. The method uses a standard-strong offline/base tracker for most frames and activates a re-entry-strong online/override branch only when a local re-entry risk is detected. The trigger fires when the base branch has been invisible and the override branch is persistently visible for two consecutive frames. After a trigger, the override branch is used only in a local temporal window of 16 frames, after which the output returns to the base branch.

This design has three important properties. First, it is **local**: it modifies only a short temporal window. Second, it is **preservative**: outside the override window, the output is exactly the base tracker output. Third, it is **selective**: the method trusts the re-entry branch only when the base branch appears to have lost the point and the override branch is stably visible.

Our experiments show that this simple local intervention substantially improves re-entry recovery. On RGB fresh20-49, B2-W16-P2 improves AJ_RD_256 from **0.3816 to 0.4454**, an absolute gain of **+0.0638** and a relative gain of approximately **+16.7%**, while AJ_256 drops by only **0.528** points. Video-level statistics show that the method improves AJ_RD on **24/30** fresh videos, with a mean video-level gain of **+0.0613** and a 95% bootstrap CI of **[0.0419, 0.0805]**. On DAVIS, B2-W16-P2 improves AJ_RD_256 from **0.5546 to 0.6251**, while almost preserving standard AJ.

To verify that these gains are truly related to re-entry and not just average metric fluctuation, we further design **ReEntry-TAP**, a controlled stress validation protocol. ReEntry-TAP contains two stress families: `translate_exit_reenter`, which creates out-of-frame re-entry, and `moving_occluder`, which creates occlusion-induced re-entry. We first study severity curves on RGB dev0-9 and then freeze all construction parameters and validate L16 stress on RGB fresh20-49. The frozen validation covers **30 videos**, **70,984 query instances**, and **32,676 re-entry events**. B2-W16-P2 improves AJ_RD under both stress families and remains positive on stress-induced-only events.

### Contributions

1. **Problem and metric framing.** We identify re-entry recovery as a key failure mode in TAP and use AJ_RD_256 as the primary metric for evaluating recovery after disappearance and reappearance.

2. **Selective Local Re-entry Override.** We propose B2-W16-P2, a training-free local override module that combines a standard-strong base branch and a re-entry-strong override branch through a persistent visibility trigger and a short temporal window.

3. **Indicator improvement.** On RGB fresh20-49, B2-W16-P2 improves AJ_RD_256 from **0.3816 to 0.4454** with only **0.528** AJ cost. On DAVIS, it improves AJ_RD_256 from **0.5546 to 0.6251**.

4. **Controlled stress validation.** We introduce ReEntry-TAP, a controlled validation protocol for out-of-frame and occlusion-induced re-entry, and show that B2-W16-P2 generalizes to frozen RGB fresh20-49 stress with consistent AJ_RD gains.

5. **Learned reliability extension.** We further train ReEntry-Guard, a lightweight runtime gate supervised by oracle routing labels on dev stress. ReEntry-Guard improves beyond the fixed B2-W16-P2 rule on all three fresh evaluations and closes 20--32% of the oracle_b2 AJ_RD gap.

---

## 2. Related Work

> TODO: insert full bibliographic references before submission.  
> Current draft uses placeholder reference names.

### Tracking Any Point

Tracking Any Point aims to estimate the trajectory and visibility of arbitrary query points in video. Standard TAP benchmarks evaluate overall point tracking quality using metrics such as AJ, OA, and average point accuracy within thresholds. Recent trackers such as CoTracker-style offline trackers and online long-term trackers have improved the quality of dense or sparse point correspondences. However, standard aggregate metrics can under-emphasize failure cases where the point disappears and later reappears.

### Re-detection and Long-term Tracking

Long-term tracking requires recovering targets after occlusion, disappearance, and appearance changes. In object tracking, re-detection has long been recognized as essential. In point tracking, however, re-entry and re-detection are more subtle because each query corresponds to a local point rather than a semantic object. Recent work such as TAPNext++ explicitly discusses AJ_RD as a metric for reappearing points, and other long-term trackers explore memory or online mechanisms for handling visibility changes. Our work is complementary: instead of proposing a new backbone tracker, we study how to locally activate a re-entry-strong branch while preserving a standard-strong base tracker.

### Robustness and Stress Testing

Controlled stress testing is useful for isolating specific failure modes that may be hidden in aggregate benchmark scores. Our ReEntry-TAP protocol is designed for re-entry-specific evaluation. Unlike a large-scale new benchmark, it is a controlled stress validation protocol that transforms existing RGB stacking videos into two interpretable families of re-entry scenarios: out-of-frame translate and moving occluder. We use this protocol to verify that the gains of B2-W16-P2 are tied to re-entry recovery rather than only standard metric fluctuation.

---

## 3. Re-entry Recovery Problem

### 3.1 Standard Tracking vs Re-entry Recovery

Standard tracking metrics reward accurate point localization and visibility estimation over a whole trajectory. However, a point trajectory may include long invisible intervals. If the tracker fails when the point reappears, this can be averaged with many easier frames. Therefore, standard AJ alone may not clearly reflect re-entry reliability.

We focus on AJ_RD_256 as the primary metric. AJ_RD evaluates post-reappearance recovery for eligible re-entry events. In this work, improvements are meaningful only if they come with controlled standard AJ cost. Thus, our evaluation treats:

```text
Primary metric:   AJ_RD_256
Constraint metric: AJ_256
Secondary metrics: OA_256, delta_avg_256
```

### 3.2 Empirical Tradeoff

On RGB fresh20-49, the standard-strong offline branch has high AJ but lower AJ_RD:

```text
offline:
AJ_RD_256 = 0.3816
AJ_256    = 79.5944
```

A global online branch is more re-entry oriented, but using it everywhere severely damages standard tracking:

```text
online_global:
AJ_RD_256 = 0.4121
AJ_256    = 44.6933
```

Therefore, global online use gives only **+0.0305 AJ_RD** but causes **-34.9011 AJ**. This is an unacceptable operating point for a tracker that must remain accurate outside re-entry windows.

This motivates selective local override.

---

## 4. Method: B2-W16-P2 Selective Local Re-entry Override

### 4.1 Overview

B2-W16-P2 combines two branches:

```text
Base branch B:
  Standard-strong tracker.
  High AJ and stable ordinary tracking.

Override branch O:
  Re-entry-strong branch.
  Better at certain reappearance events but harmful if used globally.
```

The key idea is to use the override branch only in a short window around predicted re-entry and use the base branch everywhere else.

### 4.2 Trigger

Let `v^B_t` be the base visibility prediction at frame `t`, and `v^O_t` be the override visibility prediction. A trigger fires at frame `t` when:

```text
1. The base branch has been invisible immediately before t.
2. The override branch is visible for P consecutive frames.
```

For B2-W16-P2, `P = 2`.

Formally:

```text
g_t = 1 if invisible_run(v^B, t) >= 1 and v^O_t = 1 and v^O_{t+1} = 1
```

The persistent two-frame visibility requirement reduces unstable false triggers.

### 4.3 Local Override Window

When a trigger fires at frame `t`, B2-W16-P2 uses the override branch for a local window:

```text
[t - 1, t + 16]
```

Outside this window, the final prediction is exactly the base prediction.

Let `m_t` be the override mask. The final output is:

```text
Y_t = O_t, if m_t = 1
Y_t = B_t, otherwise
```

The same selection applies to both point coordinates and visibility.

### 4.4 ReEntry-Guard: Learned Reliability Extension

B2-W16-P2 is a training-free rule: once the trigger fires, the local override window is accepted. Oracle analysis shows that this fixed rule is strong but not optimal. For some triggered re-entry queries, the override improves AJ_RD; for others, preserving the base branch is better. This motivates an optional learned reliability gate, **ReEntry-Guard**.

ReEntry-Guard keeps the same candidate local override windows as B2-W16-P2, but adds an accept/reject decision for eligible re-entry candidates. The gate is trained only on dev stress data using oracle routing labels:

```text
label = 1 if B2 candidate AJ_RD_256 > offline AJ_RD_256 for the query
label = 0 otherwise
```

At test time, the gate uses only runtime features available from the two predicted branches, including visibility disagreement, base invisible-run length, override visible persistence, base/override geometry distance, local motion smoothness, and trigger timing. No ground-truth features are used at inference.

Thus the full framework has two instantiations:

```text
B2-W16-P2:
  training-free fixed local override

ReEntry-Guard:
  oracle-guided learned reliability gate over the same local override candidates
```

In our experiments, ReEntry-Guard is trained on RGB dev0-9 translate/occluder L16 stress and then frozen for RGB fresh20-49 natural and frozen stress validation.

### 4.5 Design Properties

B2-W16-P2 has three properties that make it suitable for the re-entry/AJ tradeoff:

```text
Locality:
  Only a short region around re-entry is modified.

Preservation:
  Outside local override windows, the output is identical to the base tracker.

Selective re-entry trust:
  The override branch is trusted only when the base branch appears to have lost the point and the override branch is persistently visible.
```

This makes the method a training-free reliability intervention rather than a global tracker replacement.

---

## 5. ReEntry-TAP Controlled Stress Validation

### 5.1 Motivation

Natural benchmarks are necessary but may not isolate the reason for improvement. To verify that B2-W16-P2 improves re-entry recovery specifically, we construct controlled re-entry stress variants. The goal is not to introduce a large-scale new benchmark, but to provide a reproducible stress validation protocol.

### 5.2 Translate Exit-Reenter Stress

The `translate_exit_reenter` stress shifts the video without wrap-around so that points can leave the frame and later re-enter. Ground-truth coordinates and visibility are transformed accordingly. A point becomes invisible when it moves out of frame and becomes visible again when it re-enters.

For the frozen validation, parameters are fixed:

```text
L = 16
amplitude = 0.4W
t0 = 40
ramp = 8
fill = frame_mean
```

### 5.3 Moving Occluder Stress

The `moving_occluder` stress overlays a vertical full-height moving bar. Coordinates remain unchanged, but visibility is set to false when a point lies under the occluder.

Frozen validation parameters:

```text
L = 16
speed = 4 px/frame
t0 = 40
fill = frame_mean
```

### 5.4 Frozen Validation Protocol

We design and analyze severity curves on RGB dev0-9. Then we freeze all stress parameters and validate on RGB fresh20-49.

Frozen RGB fresh20-49 stress scale:

```text
translate L16:
30 videos
36,176 query instances
12,728 re-entry events

occluder L16:
30 videos
34,808 query instances
19,948 re-entry events

combined:
70,984 query instances
32,676 re-entry events
```

---

## 6. Experiments

### 6.1 Experimental Setup

We evaluate on natural RGB fresh20-49, DAVIS, controlled RGB dev0-9 stress, and frozen RGB fresh20-49 stress. The main compared methods are:

```text
offline:
  standard-strong base branch

online_global:
  re-entry-strong branch used globally

B2-W16-P2:
  selective local override using online branch only in predicted re-entry windows
```

Metrics:

```text
AJ_RD_256: primary re-entry recovery metric
AJ_256: standard tracking constraint metric
OA_256: visibility / occlusion accuracy metric
delta_avg_256: average point accuracy over thresholds
```

---

### 6.2 Main Natural Result on RGB fresh20-49

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 |
| online_global | 0.4121 | 44.6933 | 55.7585 |
| B2-W16-P2 | 0.4454 | 79.0664 | 92.8804 |

B2-W16-P2 vs offline:

```text
AJ_RD_256: 0.3816 -> 0.4454
absolute gain: +0.0638
relative gain: about +16.7%
AJ_256: 79.5944 -> 79.0664
AJ cost: -0.5280
```

B2-W16-P2 vs online_global:

```text
AJ_RD_256: +0.0333
AJ_256:    +34.3731
```

This shows that B2-W16-P2 not only avoids the online/global AJ collapse, but also achieves higher AJ_RD than the online branch itself.

---

### 6.3 Natural Result on DAVIS

| Method | AJ_RD_256 | AJ_256 |
|---|---:|---:|
| offline | 0.5546 | 70.0510 |
| global B1 | 0.6279 | 47.4046 |
| B2-W16-P2 | 0.6251 | 69.0119 |

B2-W16-P2 vs offline:

```text
AJ_RD_256: +0.0705
AJ_256:    -1.0391
```

B2-W16-P2 vs global B1:

```text
AJ_RD_256: -0.0028
AJ_256:    +21.6073
```

The result shows that local override retains almost all of the global re-entry gain while avoiding the severe standard tracking collapse.

---

### 6.4 Natural Ablation on RGB fresh20-49

| Method | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD | ΔAJ |
|---|---:|---:|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 | — | — |
| online_global | 0.4121 | 44.6933 | 55.7585 | +0.0305 | -34.9011 |
| b2_fullpost_p1 | 0.4219 | 78.9590 | 92.7706 | +0.0403 | -0.6354 |
| b2_w8_p2 | 0.4462 | 79.0753 | 92.8853 | +0.0646 | -0.5191 |
| b2_w16_p1 | 0.4456 | 79.0634 | 92.8891 | +0.0640 | -0.5310 |
| B2-W16-P2 | 0.4454 | 79.0664 | 92.8804 | +0.0638 | -0.5280 |
| b2_w32_p2 | 0.4421 | 79.0451 | 92.8736 | +0.0605 | -0.5493 |

Interpretation:

```text
1. Global online use is not a viable tradeoff.
2. Full-post override is better than global online but weaker than local windows.
3. W8/W16 local windows give the strongest AJ_RD/AJ tradeoff.
4. B2-W16-P2 is a conservative and effective operating point.
```

---

### 6.5 Natural Video-level Robustness

B2-W16-P2 vs offline on RGB fresh20-49:

| Metric | Mean Δ | 95% CI | Positive videos | Sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD_256 | +0.0613 | [0.0419, 0.0805] | 24/30 | 0.000546 |
| AJ_256 | -0.5280 | [-0.9716, -0.0668] | 7/30 positive | 0.005223 |

Additional AJ cost:

```text
AJ drop > 1 point: 10/30 videos
AJ drop > 2 points: 3/30 videos
```

B2-W16-P2 vs online_global:

| Metric | Mean Δ | 95% CI | Positive videos |
|---|---:|---:|---:|
| AJ_RD_256 | +0.0390 | [0.0166, 0.0669] | 23/30 |
| AJ_256 | +34.3731 | [32.9299, 35.7316] | 30/30 |

The main natural gain is therefore not driven by a small number of videos.

---

### 6.6 Controlled Dev Stress Severity

#### Translate stress

| L | offline AJ_RD | B2 AJ_RD | ΔAJ_RD | ΔAJ |
|---:|---:|---:|---:|---:|
| 8 | 0.4709 | 0.5513 | +0.0804 | -0.0767 |
| 16 | 0.4708 | 0.5485 | +0.0777 | -0.0312 |
| 32 | 0.4614 | 0.5452 | +0.0838 | +0.0782 |

#### Moving-occluder stress

| L | offline AJ_RD | B2 AJ_RD | ΔAJ_RD | ΔAJ |
|---:|---:|---:|---:|---:|
| 8 | 0.6417 | 0.6875 | +0.0458 | -0.4282 |
| 16 | 0.6422 | 0.6873 | +0.0451 | -0.5198 |
| 32 | 0.6388 | 0.6825 | +0.0437 | -0.7156 |

The gains persist across severity levels and stress families.

---

### 6.7 Frozen RGB fresh20-49 Stress Validation

#### Translate L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.4788 | 75.1940 | 90.7805 |
| online_global | 0.4981 | 43.0978 | 56.7847 |
| B2-W16-P2 | 0.5310 | 74.8457 | 92.2166 |

B2-W16-P2 vs offline:

```text
AJ_RD_256 +0.0522
AJ_256    -0.3483
```

#### Occluder L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.6311 | 77.6280 | 90.8918 |
| online_global | 0.6146 | 43.1490 | 58.0983 |
| B2-W16-P2 | 0.6588 | 76.7991 | 92.1682 |

B2-W16-P2 vs offline:

```text
AJ_RD_256 +0.0277
AJ_256    -0.8289
```

Frozen stress video-level robustness:

| Stress | Metric | Mean Δ | 95% CI | Positive videos |
|---|---|---:|---:|---:|
| translate L16 | AJ_RD_256 | +0.0451 | [0.0317, 0.0586] | 27/30 |
| occluder L16 | AJ_RD_256 | +0.0263 | [0.0163, 0.0369] | 26/30 |

---

### 6.8 Stress-induced-only Analysis

To ensure that the controlled stress gains are not due only to natural re-entry events already present in the source videos, we separate events into natural, stress-induced, and mixed categories.

| Stress | Stress-induced events | Stress-induced rate | offline | online | B2 | B2-offline |
|---|---:|---:|---:|---:|---:|---:|
| translate L16 | 2,284 | 0.2364 | 0.7618 | 0.7478 | 0.7802 | +0.0184 |
| occluder L16 | 10,072 | 0.6235 | 0.7415 | 0.7118 | 0.7541 | +0.0126 |

The method remains positive on stress-induced-only events in the frozen 30-video validation.

---

### 6.9 Oracle Upper-bound

We estimate headroom by a per-query oracle that selects a candidate branch over offline if and only if the candidate gives higher per-query AJ_RD_256 on an eligible re-entry query. Non-re-entry queries remain offline. This is not a deployable method; it is an upper bound for routing / gating quality.

#### Natural RGB fresh20-49

| Method | AJ_RD_256 | AJ_256 |
|---|---:|---:|
| offline | 0.3816 | 79.5944 |
| B2-W16-P2 | 0.4454 | 79.0664 |
| oracle_b2 | 0.4597 | 79.5633 |
| oracle_online | 0.4677 | 78.2940 |

Headroom:

```text
oracle_b2 vs B2-W16-P2:
AJ_RD +0.0143
AJ    +0.4969
```

#### Frozen translate L16

```text
B2-W16-P2: 0.5310 AJ_RD / 74.8457 AJ
oracle_b2: 0.5437 AJ_RD / 75.3044 AJ
headroom: +0.0127 AJ_RD / +0.4587 AJ
```

#### Frozen occluder L16

```text
B2-W16-P2: 0.6588 AJ_RD / 76.7991 AJ
oracle_b2: 0.6732 AJ_RD / 77.6702 AJ
headroom: +0.0144 AJ_RD / +0.8711 AJ
```

Interpretation:

```text
B2-W16-P2 captures most of the available gain under simple local override.
A better reliability gate could still add roughly +1.3 to +1.4 AJ_RD points and recover some AJ.
```

---

### 6.10 ReEntry-Guard Learned Reliability Gate

We train ReEntry-Guard on dev translate_L16 and dev occluder_L16 using oracle accept/reject labels. The training set contains 7,659 eligible re-entry samples, with 3,712 positive and 3,947 negative examples. We evaluate the frozen gate on three fresh20-49 settings: natural RGB fresh20-49, frozen translate_L16, and frozen occluder_L16.

The best v2 model is a random forest gate with threshold 0.40.

| Setting | B2 AJ_RD | B2 AJ | ReEntry-Guard AJ_RD | ReEntry-Guard AJ | ΔAJ_RD vs B2 | ΔAJ vs B2 |
|---|---:|---:|---:|---:|---:|---:|
| RGB fresh20-49 natural | 0.4454 | 79.0664 | 0.4499 | 79.1749 | +0.0045 | +0.1085 |
| fresh20-49 translate L16 | 0.5310 | 74.8457 | 0.5336 | 74.8490 | +0.0026 | +0.0033 |
| fresh20-49 occluder L16 | 0.6588 | 76.7991 | 0.6632 | 77.0761 | +0.0044 | +0.2770 |

Video-level statistics show that the gain is small but consistent. On natural RGB fresh20-49, ReEntry-Guard improves AJ_RD over B2 on 19/30 videos with mean video-level gain +0.0045 and 95% CI [0.0006, 0.0091]. On frozen occluder L16, it improves AJ_RD on 28/30 videos with mean gain +0.0043 and 95% CI [0.0032, 0.0057], while also improving AJ by +0.277 on average.

Compared with the oracle_b2 upper bound, ReEntry-Guard closes about 31.5% of the natural AJ_RD gap, 20.5% of the translate stress gap, and 30.6% of the occluder stress gap. This confirms that runtime reliability gating can improve beyond the fixed B2-W16-P2 rule, although the remaining oracle gap suggests room for stronger tracker-native uncertainty features.

## 7. Analysis

### 7.1 Why Local Override Works

The offline/base branch is reliable for standard tracking, but it may fail to recover after disappearance. The online/override branch can sometimes re-detect the point, but it is too unstable to use globally. Local override restricts the override branch to the region where its benefit is most likely: the predicted re-entry window.

This explains the ablation pattern:

```text
online_global:
  small AJ_RD gain, catastrophic AJ loss.

fullpost:
  less catastrophic, but overuses override after trigger.

W8/W16 local windows:
  high AJ_RD gain with small AJ cost.
```

### 7.2 Why P2 Is Used

W16-P1 and W16-P2 are close on RGB fresh20-49. P2 is used as the main setting because it is a more conservative persistent trigger. It reduces unstable one-frame override activations and is consistent with the natural false-trigger analysis on DAVIS. In the final method, we prefer the conservative trigger because the paper goal is not simply maximizing AJ_RD, but improving AJ_RD while preserving standard tracking.

### 7.3 Headroom and Future Learned Gates

The oracle upper-bound suggests that B2-W16-P2 is not the final possible solution. A better gate could choose when to use override more accurately. The consistent oracle_b2 gain of about +0.013 to +0.014 AJ_RD indicates that a learned or feature-based reliability gate could be a promising future direction.

ReEntry-Guard v2 validates this direction: a lightweight random-forest gate trained on dev stress improves over B2-W16-P2 on all three fresh evaluations and closes 20--32% of the oracle_b2 gap. However, the gains remain modest. Prior generic appearance-feature pilots also did not meaningfully improve harmful override detection, suggesting that future gates may need tracker-specific correspondence uncertainty, dense correspondence confidence, or better temporal uncertainty modeling.

---

## 8. Limitations

1. **Not a universal SOTA tracker.** B2-W16-P2 is an inference-time reliability intervention on a base/override tracker pair. We do not claim to outperform all point trackers under every benchmark.

2. **Requires two branches.** The current implementation uses both a base tracker and an override branch. The local fusion itself is cheap, but the full system requires predictions from both branches.

3. **Heuristic trigger and modest learned-gate gains.** The fixed B2 trigger is simple and training-free. ReEntry-Guard shows that learned reliability gating can improve beyond B2, but the gains are modest and do not close the full oracle gap.

4. **Controlled stress is not a replacement for large-scale benchmarks.** ReEntry-TAP is a controlled validation protocol, not a new large-scale benchmark. Its purpose is to isolate re-entry failure modes.

5. **External baselines remain limited.** The current main ReEntry-TAP comparisons are within a CoTracker-style offline/online branch family. We attempted TAPNext and TrackOn2 stress baselines: TAPNext was runnable but not parity-established under the stress protocol, and TrackOn2 was blocked by missing `mmcv.ops` support in the current environment. A parity-valid TrackOn2 first-input supplement shows small positive plug-in gains, but future work should evaluate TrackOn2, TAPNext++, AllTracker, and other long-term point trackers under a fully aligned ReEntry-TAP protocol.

---

## 9. Conclusion

We studied the re-entry recovery problem in Tracking Any Point and showed that a re-entry-strong branch can improve AJ_RD but causes severe standard tracking collapse when used globally. We proposed B2-W16-P2, a training-free Selective Local Re-entry Override module that activates the re-entry branch only inside predicted local re-entry windows. On RGB fresh20-49, B2-W16-P2 improves AJ_RD_256 from 0.3816 to 0.4454 while reducing AJ by only 0.528. On DAVIS, it improves AJ_RD_256 from 0.5546 to 0.6251. Controlled ReEntry-TAP stress validation further shows that the gain persists under out-of-frame and moving-occluder re-entry, including frozen 30-video fresh20-49 validation and stress-induced-only event analysis. These results demonstrate that selective local override is a simple and effective way to improve re-entry recovery while preserving standard tracking performance.

---

## Appendix A. Main Artifact References

```text
docs/reentry_tap_method_paper_tables_2026-07-01.md
docs/reentry_tap_method_ablation_statistics_oracle_2026-07-01.md
docs/reentry_guard_v1_v2_results_2026-07-01.md
docs/reentry_tap_external_baseline_smoke_2026-07-01.md
docs/reentry_stress_rgb_fresh20_49_frozen_validation_2026-07-01.md
docs/reentry_stress_intersection_ablation_stats_2026-07-01.md
docs/reentry_stress_trigger_taxonomy_2026-07-01.md
docs/REENTRY_TAP_REPRODUCIBILITY.md
```

## Appendix B. Figures

```text
docs/figures/reentry_tap_method/natural_tradeoff_aj_vs_ajrd.png
docs/figures/reentry_tap_method/natural_ablation_ajrd.png
docs/figures/reentry_tap_method/frozen_fresh20_49_b2_gain_ajrd.png
```

## Appendix C. External Baseline Feasibility and Supplement

### C.1 TAPNext ReEntry-TAP stress smoke

We implemented a TAPNext exporter for ReEntry-TAP stress datasets:

```text
scripts/export_tapnext_reentry_stress_cache.py
```

The exporter successfully runs on dev0 translate_L16 with 1,116 queries. However, the available local TAPNext checkpoint / adapter does not establish a usable strong baseline under this stress protocol. On the same dev0 translate_L16 video:

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| CoTracker3 offline | 0.6667 | 76.0920 | 88.5175 |
| CoTracker3 online | 0.7061 | 46.4554 | 61.2806 |
| B2-W16-P2 | 0.7239 | 77.2732 | 90.6263 |
| TAPNext local checkpoint | 0.0311 | 10.7467 | 28.5831 |

Because parity is not established, this TAPNext stress result should not be interpreted as a strong external comparison. It is recorded only as an engineering feasibility attempt.

### C.2 TrackOn2 stress smoke blocker

We also implemented a TrackOn2 exporter for ReEntry-TAP stress datasets:

```text
scripts/export_trackon2_reentry_stress_cache.py
```

The current environment is blocked by a missing dependency:

```text
ModuleNotFoundError: No module named 'mmcv'
from mmcv.ops import MultiScaleDeformableAttention
```

Therefore, a strict ReEntry-TAP stress comparison with TrackOn2 is pending an environment with `mmcv.ops` support. We do not use a fallback attention implementation because that would change the TrackOn2 architecture and would not be a valid external baseline.

### C.3 Parity-valid TrackOn2 first-input supplement

Although TrackOn2 stress evaluation is blocked in the current environment, we have a parity-valid TrackOn2 first-query/input-resolution DAVIS bridge. Under that protocol, B2-W16-P2 can be applied as a plug-in local override on top of TrackOn2:

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| TrackOn2 first-input | 0.5444 | 67.0406 | 93.0615 |
| B2-W16-P2, TrackOn2 base + CoTracker3 override | 0.5509 | 67.1260 | 93.2224 |

Gain over TrackOn2:

```text
AJ_RD_256 +0.0065
AJ_256    +0.0854
OA_256    +0.1609
```

This is a supplemental plug-in generality result, not a main-table comparison. It supports the claim that selective local override can provide small positive gains on top of an external reproduced baseline under a parity-valid protocol.

## Appendix D. TODO Before Submission

```text
1. Add formal citations and bibliography.
2. Decide final title: method-first or ReEntry-TAP-first.
3. Add one external strong baseline smoke if engineering allows.
4. Convert this Markdown draft into LaTeX conference format.
5. Move some large analysis tables to appendix.
6. Polish abstract and introduction after target venue is chosen.
```
