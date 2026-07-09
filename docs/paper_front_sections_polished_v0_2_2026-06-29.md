# Polished Front Sections v0.2 — 2026-06-29

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
