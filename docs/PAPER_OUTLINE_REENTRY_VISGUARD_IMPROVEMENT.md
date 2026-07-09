# Paper Outline — ReEntry-VisGuard Improvement Paper

Date: 2026-07-02

## Working title

Preferred:

```text
Improving Re-detection in Tracking Any Point via Coordinate-Preserving Visibility Recovery
```

Alternative:

```text
ReEntry-VisGuard: A Plug-in Visibility Recovery Layer for Improving Re-detection in Tracking Any Point
```

## One-sentence thesis

```text
ReEntry-VisGuard improves TAP re-detection by preserving reliable base coordinates and locally recovering visibility in predicted re-entry windows, yielding stable AJ_RD gains under small standard-AJ cost.
```

## Paper identity

This should be written as an **improvement method paper**, not only a diagnosis paper.

```text
Core method type:
  inference-time plug-in enhancement layer

Target metric:
  AJ_RD_256 / re-detection reliability

Constraint metric:
  AJ_256 / standard TAP quality preservation

Core mechanism:
  coordinate-preserving visibility recovery
```

Do not write it as:

```text
new TAP backbone
universal TAP SOTA tracker
CoTracker-only engineering trick
```

---

## Abstract skeleton

```text
Tracking Any Point (TAP) methods must recover query points after occlusion or frame exit, but re-detection remains a weak point of existing trackers. We study re-entry recovery through a channel-wise decomposition of TAP predictions into coordinates and visibility. In our evaluated settings, the offline/base tracker often preserves accurate re-entry coordinates, while its predicted visibility remains overly conservative. Motivated by this observation, we propose ReEntry-VisGuard, a prediction-only plug-in layer that preserves base coordinates and locally recovers visibility in predicted re-entry windows. On RGB fresh20-49 natural re-entry and controlled translate/occluder ReEntry-TAP stress settings, ReEntry-VisGuard-W8P2 consistently improves AJ_RD over the offline base while keeping standard AJ within a small margin. Our results show that coordinate-preserving visibility recovery is a simple and effective route for improving TAP re-detection without retraining a tracker backbone.
```

---

## Main contributions

### Contribution 1 — Plug-in re-detection enhancement

```text
We propose ReEntry-VisGuard, a prediction-only plug-in visibility recovery layer for improving TAP re-detection. It preserves base coordinates and replaces visibility only in predicted re-entry windows.
```

### Contribution 2 — Channel-wise intervention principle

```text
We decompose re-entry predictions into coordinate and visibility channels, showing that channel-wise intervention can recover AJ_RD while avoiding the standard-AJ damage of global online/override prediction.
```

### Contribution 3 — Natural + controlled stress validation

```text
We validate the method on RGB fresh20-49 natural re-entry and controlled translate/occluder ReEntry-TAP stress settings, with query-weighted metrics and video-level paired statistics.
```

### Contribution 4 — Generality target, if cross-source succeeds

```text
We demonstrate that coordinate-preserving visibility recovery can transfer across tracker / visibility-source combinations.
```

If cross-source experiments fail or remain inconclusive, demote Contribution 4 to an appendix analysis:

```text
We analyze external source feasibility and identify parity requirements for future plug-in generalization.
```

---

## Core method definition

### Inputs

Base tracker output:

```text
p_b(t): base coordinates
v_b(t): base visibility
```

Override / responsive source output:

```text
p_o(t): override coordinates
v_o(t): override visibility
```

### Trigger

A predicted re-entry trigger occurs at frame `t` when:

```text
base predicted invisible after query
and override predicted visible for P consecutive frames
```

Final main setting:

```text
P = 2
W = 8
pre = 1
window = [t - 1, t + 8]
```

### Prediction

```text
p_hat(t) = p_b(t)

v_hat(t) =
    v_o(t), if t is inside a predicted re-entry window
    v_b(t), otherwise
```

### Important claim

```text
No GT re-entry window, GT visibility, or GT coordinate is used at inference.
```

---

## Main result tables to include

### Table 1 — RGB fresh20-49 natural

Use a channel-aware table:

| Method | Coordinate source | Visibility source | AJ_RD_256 | AJ_256 | OA_256 |
|---|---|---|---:|---:|---:|
| offline | offline | offline | 0.3816 | 79.5944 | 91.4636 |
| online_global | online | online | 0.4121 | 44.6933 | 55.7585 |
| full_w8_p2 | local override | local override | 0.4462 | 79.0753 | 92.8853 |
| ReEntry-VisGuard-W8P2 | offline | local override visibility | 0.4510 | 79.1110 | 92.8853 |
| ReEntry-Guard RF | learned local | learned local | 0.4499 | 79.1749 | 92.0864 |

Main interpretation:

```text
Online improves AJ_RD but collapses AJ. Local full override fixes much of the tradeoff. ReEntry-VisGuard further improves AJ_RD by transferring only visibility while preserving base coordinates.
```

### Table 2 — controlled stress, fresh20-49

| Setting | Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---|---:|---:|---:|
| translate_L16 | offline | 0.4788 | 75.1940 | 90.7805 |
| translate_L16 | ReEntry-VisGuard-W8P2 | 0.5336 | 74.7705 | 92.2245 |
| occluder_L16 | offline | 0.6311 | 77.6280 | 90.8918 |
| occluder_L16 | ReEntry-VisGuard-W8P2 | 0.6659 | 77.0436 | 92.1748 |

Main interpretation:

```text
The improvement persists under controlled re-entry stress, not only in natural aggregate data.
```

### Table 3 — video-level paired statistics

ReEntry-VisGuard-W8P2 vs offline:

| Setting | mean ΔAJ_RD | 95% bootstrap CI | + / - videos | sign p |
|---|---:|---:|---:|---:|
| natural | +0.0677 | [0.0512, 0.0848] | 28 / 1 | 1.1e-7 |
| translate_L16 | +0.0475 | [0.0343, 0.0608] | 28 / 2 | 8.7e-7 |
| occluder_L16 | +0.0337 | [0.0242, 0.0442] | 28 / 2 | 8.7e-7 |

Main interpretation:

```text
The gain is not driven by a few videos; it is stable under video-level paired testing.
```

### Table 4 — channel-wise factorial intervention

To be generated by `scripts/eval_reentry_channel_factorial.py`:

```text
base_coord + base_vis
override_coord + override_vis
override_coord + base_vis
base_coord + override_vis
base_coord + local_override_vis = ReEntry-VisGuard
```

This table is the core improvement-paper evidence. It should show that the gain is from the visibility channel, not arbitrary ensembling.

---

## Main figures

### Figure 1 — Teaser

Visual story:

```text
point disappears / re-enters
base coordinate is correct
base visibility remains invisible
ReEntry-VisGuard keeps same coordinate and recovers visible state
```

### Figure 2 — Method diagram

Show:

```text
base tracker branch
responsive visibility source
detected re-entry window
coordinate preservation
visibility replacement
```

### Figure 3 — Coordinate-vs-visibility diagnosis

Plot:

```text
coord8_rate
predvis_recall
first_coord8_rate
first_predvis_rate
```

Key message:

```text
coordinates saturate, visibility lags.
```

### Figure 4 — Main result tradeoff

Plot AJ_RD gain vs AJ change:

```text
x = ΔAJ_256 vs offline
y = ΔAJ_RD_256 vs offline
```

ReEntry-VisGuard should sit in:

```text
high AJ_RD gain, small AJ loss
```

### Figure 5 — Qualitative success/failure panels

Need at least:

```text
2 natural successes
2 stress successes
1 false-visible / harmful failure
1 coordinate-failure limitation
```

---

## Section outline

### 1. Introduction

Story:

```text
TAP is useful, but long-term use requires re-detection after occlusion / frame exit.
Recent work has identified re-detection as a blind spot.
Existing improvement routes use better matching, memory, training, or data.
We ask whether re-detection can be improved at inference time by channel-wise visibility recovery.
```

### 2. Related Work

Organize by improvement routes:

```text
2.1 TAP formulation and benchmarks: TAP-Vid
2.2 Matching and refinement: TAPIR, LocoTrack
2.3 Training and data scaling: BootsTAP, CoTracker3, PointOdyssey, SynthVerse
2.4 Online memory and re-detection: TAPNext, TAPNext++, Track-On2
2.5 Our distinction: inference-time coordinate-preserving visibility recovery
```

### 3. Re-detection improvement formulation

Define:

```text
AJ_RD_256
standard AJ constraint
coordinate channel
visibility channel
predicted re-entry windows
```

### 4. Method: ReEntry-VisGuard

Include:

```text
base / override outputs
trigger
window
coordinate preservation
visibility replacement
W8P2 selection
prediction-only inference
```

### 5. Experimental setup

Include:

```text
RGB fresh20-49 natural
ReEntry-TAP translate_L16
ReEntry-TAP occluder_L16
metrics: AJ_RD_256, AJ_256, OA_256
baselines: offline, online_global, full local override, ReEntry-Guard
protocol: W8P2 selected on dev, frozen on fresh20-49
```

### 6. Results

Subsections:

```text
6.1 Natural re-entry improvement
6.2 Controlled re-entry stress improvement
6.3 Video-level robustness
6.4 Channel-wise intervention study
6.5 W/P ablation
6.6 Qualitative examples
```

### 7. Sanity and limitations

Include:

```text
no GT at inference
cache alignment
coord1/2/4/8/16 sanity
visibility shuffle
coordinate perturbation
external baseline parity limitations
failure taxonomy
```

### 8. Conclusion

Main final sentence:

```text
ReEntry-VisGuard shows that existing TAP trackers can recover substantial re-detection performance through a simple coordinate-preserving visibility recovery layer, highlighting visible-state recovery as a practical path for improving long-term TAP reliability.
```

---

## Claims allowed

Strong but safe:

```text
ReEntry-VisGuard robustly improves AJ_RD over the offline base across natural and controlled re-entry stress settings.
```

```text
The method preserves standard AJ within a small margin while improving re-detection reliability.
```

```text
Channel-wise visibility recovery is cleaner than full coordinate+visibility override in our final settings.
```

If cross-source succeeds:

```text
The plug-in principle transfers across multiple tracker / visibility-source combinations.
```

---

## Claims not allowed

Do not claim:

```text
universal TAP SOTA
beating TAPNext++ / TrackOn2 globally
coordinate localization is never a problem
visibility-only is always better than full B2
```

Correct boundary:

```text
In our evaluated re-entry settings, the recoverable bottleneck is primarily visibility-state recovery, and coordinate-preserving visibility recovery is an effective inference-time improvement.
```

---

## Immediate TODO list

```text
1. Implement clean runner: scripts/run_reentry_visguard_w8p2_eval.py
2. Implement no-leak sanity: scripts/audit_reentry_visguard_no_leak.py
3. Implement channel factorial: scripts/eval_reentry_channel_factorial.py
4. Generate paper-ready tables from the three scripts.
5. Start LocoTrack cross-source smoke.
6. Rewrite LaTeX V2 around improvement framing.
```
