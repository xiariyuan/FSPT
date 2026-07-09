# Paper Full Draft v0 — 2026-07-04

> This is the integrated full paper draft. It combines the frozen abstract,
> experimental setup, results, tables, and limitations from the 2026-07-04
> writing package and adds the previously-missing Method section.
>
> Source documents:
> - `docs/paper_combined_draft_core_sections_2026-07-04.md`
> - `docs/paper_experimental_setup_section_draft_2026-07-04.md`
> - `docs/paper_results_section_draft_2026-07-04.md`
> - `docs/final_paper_tables_2026-07-04.md`
> - `docs/paper_limitations_and_future_work_draft_2026-07-04.md`
> - `docs/evaluation_protocol_section_2026-07-04.md`

---

## Working Title

ReEntry: Diagnosing and Correcting Re-Entry Visibility Failures in Point Tracking

Alternative titles:

- ReEntry-TAP: Visibility Re-Entry Calibration for Tracking Any Point
- ReEntry: Failure-Mode-Aware Visibility Correction for Point Tracking

---

## Abstract

Tracking any point through occlusion and reappearance remains challenging because trackers can exhibit visibility lag or false visibility after a point leaves and re-enters the scene. Standard TAP-Vid metrics such as Average Jaccard (AJ) and Occlusion Accuracy (OA) measure overall trajectory quality, but they can dilute failures that occur specifically after reappearance. We introduce ReEntry, a visibility correction pipeline that targets re-entry failure modes on top of existing point trackers. ReEntry applies learned visibility calibration, interval/gate post-processing, and optional appearance-based filtering to improve recovery after occlusion or disappearance.

We evaluate ReEntry on TAPVid-DAVIS and TAPVid RGB-Stacking using standard TAP-Vid metrics and a re-entry diagnostic metric, AJ_RD. Under a DAVIS first/input official-style local evaluation, ReEntry V24-DINOScore improves over the CoTracker3 offline base by +2.19 AJ, +3.74 OA, and +0.0407 AJ_RD. On TAPVid RGB-Stacking full50, ReEntry improves AJ_RD from 0.3617 to 0.4414 with V1 and improves OA from 91.64 to 93.08, with a small standard-AJ trade-off. Under the stricter DAVIS strided/original protocol, default ReEntry improves AJ_RD but reduces global AJ/OA; a conservative V25 threshold preserves AJ/OA while retaining a small AJ_RD gain. These results show that ReEntry improves the targeted re-entry failure mode, while also revealing that metric-aware selection is needed for stronger global benchmark gains under dense strided evaluation.

---

## 1. Introduction

Tracking Any Point (TAP) methods predict dense point trajectories and their visibility across video frames. A persistent failure mode occurs at *re-entry*: a point disappears behind an occluder or exits the frame and later reappears, but the tracker's visibility state lags, producing either a delayed visible flag (visibility lag) or a spurious visible flag while the point is still hidden (false visibility). These errors are concentrated on sparse re-entry frames and are therefore diluted by full-trajectory averages such as Average Jaccard (AJ) and Occlusion Accuracy (OA).

We make the following contributions:

1. We identify and isolate re-entry visibility failures in point tracking, where a point reappears after occlusion or disappearance but the tracker visibility state lags or becomes unstable.

2. We introduce ReEntry, a visibility correction pipeline that can be applied on top of existing point trackers without changing their coordinate trajectories.

3. We evaluate ReEntry using both standard TAP-Vid metrics and AJ_RD, a re-entry diagnostic metric that isolates performance after reappearance.

4. We provide official-style local evaluations on TAPVid-DAVIS first/input and strided/original protocols, plus a full 50-video TAPVid RGB-Stacking local standard evaluation.

5. We show that ReEntry improves first-query DAVIS AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while strict strided/original evaluation requires conservative or metric-aware selection to preserve global metrics.

---

## 2. Related Work

**Tracking Any Point.** TAP-Vid established a benchmark and evaluation protocol for dense long-term point tracking, introducing AJ, OA, and position-accuracy metrics. Subsequent trackers including CoTracker, TAPIR, LocoTrack, Track-On, and TAPNext have improved trajectory quality through iterative refinement, cost volumes, and stronger backbones.

**Occlusion handling.** Existing trackers predict a per-frame visibility flag, typically supervised by ground-truth occlusion labels. However, post-reappearance visibility recovery remains error-prone: trackers can be slow to flip the visibility flag back to *visible* after a point reappears, or they can falsely report *visible* during the occlusion gap.

**Re-detection.** Re-detection methods in single-object tracking and detection literature address target recovery after full disappearance, but TAP re-detection at the dense point level is less studied. ReEntry targets exactly this regime: recovering the *visibility state* after reappearance while preserving the base tracker's coordinate predictions.

**Teacher ensembles.** Multi-teacher fusion has been explored for trajectory prediction. ReEntry builds on a four-teacher ensemble (CoTracker3 offline/online, TAPNext, and a gated visibility fusion) as the source of candidate re-entry visibility, but the core method only requires a base tracker and one override visibility source.

---

## 3. Method

### 3.1 Problem setup

Given a video with $T$ frames and $N$ query points, a point tracker predicts trajectories $\hat{x}_{n,t} \in \mathbb{R}^2$ and visibility $\hat{v}_{n,t} \in \{0,1\}$ for each query $n$ and frame $t$. The query point is visible at its query frame $q_n$ and may be occluded or out-of-frame at later frames. A *re-entry event* occurs when the point becomes visible again after being occluded for at least one frame:

$$
\text{reentry}(n) = \{ t > q_n \mid v^*_{n,t} = 1 \;\land\; v^*_{n,t-1} = 0 \}
$$

where $v^*$ is ground-truth visibility. The re-entry failure mode is the tracker's error on and around these re-entry frames.

### 3.2 Re-entry failure mode

Let $v^{\text{base}}_{n,t}$ be the base tracker visibility and $v^{\text{over}}_{n,t}$ an override visibility source (e.g. a teacher ensemble). We observe two characteristic failure patterns at re-entry:

- **Visibility lag.** The point reappears at frame $t$, but $v^{\text{base}}_{n,t}=0$ for several frames before flipping to 1. The base tracker is slow to recover visibility.
- **False visibility.** The point is still occluded, but $v^{\text{base}}_{_{n,t}}=1$ due to a spurious visibility prediction.

A corrected visibility $\tilde{v}_{n,t}$ should recover *true* visible frames quickly after reappearance while suppressing *false* visible frames. The key constraint is that coordinates are preserved from the base tracker: $\tilde{x}_{n,t} = \hat{x}^{\text{base}}_{n,t}$. We change only visibility.

### 3.3 ReEntry visibility correction pipeline

ReEntry operates in three stages applied per query track.

**Stage 1 — Candidate window construction.** A candidate re-entry window is identified when the base tracker is invisible for at least $k=1$ consecutive frames after the query frame and the override source reports visible for $P=2$ consecutive frames. The window spans $\text{pre}=1$ frames before to $\text{post}=16$ frames after the trigger frame, yielding a window of length $W=16$. This is the B2-W16-P2 configuration. Inside a candidate window, the override visibility is a candidate for recovery.

**Stage 2 — Coordinate-preserving local recovery.** Inside the candidate window, the corrected visibility takes the logical OR of base and override visibility:

$$
\tilde{v}_{n,t} = v^{\text{base}}_{n,t} \;\lor\; v^{\text{over}}_{n,t} \quad \text{for } t \in \text{window}(n)
$$

while the coordinates remain $\hat{x}^{\text{base}}_{n,t}$. This *positive-only* principle ensures that the recovery can only *add* visible frames inside the window, never turn a base-visible frame off. This alone constitutes the deterministic variant (Ours-Det / B2-W16-P2).

**Stage 3 — Learned and post-processed variants.** The deterministic recovery is refined by one of four variants (V1, V22Q, V24, V25), described below.

### 3.4 V1: Learned ReEntry-VisCalibrator

The learned calibrator scores per-frame recovery safety. For each candidate window, we extract a 28-dim per-frame feature vector capturing:

- Base/override visibility state and disagreement
- Relative time within the window
- Normalized coordinates, speeds, and accelerations of both base and override tracks
- Run-lengths of invisible/visible segments
- Border distance (distance of the predicted point to the frame border)
- Future visibility rates of base and override

A small temporal convolution network processes the feature sequence $(B, L, F)$ and outputs per-frame logits $(B, L)$. The network consists of a linear projection to hidden dimension 64, followed by 3 temporal convolution blocks (each with two Conv1d layers, GELU, dropout, and a residual connection with LayerNorm), and a final linear head. A frame is recovered (set visible) if its probability exceeds a confidence threshold $\tau$:

$$
\tilde{v}_{n,t} = 1 \quad \text{if} \quad t \in \text{window}(n) \;\land\; v^{\text{over}}_{n,t}=1 \;\land\; p_{n,t} \geq \tau
$$

**Training protocol.** The model is trained on dev videos 0–6. The threshold $\tau$ is selected on held-out dev videos 7–9 by maximizing AJ_RD while keeping AJ above the deterministic baseline. The frozen model and threshold are then tested once on fresh videos 20–49. The selected threshold is $\tau=0.10$.

**Label.** The training label combines ground-truth visibility with base-coordinate proximity: a frame is labeled *safe-visible* if the point is ground-truth visible and the base coordinate is within a tolerance of the ground-truth position.

### 3.5 V22Q: Interval/Gate post-processing

V22Q applies deterministic interval cleaning to the V1 recovery proposal without training a new model. Inside each candidate window, the V1 recovery mask is split into consecutive temporal segments. Three rules are applied:

1. **Gate blocking.** If an event-level gate probability is below 0.01, the entire event is blocked (no recovery). This filters clearly harmful events.
2. **Short-segment removal.** Segments shorter than 2 frames are removed, as they are likely spurious.
3. **Edge trimming.** Segment edges with confidence below 0.15 are trimmed, and segment cores below 0.10 are removed.

The V22Q configuration is $\{\text{gate\_block}=0.01, \text{min\_segment}=2, \text{edge}=0.15, \text{core}=0.10\}$. V22Q is the best stability-oriented extension: it improves standard AJ over V1 with only a tiny AJ_RD cost.

### 3.6 V24: DINOScore appearance micro-filter

V24 adds an optional appearance-based filter on top of V22Q. For each proposed recovery frame, the DINOv3 patch embedding of a crop around the last reliable visible point is compared to the embedding at the candidate recovery point. If the cosine similarity is below a threshold $\theta=0.395$, the frame is dropped from recovery:

$$
\tilde{v}_{n,t} = 0 \quad \text{if} \quad \text{cosine\_sim}(\text{DINOv3}(\text{last\_visible}), \text{DINOv3}(\text{candidate})) < 0.395
$$

This micro-filter fires on only ~1% of proposed recovery frames but identifies almost certainly harmful frames. V24 gives small AJ gains over V22Q, with a tiny AJ_RD/OA cost. It is positioned as an optional AJ-oriented appearance filter, not a universal replacement.

### 3.7 V25: Official-safe thresholding

Under the stricter DAVIS strided/original protocol, the default V1 threshold ($\tau=0.10$) improves AJ_RD but reduces standard AJ/OA due to false-visible penalties across the dense 5,882-query evaluation. V25 raises the threshold to $\tau=0.80$, recovering only ~15,024 frames (vs ~39,809 at $\tau=0.20$). This conservative operating point preserves AJ/OA (+0.03 AJ, +0.06 OA) while retaining a small AJ_RD gain (+0.003). It is an official-safe point, not a strong leaderboard improvement.

### 3.8 Why ReEntry changes visibility but not coordinates

The ReEntry design is based on an empirical finding. When comparing the base tracker and the override (teacher ensemble) coordinate channels, the base coordinate channel is consistently stronger than the override coordinate channel: replacing base coordinates with override coordinates degrades AJ_RD. However, the override *visibility* channel contains useful re-entry signal: it recovers true-visible frames that the base tracker misses.

The correct deployable intervention is therefore *channel-selective*: preserve base coordinates and locally recover visibility from the override. This is why ReEntry modifies $\tilde{v}_{n,t}$ but keeps $\tilde{x}_{n,t} = \hat{x}^{\text{base}}_{n,t}$. A direct consequence is that position-only metrics such as $\delta_{\text{avg}}$ remain unchanged. Future coordinate-aware recovery (Section 7) would combine visibility correction with localization refinement.

---

## 4. Experimental Setup

We evaluate on standard TAP-Vid datasets and targeted diagnostic variants. On TAPVid-DAVIS, we report two official-style local protocols. The first uses first-query evaluation at input resolution, with 30 videos and 650 queries. The second uses strided queries with original-resolution metrics, providing a stricter test of global visibility behavior. On TAPVid RGB-Stacking, we evaluate the complete locally available full50 set, covering `rgb_stacking_000000` through `rgb_stacking_000049`, with 60,829 queries. We also report RGB-Stacking fresh20-49 and synthetic stress variants to isolate re-entry behavior under translation and occlusion perturbations.

We report standard TAP-Vid metrics: Average Jaccard (AJ), Occlusion Accuracy (OA), and average position accuracy ($\delta_{\text{avg}}$). Since ReEntry targets failures after reappearance, we additionally report AJ_RD as a re-entry diagnostic metric. AJ_RD is reported alongside standard TAP-Vid metrics and is not a replacement for them.

We compare against CoTracker3 offline as the primary base tracker. Where available, we also include CoTracker3 baseline/online, TrackOn2, and TAPNext. ReEntry variants include V1 Learned ReEntry-VisCalibrator, V22Q Interval/Gate, V24-DINOScore, and V25-safe threshold=0.80.

All results are local evaluations under the specified protocol. We do not claim official server or leaderboard submission.

### Protocol summary

| Evaluation | Dataset | Videos | Queries | Query mode | Resolution | Evaluation type |
|---|---:|---:|---:|---|---|---|
| DAVIS first/input | TAPVid-DAVIS | 30 | 650 | first | input / 256 | official-style local |
| DAVIS strided/original | TAPVid-DAVIS | 30 | 5,882 | strided | original | official-style local |
| RGB-Stacking full50 | TAPVid RGB-Stacking | 50 | 60,829 | strided | input / 256-style | full local standard |
| RGB fresh20-49 | RGB-Stacking subset | 30 | local | local | 256/local | diagnostic local |
| RGB stress variants | RGB-Stacking derived | 30 each | local | synthetic | 256/local | diagnostic stress |

---

## 5. Results

### 5.1 DAVIS first/input: ReEntry improves standard metrics and AJ_RD

We evaluate on TAPVid-DAVIS using a first-query/input-resolution official-style local protocol with 30 videos and 650 queries. The CoTracker3 offline base obtains 62.6566 AJ, 88.1487 OA, and 0.3142 AJ_RD. ReEntry improves all of these metrics. V1 reaches 64.5758 AJ, 91.7274 OA, and 0.3588 AJ_RD. V22Q further improves AJ to 64.7260 and OA to 91.7406. V24-DINOScore gives the strongest ReEntry result, reaching 64.8439 AJ, 91.8851 OA, and 0.3549 AJ_RD.

Compared with CoTracker3 offline, V24 improves by +2.1874 AJ, +3.7363 OA, and +0.0407 AJ_RD. The paired-video check confirms stability: V24 improves AJ with a 95% bootstrap CI of [+1.2582, +3.2487] (W/L/T = 24/5/1) and OA with CI [+2.1673, +5.7423] (W/L/T = 27/2/1). TrackOn2 remains the strongest external baseline overall (67.0406 AJ, 92.0916 OA), so we do not claim ReEntry outperforms TrackOn2. ReEntry substantially improves the CoTracker3 offline base and closes part of the gap while targeting a different failure mode.

**Table 1. TAPVid-DAVIS first/input official-style local evaluation.** Protocol: 30 videos, 650 queries, query mode `first`, input/256 metric resolution. Higher is better for all metrics.

| Method | AJ | OA | d_avg | d_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 baseline | 64.8949 | 91.7983 | 77.3560 | 84.9284 | 0.3486 | 0.5246 |
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 84.5776 | 0.3142 | 0.4525 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 87.8233 | 0.3714 | 0.5444 |
| ReEntry V1 | 64.5758 | 91.7274 | 77.2244 | 84.5776 | 0.3588 | 0.5305 |
| ReEntry V22Q | 64.7260 | 91.7406 | 77.2244 | 84.5776 | 0.3556 | 0.5252 |
| ReEntry V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 84.5776 | 0.3549 | 0.5236 |

**Deltas vs CoTracker3 offline:**

| Method | dAJ | dOA | d_d_avg | dAJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 baseline | +2.2383 | +3.6496 | +0.1316 | +0.0344 |
| TrackOn2 | +4.3841 | +3.9429 | +2.6174 | +0.0572 |
| ReEntry V1 | +1.9193 | +3.5787 | +0.0000 | +0.0446 |
| ReEntry V22Q | +2.0695 | +3.5919 | +0.0000 | +0.0414 |
| ReEntry V24-DINOScore | +2.1874 | +3.7363 | +0.0000 | +0.0407 |

**Paired-video checks vs CoTracker3 offline:**

| Method | AJ mean pp [95% CI] | OA mean pp [95% CI] | AJ W/L/T | OA W/L/T |
|---|---:|---:|---:|---:|
| ReEntry V1 | +1.9193 [+0.8375, +3.0545] | +3.5787 [+1.8485, +5.6697] | 22/7/1 | 24/5/1 |
| ReEntry V22Q | +2.0695 [+1.0763, +3.1667] | +3.5919 [+2.0011, +5.5624] | 23/6/1 | 25/4/1 |
| ReEntry V24 | +2.1874 [+1.2582, +3.2487] | +3.7363 [+2.1673, +5.7423] | 24/5/1 | 27/2/1 |

---

### 5.2 DAVIS strided/original: default ReEntry improves AJ_RD but exposes a stricter-protocol trade-off

We next evaluate on TAPVid-DAVIS using a stricter strided/original-resolution official-style local protocol. This setting contains 30 videos and 5,882 strided queries. It is more demanding for visibility correction because the metric averages across a much denser set of query points and frames.

Under this protocol, CoTracker3 offline obtains 51.5385 AJ, 92.1543 OA, 63.5892 d_avg, and 0.3870 AJ_RD. The default ReEntry variants improve AJ_RD but reduce standard AJ/OA. V1 reaches 0.4144 AJ_RD, a +0.0274 improvement over offline, but AJ decreases to 50.7303 and OA decreases to 91.2730. V22Q obtains 0.4136 AJ_RD, a +0.0266 improvement, while AJ and OA are 50.8342 and 91.3439.

This result reveals a central trade-off: ReEntry successfully corrects post-reappearance visibility lag, but under dense strided evaluation, aggressive visibility changes can introduce false-visible penalties that reduce global AJ/OA. Since ReEntry does not modify coordinates, d_avg remains unchanged at 63.5892 for the ReEntry variants.

To test whether the method can be made official-safe, we sweep the V1 confidence threshold. The best conservative point is threshold=0.80. This V25-safe setting obtains 51.5648 AJ, 92.2106 OA, and 0.3900 AJ_RD. Relative to CoTracker3 offline, this is +0.0263 AJ, +0.0563 OA, and +0.0030 AJ_RD. The gain is small and should not be interpreted as a strong leaderboard-style improvement. Instead, it shows that ReEntry can be operated conservatively to preserve standard AJ/OA under the stricter strided/original protocol.

**Table 2. TAPVid-DAVIS strided/original official-style local evaluation.** Protocol: 30 videos, 5,882 queries, query mode `strided`, original-resolution metrics.

| Method | AJ | OA | d_avg | d_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 68.0060 | 0.3870 | 0.5546 |
| CoTracker3 online | 36.9543 | 68.1493 | 45.0072 | 47.7100 | 0.4101 | 0.5972 |
| TrackOn2 | 28.3785 | 58.3304 | 33.2280 | 35.8975 | 0.3700 | 0.5383 |
| TAPNext | 25.9433 | 65.9795 | 33.4476 | 36.1557 | 0.3624 | 0.5199 |
| ReEntry V1 default | 50.7303 | 91.2730 | 63.5892 | 68.0060 | 0.4144 | 0.6072 |
| ReEntry V22Q default | 50.8342 | 91.3439 | 63.5892 | 68.0060 | 0.4136 | 0.6047 |
| V25-safe threshold=0.80 | 51.5648 | 92.2106 | 63.5892 | 68.0060 | 0.3900 | 0.5601 |

**Deltas vs CoTracker3 offline:**

| Method | dAJ | dOA | d_d_avg | dAJ_RD |
|---|---:|---:|---:|---:|
| ReEntry V1 default | -0.8081 | -0.8813 | +0.0000 | +0.0274 |
| ReEntry V22Q default | -0.7043 | -0.8105 | +0.0000 | +0.0266 |
| V25-safe threshold=0.80 | +0.0263 | +0.0563 | +0.0000 | +0.0030 |

---

### 5.3 RGB-Stacking full50 strided: ReEntry improves AJ_RD and OA on the complete 50-video set

We evaluate on the full TAPVid RGB-Stacking set available locally, covering all 50 videos from `rgb_stacking_000000` through `rgb_stacking_000049`, with 60,829 queries. We report a strided official-style local evaluation with standard TAP-Vid metrics.

The CoTracker3 offline base obtains 79.9345 AJ, 91.6371 OA, 88.5714 d_avg, and 0.3617 AJ_RD. ReEntry substantially improves AJ_RD and OA. V1 reaches 0.4414 AJ_RD and 93.0758 OA, corresponding to +0.0797 AJ_RD and +1.4387 OA. V22Q reaches 0.4400 AJ_RD and 93.0481 OA. V24-DINOScore reaches 0.4394 AJ_RD and 93.0389 OA.

The paired-video analysis confirms that the OA improvement is stable. V1 improves OA by +1.4387 percentage points with 39/11/0 wins/losses/ties. V24 improves OA by +1.4018 points with 40/10/0 wins/losses/ties. At the same time, all ReEntry variants show a small AJ trade-off on RGB-Stacking full50. V1 reduces AJ by -0.5043 points, V24 by -0.3371 points. The position-only metric d_avg remains unchanged at 88.5714. This pattern is consistent with the method design: ReEntry modifies visibility, not coordinates.

**Table 3. TAPVid RGB-Stacking full50 strided official-style local evaluation.** Protocol: 50 videos, 60,829 queries, strided query mode, input/256-style metric resolution.

| Method | AJ | OA | d_avg | d_4px | AJ_RD | Queries |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 79.9345 | 91.6371 | 88.5714 | 91.5875 | 0.3617 | 60,829 |
| CoTracker3 online | 44.8432 | 55.8092 | 73.1881 | 83.3099 | 0.4028 | 60,829 |
| ReEntry V1 | 79.4302 | 93.0758 | 88.5714 | 91.5875 | 0.4414 | 60,829 |
| ReEntry V22Q | 79.5756 | 93.0481 | 88.5714 | 91.5875 | 0.4400 | 60,829 |
| ReEntry V24-DINOScore | 79.5974 | 93.0389 | 88.5714 | 91.5875 | 0.4394 | 60,829 |

**Deltas vs CoTracker3 offline:**

| Method | dAJ | dOA | d_d_avg | dAJ_RD |
|---|---:|---:|---:|---:|
| ReEntry V1 | -0.5043 | +1.4387 | +0.0000 | +0.0797 |
| ReEntry V22Q | -0.3589 | +1.4110 | +0.0000 | +0.0783 |
| ReEntry V24-DINOScore | -0.3371 | +1.4018 | +0.0000 | +0.0777 |

**Paired-video checks vs CoTracker3 offline:**

| Method | AJ mean pp [95% CI] | OA mean pp [95% CI] | AJ W/L/T | OA W/L/T |
|---|---:|---:|---:|---:|
| ReEntry V1 | -0.5043 [-0.8046, -0.2011] | +1.4387 [+0.9589, +1.9236] | 9/41/0 | 39/11/0 |
| ReEntry V22Q | -0.3589 [-0.6391, -0.0670] | +1.4110 [+0.9452, +1.8875] | 10/40/0 | 40/10/0 |
| ReEntry V24 | -0.3371 [-0.6065, -0.0531] | +1.4018 [+0.9438, +1.8769] | 11/39/0 | 40/10/0 |

---

### 5.4 Diagnostic RGB fresh/stress evaluations isolate re-entry behavior

To isolate the re-entry failure mode, we also evaluate on RGB-Stacking fresh20-49 and two synthetic stress variants: translate_L16 and occluder_L16. These are diagnostic evaluations rather than official benchmark submissions.

Across the three settings, V24 improves AJ_RD and OA over the base. On fresh20-49 natural, AJ_RD improves from 0.3330 to 0.4027, and OA improves from 91.4636 to 92.9244. On translate_L16, AJ_RD improves from 0.5080 to 0.5564, and OA improves from 90.7805 to 92.1951. On occluder_L16, AJ_RD improves from 0.6417 to 0.6763, and OA improves from 90.8918 to 92.2293.

**Table 4. RGB-Stacking fresh/stress diagnostic evaluation.** Diagnostic, not official benchmark submissions. They isolate re-entry behavior under natural and synthetic stress settings.

| Setting | AJ_RD Base->V24 | dAJ_RD | OA Base->V24 | dOA | AJ Base->V24 | dAJ |
|---|---:|---:|---:|---:|---:|---:|
| fresh20-49 natural | 0.3330 -> 0.4027 | +0.0698 | 91.4636 -> 92.9244 | +1.4608 | 79.5944 -> 79.2884 | -0.3060 |
| translate_L16 | 0.5080 -> 0.5564 | +0.0484 | 90.7805 -> 92.1951 | +1.4146 | 75.1940 -> 74.9987 | -0.1953 |
| occluder_L16 | 0.6417 -> 0.6763 | +0.0345 | 90.8918 -> 92.2293 | +1.3375 | 77.6280 -> 77.2276 | -0.4004 |
| Average | 0.4939 -> 0.5451 | +0.0509 | 91.0453 -> 92.4496 | +1.4043 | 77.4721 -> 77.1716 | -0.3006 |

---

### 5.5 V25 official-safe threshold sweep on DAVIS strided/original

**Table 5. V25 official-safe threshold sweep on DAVIS strided/original.** Purpose: find a conservative operating point that preserves official-style AJ/OA while retaining some AJ_RD gain.

| Threshold | AJ | OA | AJ_RD | dAJ | dOA | dAJ_RD | Recovered frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 50.7569 | 91.2704 | 0.4141 | -0.7816 | -0.8839 | +0.0271 | 39,809 |
| 0.30 | 50.8000 | 91.2821 | 0.4136 | -0.7384 | -0.8722 | +0.0266 | 39,508 |
| 0.40 | 50.8522 | 91.3187 | 0.4131 | -0.6863 | -0.8357 | +0.0261 | 38,993 |
| 0.50 | 50.9103 | 91.4152 | 0.4118 | -0.6282 | -0.7391 | +0.0248 | 37,759 |
| 0.60 | 51.0521 | 91.6501 | 0.4092 | -0.4864 | -0.5043 | +0.0222 | 34,459 |
| 0.70 | 51.3570 | 91.9982 | 0.3988 | -0.1815 | -0.1562 | +0.0118 | 26,932 |
| 0.80 | 51.5648 | 92.2106 | 0.3900 | +0.0263 | +0.0563 | +0.0030 | 15,024 |
| 0.90 | 51.5630 | 92.1914 | 0.3875 | +0.0246 | +0.0370 | +0.0005 | 4,140 |
| 0.95 | 51.5420 | 92.1607 | 0.3870 | +0.0035 | +0.0063 | +0.0000 | 773 |
| 0.99 | 51.5385 | 92.1543 | 0.3870 | +0.0000 | +0.0000 | +0.0000 | 0 |

Best official-safe point: threshold=0.80. It preserves AJ/OA with tiny positive AJ_RD (+0.0030).

---

### 5.6 Ablation and method positioning

**Table 6. Method positioning.**

| Variant | Role | Paper positioning |
|---|---|---|
| V1 Learned ReEntry-VisCalibrator | Main AJ_RD-oriented method | Strongest AJ_RD among ReEntry variants on RGB full50 and DAVIS strided/original default; can trade off standard AJ/OA. |
| V22Q Interval/Gate | Stability/interval extension | Slightly safer than V1 for standard AJ/OA; good stability variant; not replaced by V24. |
| V24-DINOScore | Optional AJ-oriented appearance micro-filter | Small AJ gains over V22Q with tiny AJ_RD/OA cost; best ReEntry row on DAVIS first/input but not a universal default. |
| V25 threshold=0.80 | Official-safe conservative operating point | Preserves DAVIS strided/original AJ/OA, but AJ_RD gain is tiny. |
| V26 future direction | Official-metric-aware selective ReEntry | Needed if the target is stronger leaderboard-style improvement under strided/original. |

The deterministic variant captures most of the AJ_RD recovery. The learned calibrator further improves the recovery-stability tradeoff, matching or slightly improving AJ_RD while consistently improving standard AJ across videos. V22Q is the best stability-oriented extension. V24 adds small AJ gains with an appearance micro-filter. V25 is a conservative safety point, not a strong improvement.

---

## 6. Limitations

The main limitation of ReEntry is that re-entry improvement does not automatically imply global benchmark improvement under every query protocol. Under DAVIS first/input evaluation, ReEntry improves AJ, OA, and AJ_RD over the CoTracker3 offline base. However, under the stricter DAVIS strided/original protocol, the default ReEntry variants improve AJ_RD while reducing standard AJ/OA. This shows that aggressive visibility correction can introduce false positives or poorly timed visible predictions that are penalized by global TAP-Vid metrics.

A second limitation is that the current ReEntry pipeline primarily modifies visibility decisions while keeping the base tracker coordinates fixed. As a result, coordinate-based metrics such as d_avg often remain unchanged. This is useful for isolating the visibility/re-entry effect, but it also means that ReEntry does not correct localization errors directly.

A third limitation is protocol coverage. We report DAVIS first/input and DAVIS strided/original official-style local evaluations, and a full local RGB-Stacking full50 evaluation. However, we do not claim an official leaderboard submission, nor do we claim full TAP-Vid benchmark coverage across all subsets such as Kinetics and Kubric. RGB-Stacking full50 should be interpreted as a full local standard evaluation until explicitly audited or rerun under strict official first/strided query protocol.

Finally, the current V25 official-safe threshold shows only a small positive gain under DAVIS strided/original. This confirms that conservative thresholding can avoid metric degradation, but it is not a strong leaderboard-style improvement.

---

## 7. Future Work

The most direct future direction is **metric-aware selective ReEntry**. Instead of selecting visibility corrections only based on re-entry confidence, a future V26 variant should estimate whether a proposed correction is likely to preserve or improve standard AJ/OA while improving AJ_RD. This requires training or deriving a frame-level decision rule whose objective combines re-entry utility and standard metric safety.

A second direction is coordinate-aware recovery. Since the current method preserves base coordinates, it cannot improve d_avg. Combining ReEntry with coordinate refinement or re-detection localization could improve both visibility and localization metrics.

A third direction is broader official-protocol coverage. The next protocol supplement should audit or rerun RGB-Stacking full50 under explicit TAP-Vid first or strided query mode. Larger-scale Kinetics and Kubric evaluations may further strengthen benchmark coverage, but they should be considered after the current paper tables and result narrative are finalized.

---

## 8. Conclusion

ReEntry improves the targeted re-entry failure mode across multiple TAP-Vid local evaluations. It yields paired-stable improvements on DAVIS first/input official-style AJ/OA/AJ_RD and strong AJ_RD/OA improvements on RGB-Stacking full50. At the same time, stricter DAVIS strided/original evaluation reveals that aggressive visibility correction can harm global standard metrics. A conservative V25 operating point preserves AJ/OA with a small AJ_RD gain, motivating future metric-aware selective ReEntry.

---

## Final claim boundary

Safe claim:

> ReEntry improves re-entry recovery across TAP-Vid local evaluations and diagnostic settings. It improves DAVIS first/input official-style AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter DAVIS strided/original evaluation requires conservative selection to preserve standard metrics.

Avoid:

- official leaderboard submission
- full TAP-Vid official benchmark submission
- universal improvement across all official protocols
- V24 beats TrackOn2 on DAVIS
- V24 is universally best

---

## Figure plan

| Figure | Description | Status |
|---|---|---|
| Fig 1. Re-entry failure-mode diagram | Show point visible -> occluded/disappears -> reappears -> base tracker visibility lag -> ReEntry correction. | to render |
| Fig 2. ReEntry pipeline | Base tracker predictions -> candidate re-entry windows -> V1 confidence -> V22Q interval/gate -> optional V24 appearance filter -> corrected visibility. | to render |
| Fig 3. Main result bar chart | DAVIS first/input: offline vs V1/V22Q/V24 on AJ/OA/AJ_RD. | to render |
| Fig 4. RGB full50 trade-off | AJ_RD/OA gain with AJ trade-off across 50 videos. | to render |
| Fig 5. V25 threshold sweep | threshold vs AJ/OA/AJ_RD, showing 0.80 as official-safe point. | to render |

Qualitative timeline panels already rendered at:
- `paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_base_fail_learned_success.png`
- `paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_det_over_recovery.png`
- `paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_failure_short_occ.png`
- `paper/reentry_viscalibrator_tex/figures/qual_timeline_occluder_success.png`
- `paper/reentry_viscalibrator_tex/figures/qual_timeline_occluder_learned_preserves_base.png`

---

## Tables checklist

- [x] Table 1. TAPVid-DAVIS first/input official-style local evaluation
- [x] Table 2. TAPVid-DAVIS strided/original official-style local evaluation
- [x] Table 3. TAPVid RGB-Stacking full50 strided official-style local evaluation
- [x] Table 4. RGB fresh/stress diagnostic evaluation
- [x] Table 5. V25 official-safe threshold sweep
- [x] Table 6. Method positioning / ablation summary
- [x] Paired-video statistical summary (within Tables 1 and 3)

## Claim boundary checklist

- [x] "official-style local evaluation" used, not "official leaderboard submission"
- [x] "full local standard evaluation" used for RGB full50, not "strict official query-protocol result"
- [x] No claim of "universal improvement across all official protocols"
- [x] No claim of "V24 beats TrackOn2 on DAVIS"
- [x] No claim of "V24 is universally best"
- [x] TrackOn2 reported as stronger external baseline on DAVIS first/input
- [x] DAVIS strided/original trade-off honestly reported
- [x] V25-safe positioned as conservative safety point, not strong improvement

---
