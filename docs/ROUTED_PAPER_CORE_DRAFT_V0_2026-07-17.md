# Route-D Paper Core Draft V0

Date: 2026-07-17
Status: component-paper draft; submission packaging is held pending strong-backbone validation; numerical authority remains the corrected official-scale audit
Target: build toward at least CCF-B / CAS Zone 2 without overstating current evidence

## 0. Working identity

### Preferred working title

**Risk-Calibrated Sparse Feedback Routing for Online Point Tracking**

### Alternative titles

1. **Route-D: Multi-Threshold Risk-Aware Candidate Routing for Online Point Tracking**
2. **When Should an Online Point Tracker Relocalize? Risk-Calibrated Sparse Closed-Loop Routing**
3. **Selective Global Correction with Closed-Loop Feedback for Tracking Any Point**

### One-sentence thesis

A frozen online point tracker can benefit from global relocalization candidates without accepting them indiscriminately: a Kubric-trained controller that predicts multi-threshold candidate utility, calibrates action risk, and sparsely feeds accepted corrections back into tracker state improves official-scale TAP metrics on a preregistered external Kinetics evaluation.

### Correct paper identity

```text
Method type:
  inference-time candidate-routing controller for an online point tracker

Core decision:
  keep the local hypothesis or accept one global relocalization hypothesis

Core learning signal:
  candidate correctness at multiple pixel thresholds

Core safety mechanism:
  calibrated beneficial-action probability plus profile constraints

Core temporal mechanism:
  accepted corrections update the next-frame prior and tracker commit state
```

The paper is not currently a universal TAP tracker or a leaderboard/SOTA paper. It is a selective-intervention paper about when and how a causal tracker should accept global corrections.

---

## 1. Claim boundary

### Claims currently supported

1. A controller trained and calibrated only on disjoint Kubric partitions improves a frozen MMP tracker on an exact order-preserving local materialization of 1,144 of the 1,147 uniquely annotated video segments in the byte-verified official TAP-Vid-Kinetics release CSV.
2. Under the frozen corrected official-scale protocol, closed-loop Route-D improves both Average Jaccard and average point-threshold accuracy over an independent baseline, with paired-video 95% bootstrap confidence intervals strictly above zero.
3. Closed-loop Route-D also improves both metrics over the corresponding open-loop intervention, showing that state feedback contributes beyond framewise candidate replacement.
4. The intervention is sparse: the corrected full-run global selection rate is approximately 4.02%.
5. The aggregate gain is not a per-video safety guarantee; substantial negative cases remain.

### Claims not currently supported

```text
- state of the art on TAP-Vid;
- universal improvement across datasets, trackers, or videos;
- a model-agnostic plug-in demonstrated on multiple backbones;
- a formal no-harm or selective-risk guarantee;
- an official leaderboard submission;
- evaluation on every release-CSV segment;
- superiority over TAPNext++, Track-On2, ReTracker, CoTracker3, or other modern trackers;
- Kinetics-specific stability tuning.
```

### Required official scope sentence

> Exact order-preserving local materialization of 1,144 of the 1,147 uniquely annotated video segments in the byte-verified official TAP-Vid-Kinetics release CSV, evaluated with pinned official metric formulas and a frozen controller. Three release-CSV segments were not materialized.

### Current publication decision

The corrected Kinetics result passes its preregistered statistical gate, but the resulting full system is not yet competitive enough to support a strong benchmark paper:

```text
baseline AJ:                    32.49
Route-D closed-loop AJ:         34.80
absolute gain:                  +2.30 AJ points
relative gain:                  approximately +7.09%

relative improvement:           strong
paired statistical evidence:    strong
open/closed mechanism evidence: strong
absolute tracker performance:   weak
modern-baseline competitiveness:insufficient
```

The paper must therefore remain a component-level selective-routing result until Route-D is validated prospectively on a strong backbone. The decisive next question is whether the gain survives when the native tracker itself is close to a modern published reference.

---

## 2. Draft abstract

Online point trackers must balance two conflicting hypotheses at every frame. Local tracking preserves temporal continuity but can drift or fail after large motion, whereas global relocalization can recover the target but may introduce destructive jumps. Existing trackers typically learn this behavior inside an end-to-end architecture or apply fixed matching heuristics. We instead study global correction as a selective action over a frozen tracker. We propose Route-D, a risk-calibrated candidate-routing controller that estimates each local and global hypothesis's correctness profile at multiple pixel thresholds, ranks global candidates by coarse utility, and accepts a correction only when a calibrated action model and explicit fine-to-coarse profile constraints agree. Accepted corrections are sparsely fed back into the subsequent tracking prior and commit state, yielding a true closed-loop intervention. All scorer, tree, calibration, and policy choices are fitted using disjoint Kubric-only partitions and frozen before external evaluation. On an exact order-preserving local materialization of 1,144 of the 1,147 uniquely annotated segments in the byte-verified official TAP-Vid-Kinetics release CSV, Route-D improves Average Jaccard by 0.023043 and average point-threshold accuracy by 0.027530 over an independent baseline. Paired-video bootstrap intervals are [0.020567, 0.025569] and [0.024800, 0.030260], respectively. Closed-loop feedback also significantly improves over the same controller applied open loop. The controller selects a global correction on only 4.02% of active decisions, but failures remain heterogeneous and can be severe. These results support risk-aware sparse feedback routing as a promising complement to end-to-end online point-tracking architectures, while motivating stronger cross-backbone and stability validation.

---

## 3. Draft introduction

Tracking Any Point requires estimating the trajectory and visibility of an arbitrary queried surface point over a video. The problem is especially difficult in an online setting, where predictions must be produced causally and a tracker cannot use future frames to repair earlier mistakes. Recent systems address this challenge with recurrent token prediction, temporal refinement, global receptive fields, or dedicated memory. Despite these advances, a causal tracker still faces a recurring decision: whether to preserve a temporally consistent local estimate or replace it with a potentially corrective global match.

This decision is asymmetric. Retaining the local hypothesis may preserve a drifted trajectory, but accepting an incorrect global candidate can create a much larger discontinuity. The damage may extend beyond one frame because the selected point can influence the next-frame search center, temporal prior, confidence state, and memory. Consequently, candidate selection should not be treated only as framewise ranking. It is a risk-sensitive control decision whose consequences unfold through the tracker state.

We study this problem without retraining the base tracker. At each active frame, the frozen tracker exposes one local candidate and a set of global relocalization candidates. A learned multi-threshold scorer predicts, for every candidate, the probability of falling within 1, 2, 4, 8, and 16 pixels of the target. These predictions form a candidate utility profile rather than a single confidence score. Route-D first identifies the global candidate with the strongest coarse-threshold profile. It then constructs an action representation comparing that candidate against the local hypothesis, including predicted threshold profiles, raw candidate diagnostics, and local-global margins. A calibrated ExtraTrees controller estimates whether the global action is beneficial, and a frozen policy accepts the action only when its calibrated probability and fine-to-coarse utility constraints pass.

The accepted action can be evaluated in two ways. In open-loop mode, the correction changes only the reported point at the current frame. In closed-loop mode, the selected point also updates the subsequent prior and tracker commit state. This distinction isolates a central question: does a correction help merely because one candidate is better at the current frame, or because the corrected state prevents future error propagation? Our full external evaluation shows statistically significant gains for closed loop over both the independent baseline and open loop.

We enforce a strict evidence protocol. The multi-threshold scorer, tree configuration, isotonic calibration, and action policy are fitted using disjoint partitions of a causal Kubric cache. The controller is frozen before the final Kinetics evaluation, and no Kinetics observation is used to modify the scorer, controller, policy, fusion, or guard. We pin the official TAP-Vid metric implementation and evaluate an exact order-preserving local materialization of 1,144 of 1,147 uniquely annotated segments from the byte-verified official release CSV.

Our current contributions are:

1. **Multi-threshold action representation.** We model local and global hypotheses through predicted correctness profiles at multiple spatial thresholds, preserving the difference between fine localization and coarse recovery.
2. **Risk-calibrated sparse routing.** We introduce a calibrated action gate with explicit profile constraints that decides when a global candidate should replace the local candidate.
3. **Open-loop versus closed-loop causal evaluation.** We separate framewise candidate replacement from state-feedback effects and show that closed-loop feedback provides an additional statistically resolved gain.
4. **Frozen cross-domain evidence with audited scope.** We provide a preregistered, official-scale Kinetics evaluation with exact package lineage, paired-video uncertainty, undefined-metric handling, and explicit severe-failure reporting.

The current evidence establishes a strong component-level result for the frozen MMP tracker. It does not yet establish broad cross-backbone generality or state-of-the-art TAP performance.

The next decisive validation is Route-D on CoTracker3 online true-streaming under an independently preregistered protocol. That integration must first pass native parity, deterministic candidate export, coordinate/raster/query checks, and isolated closed-loop state-writeback tests. Kinetics remains frozen and cannot be used to choose the new adapter or controller.

---

## 4. Related-work positioning

### 4.1 Matching and trajectory refinement

TAPIR independently initializes framewise matches and then refines trajectories using temporal and local-correlation reasoning. LocoTrack improves matching with local all-pair correspondence. These methods primarily strengthen the candidate-generation and trajectory-estimation architecture. Route-D instead assumes that local and global candidates already exist and learns a selective action over them.

### 4.2 Online recurrent and memory-based tracking

TAPNext formulates online TAP as recurrent next-token prediction. Track-On and Track-On2 use causal spatial/context memory for long-term online tracking. SPOT also propagates information with streaming memory. Route-D must therefore not claim novelty for online operation, memory, or causal recurrence. Its distinct focus is whether a frozen causal tracker should accept a global correction, how that decision should reflect multiple localization scales, and how the accepted action affects future state.

### 4.3 Global re-detection and long-term recovery

ReTracker uses global-receptive-field matching and two-view pretraining to improve online recovery after long occlusion and viewpoint change. TAPNext++ explicitly identifies re-detection as a blind spot and improves it through long-sequence training and tailored augmentation. Route-D is complementary: it does not introduce a new global matcher or new training corpus; it learns a sparse risk-aware controller over existing local/global hypotheses.

### 4.4 Training and real-data scaling

BootsTAP and CoTracker3 improve point tracking through real-video pseudo-labeling or bootstrapped training. These works target representation and training quality. Route-D is an inference-time decision layer trained on synthetic candidate evidence and frozen before external transfer.

### 4.5 Precise novelty sentence

> Unlike end-to-end online trackers that internalize recovery through architecture, memory, or training, Route-D exposes global correction as an explicit risk-sensitive action: it predicts multi-threshold utility profiles for local and global hypotheses, calibrates whether the best global candidate is beneficial, and evaluates the resulting intervention both without and with feedback into future tracker state.

### Primary literature anchors

- TAPIR: https://openaccess.thecvf.com/content/ICCV2023/html/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.html
- LocoTrack: https://eccv.ecva.net/virtual/2024/poster/2114
- BootsTAP: https://arxiv.org/abs/2402.00847
- CoTracker3: https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html
- Track-On: https://proceedings.iclr.cc/paper_files/paper/2025/hash/8fd10360ac9de462ffd36155d87ee999-Abstract-Conference.html
- Track-On2: https://arxiv.org/abs/2509.19115
- TAPNext: https://openaccess.thecvf.com/content/ICCV2025/html/Zholus_TAPNext_Tracking_Any_Point_TAP_as_Next_Token_Prediction_ICCV_2025_paper.html
- ReTracker: https://openaccess.thecvf.com/content/ICCV2025/html/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.html
- SPOT: https://openaccess.thecvf.com/content/ICCV2025/html/Dong_Online_Dense_Point_Tracking_with_Streaming_Memory_ICCV_2025_paper.html
- TAPNext++: https://openaccess.thecvf.com/content/CVPR2026F/html/Jung_TAPNext_Whats_Next_for_Tracking_Any_Point_TAP_CVPRF_2026_paper.html

---

## 5. Method

### 5.1 Online candidate set

For query point `i` and active frame `t`, the frozen tracker produces a candidate set

```text
C_it = {c_it^0, c_it^1, ..., c_it^(K-1)}.
```

Candidate `c_it^0` is the local hypothesis. Candidates with index `k > 0` are global relocalization hypotheses. Route-D does not alter candidate generation.

Each candidate has a normalized two-dimensional point and twelve prediction-only features:

```text
1. candidate quality
2. candidate entropy
3. distance from local candidate
4. distance from previous prior
5. quality gap from local
6. previous-prior confidence
7. global-candidate indicator
8. normalized candidate rank
9. exp(-distance from local)
10. exp(-distance from previous prior)
11. indicator(candidate quality > local quality)
12. constant bias channel
```

No ground truth is used at inference.

### 5.2 Multi-threshold correctness profile

A frozen neural scorer maps each candidate feature vector to logits at thresholds

```text
R = {1, 2, 4, 8, 16} pixels.
```

After a sigmoid, candidate `k` receives the profile

```text
q_it^k = [q_it^k(1), q_it^k(2), q_it^k(4), q_it^k(8), q_it^k(16)],
```

where `q_it^k(r)` estimates the probability that the candidate lies within `r` pixels of the target.

Training uses threshold-hit labels derived from candidate error on Kubric. The scorer objective combines:

```text
- threshold-wise binary cross entropy;
- monotonicity regularization across thresholds;
- a global-versus-local utility gate objective;
- ranking among global candidates;
- additional weight on harmful false-positive global actions.
```

The profile retains information lost by a single scalar confidence. A global candidate may improve coarse recovery while being less reliable at one-pixel precision.

### 5.3 Best global hypothesis

The best global candidate is ranked by its average predicted correctness excluding the strictest one-pixel head:

```text
g_it = argmax_{k > 0} mean(q_it^k(2), q_it^k(4), q_it^k(8), q_it^k(16)).
```

This coarse ranking favors candidates capable of meaningful recovery without allowing the one-pixel head alone to dominate selection.

### 5.4 Action representation

Route-D compares the best global candidate against the local hypothesis. Its action feature vector concatenates:

```text
- local multi-threshold profile;
- best-global multi-threshold profile;
- profile difference;
- local raw candidate features;
- best-global raw candidate features;
- raw-feature difference;
- one-pixel probability margin;
- coarse-threshold utility gain;
- total multi-threshold utility gain.
```

The action target on Kubric is positive when the best predicted global candidate has strictly higher realized average threshold utility than the local candidate.

### 5.5 Tree risk controller and calibration

An ExtraTrees classifier predicts whether accepting the best global candidate is beneficial. Tree configuration selection, fitting, isotonic calibration, and policy calibration use disjoint partitions of the causal Kubric training cache.

The frozen policy is:

```text
calibrated beneficial-action probability >= 0.6298701167
one-pixel probability margin             >= -0.05
coarse-threshold predicted gain          >=  0.01
total predicted gain                     >= -0.05
```

Otherwise Route-D falls back to the local candidate.

The tree family was selected after nested audits showed that the existing action feature vector retained useful decision information that small neural action controllers did not fully recover. This is an empirical model-family choice, not a claim that trees are universally superior.

### 5.6 Sparse intervention

Let `a_it` be the accepted candidate index. Route-D outputs

```text
a_it = g_it, if all calibrated policy conditions pass;
a_it = 0, otherwise.
```

The selected point is

```text
p_hat_it = c_it^(a_it).
```

The current frozen controller performs hard selection. Distance guards and soft fusion exist in the code path but are not active components of the corrected external result and must not be presented as part of the final method.

### 5.7 Open-loop and closed-loop modes

**Open loop.** Route-D changes the current reported coordinate but does not let the intervention alter subsequent tracker state.

**Closed loop.** An accepted global coordinate replaces the current point and is also written into the commit/prior pathway used by later frames. Therefore, a sparse correction can change a long suffix of the trajectory.

The comparison between these modes is a mechanism test:

```text
closed-loop gain over open-loop
= downstream value of feeding accepted corrections into tracker state.
```

### 5.8 Frozen training and evaluation separation

All controller fitting uses Kubric-only partitions. The final Kinetics protocol freezes:

```text
- base checkpoint;
- multi-threshold scorer;
- tree controller;
- isotonic calibration;
- policy thresholds;
- hard-selection behavior;
- query mode;
- input and metric resolution;
- official coordinate conversion;
- bootstrap seed and resample count.
```

No Kinetics result is used to tune any component.

---

## 6. Experimental protocol

### 6.1 External evaluation scope

```text
Dataset package:
  official TAP-Vid-Kinetics release CSV

Official unique annotation groups:
  1,147

Exact order-preserving local materialization:
  1,144

Unmaterialized release-CSV segments:
  3

Query mode:
  first

Input and metric raster:
  256 x 256

Coordinate contract:
  x * width, y * height

Bootstrap:
  20,000 paired-video resamples, seed 17
```

The three unmaterialized segments are:

```text
BBSK3Wv0jXM_000112_000122
G4Z1Ug34B5I_000224_000234
pk1yi_HMsAI_000053_000063
```

### 6.2 Systems

1. **Independent baseline/local:** a separate frozen model instance without Route-D.
2. **Route-D open loop:** frozen controller changes reported candidates without state feedback.
3. **Route-D closed loop:** same controller and policy, with accepted actions fed into subsequent tracker state.

The baseline is independently instantiated to prevent accidental sharing of Route-D state.

### 6.3 Primary decision rule

The preregistered primary gate passes only when the paired-video 95% bootstrap confidence interval lower bounds are greater than zero for:

```text
- closed loop vs independent baseline Average Jaccard;
- closed loop vs independent baseline average point-threshold accuracy.
```

### 6.4 Undefined metrics

Seven videos contain at least one undefined official metric because the corresponding denominator is empty. Raw NaNs are preserved. No metric is zero-imputed. Only the affected comparison/metric pair is excluded from its finite paired bootstrap.

---

## 7. Main results

### 7.1 Aggregate corrected official-scale results

| System | Average Jaccard | Occlusion accuracy | Average point-threshold accuracy |
|---|---:|---:|---:|
| Independent baseline/local | 0.324945 | 0.939843 | 0.420461 |
| Route-D open loop | 0.337390 | 0.939843 | 0.436503 |
| Route-D closed loop | **0.347988** | 0.939843 | **0.447991** |

Occlusion accuracy is unchanged because Route-D intervenes on point coordinates, not the visibility decision.

In the percentage-point convention commonly used in TAP tables, the baseline is 32.49 AJ and closed loop is 34.80 AJ: an absolute gain of +2.30 AJ points and a relative gain of approximately +7.09%. This conversion does not change the central limitation that the absolute system remains substantially below modern strong trackers.

### 7.2 Aggregate gains

| Comparison | Delta Average Jaccard | Delta average point-threshold accuracy |
|---|---:|---:|
| Open loop vs baseline | +0.012445 | +0.016043 |
| Closed loop vs baseline | **+0.023043** | **+0.027530** |
| Closed loop vs open loop | **+0.010598** | **+0.011488** |

### 7.3 Paired-video bootstrap

| Comparison | Metric | Finite videos | Mean gain | 95% bootstrap CI |
|---|---|---:|---:|---:|
| Open loop vs baseline | AJ | 1,138 | +0.012445 | [+0.011200, +0.013715] |
| Open loop vs baseline | Point-threshold average | 1,137 | +0.016043 | [+0.014671, +0.017425] |
| Closed loop vs baseline | AJ | 1,138 | **+0.023043** | **[+0.020567, +0.025569]** |
| Closed loop vs baseline | Point-threshold average | 1,137 | **+0.027530** | **[+0.024800, +0.030260]** |
| Closed loop vs open loop | AJ | 1,138 | **+0.010598** | **[+0.008805, +0.012395]** |
| Closed loop vs open loop | Point-threshold average | 1,137 | **+0.011488** | **[+0.009494, +0.013465]** |

The primary gate passes. The closed-loop-versus-open-loop result also resolves the additional value of feedback at the dataset aggregate level.

### 7.4 Video-level heterogeneity

| Comparison | Metric | Positive | Negative | Tie |
|---|---|---:|---:|---:|
| Open loop vs baseline | AJ | 770 | 188 | 180 |
| Open loop vs baseline | Point-threshold average | 784 | 169 | 184 |
| Closed loop vs baseline | AJ | 775 | 199 | 164 |
| Closed loop vs baseline | Point-threshold average | 783 | 188 | 166 |
| Closed loop vs open loop | AJ | 673 | 274 | 191 |
| Closed loop vs open loop | Point-threshold average | 654 | 289 | 194 |

The aggregate improvement is broad but not universal. Approximately one-sixth of finite videos decline in closed-loop AJ relative to baseline.

### 7.5 Intervention diagnostics

```text
closed-loop global selection rate:              4.0244%
open-loop global selection rate:                7.9028%
mean closed-loop trajectory difference:         2.1264 px
mean closed-loop vs open-loop trajectory diff:  1.7413 px
memory-write disagreement rate:                 0
```

The lower closed-loop selection rate is expected because state feedback changes subsequent candidate evidence and future decisions. It must not be interpreted as a separately tuned policy.

---

## 8. Mechanism interpretation

### 8.1 Why multi-threshold profiles matter

A single confidence score conflates fine localization with coarse recovery. Route-D can accept a candidate with strong 2--16 pixel utility while tolerating a small one-pixel disadvantage, but only within a frozen margin. This matches the asymmetric goal of global correction: recover a trajectory without discarding fine-localization risk.

### 8.2 Why sparse selection matters

The controller accepts only approximately 4% of active decisions. Therefore, the improvement does not come from replacing local tracking with global matching. It comes from identifying a small subset of states where the expected benefit of a global intervention exceeds its risk.

### 8.3 Why closed-loop feedback matters

Closed loop significantly outperforms open loop even though both use the same frozen controller. This implies that accepted corrections affect more than the current coordinate. By changing the next-frame prior and commit state, a successful correction can prevent persistent drift or improve future local searches.

### 8.4 Why closed loop can also fail more severely

The same feedback mechanism can amplify a wrong correction. Once an incorrect global candidate becomes the prior, later local searches and candidate evidence may be centered around the wrong location. The method therefore improves average performance while retaining a heavy-tail failure mode.

---

## 9. Severe failures and limitations

### 9.1 Severe corrected Kinetics failures

| Sample | Delta AJ | Delta point-threshold average | Mean trajectory difference |
|---|---:|---:|---:|
| `kinetics_source_s000_p000113_kinetics_s000_000113` | -0.297626 | -0.274853 | 6.265 px |
| `kinetics_source_s009_p000080_kinetics_s000_000080` | -0.252266 | -0.264887 | 9.613 px |
| `kinetics_source_s008_p000034_kinetics_s000_000034` | -0.159537 | -0.096077 | 5.851 px |

These cases are evidence of unresolved closed-loop instability. They cannot be used to tune the existing controller while retaining the current external claim.

### 9.2 Base-tracker strength

The corrected Kinetics baseline AJ is 0.324945. Modern TAP systems report substantially stronger benchmark numbers under their own published protocols. The current experiment therefore demonstrates component effectiveness on the frozen MMP tracker, not competitiveness with the strongest current trackers.

### 9.3 Single-backbone evidence

Route-D is currently validated as an integrated controller for MMP. The action interface is conceptually reusable, but no second tracker backbone has yet demonstrated equivalent local/global candidate exposure, controller transfer, and closed-loop integration.

### 9.4 No formal safety guarantee

Isotonic calibration and a risk-aware policy improve empirical selection, but the method does not provide a conformal, PAC, or sequential no-harm guarantee. “Risk-calibrated” refers to empirical probability calibration within the frozen development protocol.

### 9.5 External dataset scope

The strongest untouched claim is the exact 1,144-of-1,147 Kinetics materialization. DAVIS and RGB-Stacking appeared during prior development, and their pre-correction Route-D TAP position metrics are superseded. They must not be used as corrected headline evidence.

### 9.6 Missing official release segments

Three release-CSV segments were not materialized. Results must not be described as covering all 1,147 segments.

### 9.7 Undefined denominator cases

Seven videos have at least one undefined official metric. Their NaNs are retained and transparently excluded only from affected paired comparisons.

---

## 10. Discussion

Route-D suggests that global relocalization in an online point tracker is best viewed as a selective control action rather than an always-on matching branch. The local and global hypotheses have complementary failure modes, but a naive confidence comparison does not fully encode the asymmetric consequences of a wrong switch. Multi-threshold utility profiles expose whether a candidate is useful at fine and coarse scales, while the calibrated tree gate models the non-smooth action boundary observed in candidate diagnostics.

The closed-loop experiment is especially important. A framewise candidate selector can appear successful while having little effect on a tracker's future behavior. Conversely, one accepted correction can alter an entire trajectory suffix. By evaluating the same controller open loop and closed loop, we measure the downstream value and risk of state feedback directly. The positive full-scale result supports feedback as a core mechanism, but the severe negative videos show why stronger stability constraints remain necessary.

A future stability guard cannot be derived from Kinetics failures without invalidating the current external evidence. The next guard must be designed and selected using Kubric-only partitions, frozen prospectively, and evaluated under a new external protocol. Stronger publication evidence should also demonstrate compatibility with a second candidate-generating tracker or a substantially stronger frozen base.

---

## 11. Conclusion

We presented Route-D, a risk-calibrated sparse candidate-routing controller for online point tracking. Route-D predicts multi-threshold correctness profiles for local and global hypotheses, applies a calibrated beneficial-action gate with explicit fine-to-coarse constraints, and optionally feeds accepted global corrections into future tracker state. Under a frozen official-scale TAP-Vid-Kinetics protocol covering an exact local materialization of 1,144 of 1,147 release-CSV segments, closed-loop Route-D improves Average Jaccard and average point-threshold accuracy over both an independent baseline and open-loop routing. The result establishes the promise of selective feedback routing, but not universal safety or state-of-the-art tracking. Severe closed-loop failures, a single evaluated backbone, and the strength of the base tracker remain the central limitations.

---

## 12. Numerical authority

Do not manually change the result tables without rechecking:

```text
docs/TAPVID_KINETICS_OFFICIAL_PROTOCOL_AUDIT_2026-07-16.md
docs/ROUTED_TREE_CONTROLLER_AUDIT_2026-07-16.md
```

Primary result artifacts:

```text
/gemini/code/FSPT/outputs/routeD_causal_v3_protocol_20260715/kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.merged.json
/gemini/code/FSPT/outputs/routeD_causal_v3_protocol_20260715/kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.paired.json
/gemini/code/FSPT/outputs/routeD_causal_v3_protocol_20260715/kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.final_audit.json
```

Generated single-source paper artifacts:

```text
docs/generated/ROUTED_OFFICIALSCALE_RESULT_SOURCE_2026-07-17.json
docs/generated/ROUTED_OFFICIALSCALE_TABLES_2026-07-17.md
docs/generated/ROUTED_OFFICIALSCALE_TABLES_2026-07-17.tex
```

Regenerate only from the frozen final audit and paired artifacts with:

```bash
python scripts/build_routeD_paper_tables.py \
  --final-audit /gemini/code/FSPT/outputs/routeD_causal_v3_protocol_20260715/kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.final_audit.json \
  --paired /gemini/code/FSPT/outputs/routeD_causal_v3_protocol_20260715/kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.paired.json \
  --output-json docs/generated/ROUTED_OFFICIALSCALE_RESULT_SOURCE_2026-07-17.json \
  --output-markdown docs/generated/ROUTED_OFFICIALSCALE_TABLES_2026-07-17.md \
  --output-tex docs/generated/ROUTED_OFFICIALSCALE_TABLES_2026-07-17.tex
```
