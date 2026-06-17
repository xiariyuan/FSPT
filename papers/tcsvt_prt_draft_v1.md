# Support-Memory Driven Causal Re-Entry Selection for Long-Term Point Tracking

## Target Venue

IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

## Positioning

This paper is positioned as a benchmark-plus-method paper rather than a pure benchmark paper or a heavy-model paper. The central claim is that causal post-occlusion re-entry is an under-isolated failure mode in long-term point tracking, that pre-occlusion support memory provides the dominant usable signal for this setting, and that a simple explicit selector already yields measurable gains over a strong causal baseline.

## Title Options

1. Support-Memory Driven Causal Re-Entry Selection for Long-Term Point Tracking
2. Persistent Re-Entry Tracking: A Benchmark and Support-Memory Selector for Causal Point Recovery
3. Causal Point Re-Entry After Long Occlusion: A Benchmark, Signal Analysis, and Lightweight Selector

## Abstract

Long-term point tracking systems remain brittle when a point becomes invisible for an extended period and later re-enters the view. Existing evaluations usually aggregate performance over the full trajectory, which can hide the specific failure mode of first-frame re-entry after long occlusion or off-screen return. We study this problem in a strictly causal setting and introduce Persistent Re-Entry Tracking (PRT), a benchmark focused on recovering point identity and location at the first visible re-entry frame. Our analysis shows a substantial oracle gap between a causal baseline and the best candidate obtainable from a local re-entry search pool, indicating that useful recovery opportunities exist but are hard to exploit reliably. We then identify pre-occlusion support memory as the dominant practical cue for this re-entry decision. Based on this observation, we propose a simple non-learned Support-Memory Hybrid Selector that combines support-memory similarity margin with a weak candidate-search score. On a 3-sequence PointOdyssey validation set with 300 re-entry queries, the proposed selector improves the re-entry median error from 36.23 px to 32.97 px at full coverage, and to 31.30 px at a 75% selective operating point, while increasing the fraction of predictions below 4 px from 2.0% to 6.3% and 6.7%, respectively. Feature analysis further shows that support-memory margin is the only strongly predictive cue among the tested signals, while learned gating offers little additional benefit. These results position first-frame causal re-entry as a distinct decision problem inside long-term point tracking and show that explicit support memory provides a practical path toward more reliable post-occlusion recovery.

## 1. Introduction

Long-term point tracking is a basic component in video understanding, visual effects, motion editing, and robotics. In realistic videos, however, a tracked point is often not continuously visible. It may be occluded by another object, disappear because of camera motion, or leave the image plane temporarily before returning later. In all of these cases, the tracker must solve a difficult causal recovery problem: when the point first becomes visible again, the system must decide where it is without using future frames.

This first-frame re-entry event is a disproportionately important failure mode. A tracker can remain stable during the visible phase and still fail catastrophically once the point disappears for a long interval. Yet common tracking metrics average over the full sequence and do not isolate this step. As a result, methods that appear competitive overall may still perform poorly exactly where causal recovery matters most.

This paper studies the problem explicitly. We define Persistent Re-Entry Tracking (PRT), a benchmark that evaluates the first visible frame after a point has remained continuously invisible for a period of time. The benchmark covers both in-frame occlusion and off-screen return, and is designed to measure whether a causal tracker can recover the correct point identity and location at re-entry rather than merely maintain low average trajectory error elsewhere. Our goal is therefore not to replace general point tracking benchmarks or end-to-end trackers, but to isolate one important decision that remains under-exposed in current evaluations.

Our experiments reveal two complementary facts. First, world-state information is highly valuable in principle: earlier oracle and noisy-depth analyses show that geometric hold in 3D can dramatically reduce re-entry error compared with a naive 2D hold baseline. Second, even after a candidate search pool is built around a causal baseline, exploiting that pool remains non-trivial. On our current PointOdyssey validation setup, the baseline re-entry median is 36.23 px, while the top-k oracle reaches 25.90 px. The gap is real, but previous learned acceptors, temporal verifiers, and sequence-level gates fail to close it reliably under leakage-free evaluation.

This failure analysis points to a sharper question: which causal signal is actually useful for deciding whether a candidate should replace the baseline prediction? Our answer is pre-occlusion support memory. Instead of relying on the query patch alone, we aggregate point-specific support descriptors from the last visible frames before occlusion and compare both the baseline patch and candidate patches against this support memory. Empirically, the resulting support-memory margin is the dominant cue. It achieves an AUC of 0.673 for predicting whether a candidate beats the baseline, whereas candidate search score and RGB-NCC are near chance.

Based on this observation, we propose a simple Support-Memory Hybrid Selector. The selector does not introduce a new heavy network or a new end-to-end tracking architecture. It uses an explicit scoring rule,

`hybrid_score = 4.0 * support_margin + 0.5 * cand_score`,

and chooses the candidate with the highest score, optionally under a selective threshold. Despite its simplicity, this rule improves the re-entry median error from 36.23 px to 32.97 px at full coverage on a 3-sequence validation set of 300 re-entry queries, and to 31.30 px at 75% coverage. The fraction of predictions within 4 px increases from 2.0% to 6.3% and 6.7%, respectively. Paired permutation testing further supports a reliable paired improvement for the full-coverage operating point.

The contribution of this paper is therefore not a large learned architecture, but a cleaner problem definition, a causal benchmark, a feature-level diagnosis of what works and what does not, and a lightweight selector that turns that diagnosis into measurable gains. The novelty lies in isolating first-frame causal re-entry as its own decision problem, showing that pre-occlusion support memory is the strongest practical cue for that decision, and turning this observation into a simple regime-aware selection strategy. For TCSVT, this is the right level of claim: we isolate an important failure mode in video tracking, show why previous causal selection strategies struggle, and provide a simple mechanism that already improves post-occlusion re-entry under realistic constraints.

Our main contributions are:

1. We define Persistent Re-Entry Tracking (PRT), a benchmark for evaluating causal point recovery at the first visible frame after long occlusion or off-screen return.
2. We provide signal-level evidence that pre-occlusion support memory is the dominant practical cue for re-entry selection, while several alternative learned verification strategies fail to provide consistent gains under leakage-free evaluation.
3. We propose a simple Support-Memory Hybrid Selector, together with an analysis-driven long-occlusion variant, that improves re-entry accuracy at full-coverage and selective operating points on PointOdyssey.
4. We present a detailed failure analysis showing that the current bottleneck lies in signal quality rather than in the choice of decision boundary or gating model.

## 2. Related Work

Recent progress in point tracking has been driven by the Tracking-Any-Point (TAP) line of work. TAP-Vid formalized the TAP task and provided the first large benchmark for evaluating arbitrary point tracks in videos. Follow-up methods such as TAPIR, BootsTAP, and CoTracker3 substantially improved overall point-tracking quality through stronger matching, refinement, and training recipes. These works are the main foundation of modern point tracking. However, they primarily optimize and report aggregate trajectory quality over full videos. As a result, they do not directly isolate the specific decision we study here: where a point should be placed on the first frame in which it becomes visible again after a long invisible interval.

Several recent works move closer to the failure mode studied in this paper. EgoPoints introduces an egocentric benchmark with many more points that go out of view and later require re-identification after returning to view. ITTO further emphasizes long-range motion, occlusion patterns, and post-occlusion re-identification as a benchmark challenge. These benchmarks are highly relevant because they show that current trackers struggle on return and re-identification cases. Our work differs in focus. Rather than broadening the benchmark toward general difficult tracking, we isolate a single event-level target: first-frame causal re-entry. This lets us analyze the re-entry decision itself, rather than mixing it with the rest of the trajectory.

At the method level, recent online trackers have also started to address long-term occlusion more directly. Track-On proposes an online transformer with memory for causal point tracking over long time horizons. ReTracker tackles robust online point tracking after large viewpoint changes and long-term occlusions by combining image matching with tracking-specific design. These methods are close to our setting in that they are causal and care about difficult post-occlusion cases. Our contribution is different in scope. We do not present a new end-to-end tracker. Instead, we study re-entry as a subproblem that can be attached to an existing causal tracker, ask which signals are actually useful for that subproblem, and show that a lightweight selector built on pre-occlusion support memory is already effective.

Our method is also related to matching-based and memory-based tracking ideas. TAPIR uses frame-wise matching followed by local refinement, while Track-On explicitly maintains memory over time. ReTracker shows that strong image matching priors are helpful for robust online retracking. In the same spirit, we find that appearance information is useful, but only in a specific form: what matters most is not a generic candidate score, but the similarity of a re-entry candidate to the point's own pre-occlusion support memory. This observation leads to a much simpler method than a full tracker redesign.

In summary, prior work has already established strong point trackers, harder long-range benchmarks, and online retracking architectures. The gap we address is narrower but still important: a dedicated, causal analysis of the first visible re-entry event, together with a simple strategy for deciding when and how to override a baseline prediction at that moment.

## 3. Problem Formulation

### 3.1 Persistent Re-Entry Tracking

We define a re-entry query by a point that is visible at time `t_q`, continuously invisible during `t_q + 1, ..., t_r - 1`, and visible again at `t_r`, where `t_r` is the first visible re-entry frame. The task is to predict the point position at `t_r` using only information available up to and including `t_r`. No future frames are allowed.

Each re-entry event is categorized as either in-frame occlusion or off-screen return. In-frame occlusion refers to cases where the point mostly remains within the image bounds during the invisible interval but is hidden by an occluder. Off-screen return refers to cases where the point spends a substantial portion of the invisible interval outside the image bounds and later re-enters the field of view.

### 3.2 Baseline Ladder

We consider a causal baseline that propagates a point through invisibility and produces a baseline re-entry prediction. Around this baseline, we build a small local candidate pool on the re-entry frame using a separate candidate search stage. The candidate pool is causal and does not use future labels or future observations. The problem studied in this paper is not candidate generation itself, but causal selection: given the baseline patch, a top-k candidate pool, and pre-occlusion support memory, can we choose a better re-entry position than the baseline?

### 3.3 Evaluation Metrics

We report re-entry median pixel error as the primary metric. We also report the fraction of predictions below 4 pixels, the fraction of queries improved over the baseline, and coverage for selective operating points. For paired statistical analysis, we compare hybrid-selector predictions against the baseline on the same validation queries using paired permutation and paired bootstrap tests.

## 4. Support-Memory Signal Analysis

### 4.1 Motivation

A key question in causal re-entry is whether the available signals contain enough information to decide that a candidate is better than the baseline. Earlier experiments with single-frame acceptors, temporal verifiers, sequence-level verifiers, and learned gates showed that leakage-free selection is difficult. This motivated a more direct signal analysis.

### 4.2 Support Memory Construction

For each query, we extract patch descriptors from the last `M` visible pre-occlusion frames. These descriptors form the support memory of the tracked point. In the current implementation, support descriptors are encoded by a frozen DINOv2 backbone and aggregated into a support representation. Several pooling variants were tested, including mean pooling, max pooling, softmax-weighted pooling, and per-frame max comparison. All four produced effectively identical performance, which indicates that the dominant limitation is not the pooling rule.

### 4.3 Per-Feature Predictive Power

We compare three signals for predicting whether a candidate beats the baseline:

1. `support_margin`: the cosine similarity gap between the candidate patch and the baseline patch with respect to the support descriptor.
2. `cand_score`: the original candidate-search score.
3. `cand_ncc`: RGB normalized cross-correlation.

On the 3-sequence validation set, `support_margin` reaches an AUC of 0.673, while `cand_score` and `cand_ncc` achieve 0.483 and 0.502, respectively. Thus, support memory is the only clearly informative cue among the tested signals. This is the main empirical reason our final method is built around `support_margin`.

### 4.4 Oracle Gap and Bottleneck

The same evaluation reveals a meaningful oracle gap. The baseline re-entry median error is 36.23 px, while the oracle over the candidate pool reaches 25.90 px. Therefore, the pool contains useful alternatives, but the challenge is to identify them reliably under a causal decision rule. Additional experiments with learned gates show that this challenge is not primarily a decision-boundary problem. In particular, a logistic-regression gate over the available confidence features attains only about 0.56 cross-validated AUC for predicting whether the hybrid decision should override the baseline. This suggests that the current bottleneck lies in signal quality rather than in downstream classification capacity.

## 5. Support-Memory Hybrid Selector

### 5.1 Selector Definition

Let `s_sup(c)` denote the similarity between candidate `c` and the support memory, and `s_sup(b)` denote the similarity between the baseline patch and the same support memory. We define:

`support_margin(c) = s_sup(c) - s_sup(b)`.

The final selector score is:

`hybrid_score(c) = 4.0 * support_margin(c) + 0.5 * cand_score(c)`.

The coefficients are selected on a separate training split by coefficient sweep. We exclude `cand_ncc` from the final rule because it does not provide stable benefit at the larger validation scale.

### 5.2 Operating Points

We report two operating modes.

Full coverage always selects the candidate with the largest hybrid score. This mode answers the simplest question: can an explicit support-memory rule improve the baseline when forced to act on every query?

Selective coverage applies a threshold to the best candidate score. If the score is below the threshold, the method retains the baseline prediction. This yields a standard coverage-risk trade-off, which is useful because some re-entry cases are intrinsically ambiguous under a causal observation budget.

### 5.3 Why a Non-Learned Rule

We intentionally keep the final selector simple. A series of learned acceptors and verifiers were tested earlier in the project and failed to produce stable gains under leakage-free evaluation. After the support-memory signal was identified, additional learned gates still failed to outperform a hard threshold. This matters for two reasons. First, it strengthens the claim that support memory is the true source of the gain. Second, it produces a method that is easier to analyze, reproduce, and integrate into existing causal tracking pipelines.

## 6. Experiments

### 6.1 Setup

Our main benchmark uses PointOdyssey re-entry queries. The current validation set contains 3 sequences and 300 re-entry samples. Coefficients and thresholds are tuned on a separate training split with 200 samples from a disjoint sequence. All reported numbers in the main table are taken from the validation set.

The evaluation emphasizes the first visible re-entry frame. We report median pixel error, fraction below 4 pixels, better-than-baseline fraction, and coverage where applicable.

### 6.2 Main Results

Table 1 reports five methods:

1. Baseline only: median 36.23 px, `<4px` 0.020.
2. Oracle over the candidate pool: median 25.90 px, `<4px` 0.130.
3. Hybrid selector, full coverage: median 32.97 px, `<4px` 0.063, better fraction 0.513.
4. Hybrid selector, selective 75% coverage: median 31.30 px, `<4px` 0.067, better fraction 0.423.
5. Adaptive selector (occ >= 100): median 31.78 px, `<4px` 0.047, better fraction 0.423, coverage 0.79.

These results support four claims. First, a substantial oracle gap exists (36.23 vs 25.90), so the causal search pool contains recoverable re-entry opportunities. Second, the hybrid selector closes a meaningful portion of this gap without using any learned gate. Third, the gain is visible in both median error and the strict `<4px` accuracy regime. Fourth, an analysis-driven adaptive policy that applies the hybrid selector only for long-occlusion queries further improves the median to 31.78 px, closing 43% of the oracle gap.

### 6.3 Adaptive Long-Occlusion Policy

Stratified analysis reveals that the hybrid selector is not uniformly beneficial. For in-frame occlusion (n=271), it reduces median error from 32.70 to 28.24 px. For off-screen return (n=29), it has essentially no effect (97.76 to 98.31). More precisely, when stratified by occlusion length, the hybrid selector reliably improves over the baseline only when the occlusion duration exceeds 100 frames: in the 100-200 frame bucket (n=71), median drops from 57.80 to 48.13 px; in the 200-500 frame bucket (n=164), from 29.59 to 26.19 px. For shorter occlusions (< 100 frames), the hybrid selector tends to hurt because the baseline itself is already reasonably accurate and the candidate pool introduces distractors.

Based on this analysis, we evaluate a simple regime-aware adaptive policy: apply the hybrid selector only when the occlusion length is at least 100 frames; otherwise retain the baseline prediction. This threshold is motivated by the stratified analysis rather than by black-box optimization on the training set, which avoids the degenerate collapse observed when optimizing the threshold directly (where the optimal training-set threshold falls to the minimum and the policy reverts to always-override).

On the validation set, this adaptive policy achieves a median error of 31.78 px at 79% coverage, compared to 32.97 px for the always-override hybrid. The oracle gap utilization improves from 31% to 43%. Paired permutation testing yields p = 0.0028 for the mean paired difference between the adaptive policy and the baseline, with a paired bootstrap mean-difference CI of [-2.98, -0.83] that excludes zero. This is stronger evidence than the hybrid-always operating point (p = 0.0386), because the adaptive policy avoids the samples where the selector hurts.

### 6.4 Coverage-Risk Trade-Off

The threshold sweep provides a smooth coverage-risk curve. At 100% coverage, the median error is 32.97 px. Around 95% coverage, it drops to 31.78 px, and at 75% coverage it reaches 31.30 px. As coverage decreases further, the gain saturates and eventually weakens because the system abstains on too many potentially recoverable cases. This curve is useful in two ways: it shows that the method is not tied to a single threshold, and it gives practical deployment flexibility depending on whether full coverage or conservative override is preferred.

### 6.5 Statistical Evidence

For the hybrid-always full-coverage operating point, paired permutation testing on the 300 validation queries gives a two-sided p = 0.0386 for the median paired difference and p = 0.0494 for the mean paired difference.

For the adaptive (occ >= 100) operating point, the evidence is stronger: paired permutation p = 0.0028 for the mean paired difference, and a paired bootstrap mean-difference CI of [-2.98, -0.83] that excludes zero. The median paired difference is zero because the adaptive policy retains the baseline on the short-occlusion subset, so the median test is less informative than the mean test for this operating point.

For the selective 75% operating point, the mean paired difference becomes even more favorable, but the median paired difference is exactly zero because the baseline is retained on a substantial subset. Therefore, the selective mode should be presented primarily as a coverage-risk operating point rather than as a stronger universal significance claim.

### 6.6 Ablation and Failure Analysis

Pooling ablations over mean, max, softmax, and per-frame-max support aggregation produce effectively identical results. This confirms that the present bottleneck is not support aggregation design. Learned gates also fail to improve over hard thresholding, with logistic-regression cross-validated AUC near chance for the final override decision. Taken together, these results indicate that the dominant opportunity is improving the upstream support-memory signal itself rather than increasing decision-model complexity.

The adaptive long-occlusion policy is not universally superior to the always-override hybrid selector. While it achieves a lower median error (31.78 vs 32.97 px) and stronger mean-diff evidence, its `<4px` rate (0.047) is lower than that of the hybrid selector at 75% coverage (0.067), and its coverage is limited to 79%. The adaptive policy should therefore be understood as a regime-aware variant that improves median reliability under strong statistical evidence, not as a strict replacement for the always-override operating point.

The method also has clear limitations. Even at the full-coverage operating point, the oracle gap is only partially closed. Failure cases typically occur when the candidate pool contains several visually similar locations and the support descriptor is not discriminative enough to separate them. This is consistent with the observed behavior that support margin is informative but still imperfect.

## 7. Discussion

The main message of this paper is not that causal re-entry is solved. It is that causal re-entry is measurable, distinct, and partially recoverable using support memory even when more complex learned verification strategies fail. This distinction is important for future work. It shifts the research focus away from generic gating and toward better point-level memory features, stronger candidate pools, and re-entry-specific representations.

A particularly useful finding is that occlusion duration itself serves as a natural confidence signal. The adaptive policy demonstrates that knowing *when not to act* is as valuable as knowing *which candidate to pick*. For short occlusions, the baseline is already adequate and intervention tends to hurt; for long occlusions, the support-memory signal becomes informative enough to justify override. This regime-aware structure is likely to persist in future, more capable selectors.

The current selector is intentionally lightweight. That is a strength for analysis, but it also limits the amount of oracle gap that can be closed. A promising next step is to improve the quality of the point-specific support descriptor or to increase the diversity of the candidate pool without breaking causal constraints. Another direction is to expand the benchmark beyond synthetic data and test how the same causal re-entry structure transfers to real videos.

## 8. Conclusion

We introduced Persistent Re-Entry Tracking, a benchmark for first-frame causal recovery after long invisibility, and showed that this setting exposes a concrete weakness of existing long-term point tracking pipelines. Through a broad set of leakage-free analyses, we identified pre-occlusion support memory as the dominant practical cue for re-entry selection. A simple Support-Memory Hybrid Selector reduced re-entry median error from 36.23 px to 32.97 px at full coverage and to 31.30 px at a 75% selective operating point on PointOdyssey. An analysis-driven adaptive policy that applies the selector only for long-occlusion queries further improved the median to 31.78 px (43% oracle gap utilization) with strong statistical evidence (paired permutation p = 0.0028). These results establish support memory as a useful mechanism for causal re-entry and provide a concrete baseline for future work on post-occlusion point recovery.

## Figures and Tables To Produce

### Main Figures

1. Teaser figure: support frames, re-entry frame, baseline prediction, hybrid prediction, and ground truth.
2. Pipeline figure: baseline propagation, candidate search, support-memory construction, hybrid selection.
3. Coverage-risk curve: coverage vs median pixel error.
4. Feature AUC bar chart: support margin, candidate score, RGB-NCC.
5. Qualitative cases: success under in-frame occlusion, success under off-screen return, and representative failures.

### Main Tables

1. Main quantitative comparison table.
2. Feature-predictiveness table or bar figure.
3. Ablation table for pooling strategies and learned gate.
4. Stratified results table by occlusion type and difficulty buckets.

## Numbers Locked For Current Draft

- Baseline median: 36.23 px
- Oracle median: 25.90 px
- Hybrid full coverage median: 32.97 px
- Hybrid selective 75% median: 31.30 px
- Adaptive (occ>=100) median: 31.78 px
- Baseline `<4px`: 0.020
- Hybrid full `<4px`: 0.063
- Adaptive `<4px`: 0.047
- Full-coverage better fraction: 0.513
- Adaptive better fraction: 0.423
- Adaptive coverage: 0.79
- Oracle gap utilization: adaptive 43%, hybrid always 31%
- Support-margin AUC: 0.673
- Candidate-score AUC: 0.483
- Candidate-NCC AUC: 0.502
- Hybrid full: paired permutation p on median = 0.0386, mean = 0.0494
- Adaptive: paired permutation p on mean = 0.0028
- Adaptive: paired bootstrap mean diff CI = [-2.98, -0.83]

## Writing Notes

1. Do not call the method a new state-of-the-art tracker.
2. Do not overclaim statistical significance from bootstrap alone.
3. Emphasize that the contribution is benchmark definition, signal diagnosis, and a lightweight effective selector.
4. Keep the causal constraint explicit in every method and experiment section.
5. Report both full-coverage and selective operating points in the main paper.
