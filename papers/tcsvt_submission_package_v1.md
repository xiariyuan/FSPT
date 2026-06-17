# TCSVT Submission Package v1

## Recommended Title

Support-Memory Driven Causal Re-Entry Selection for Long-Term Point Tracking

## Alternative Title

Persistent Re-Entry Tracking: A Benchmark and Support-Memory Selector for Causal Point Recovery

## One-Sentence Pitch

We isolate causal first-frame re-entry after long invisibility as a distinct point-tracking problem, show that pre-occlusion support memory is the dominant usable cue, and provide a lightweight selector that improves re-entry accuracy under strict causal evaluation.

## Short Abstract

Long-term point trackers remain fragile when a point disappears for a long interval and later re-enters the view. Existing benchmarks usually average over the full trajectory and therefore under-expose this first-frame recovery failure mode. We introduce Persistent Re-Entry Tracking (PRT), a benchmark for evaluating causal recovery at the first visible frame after long occlusion or off-screen return. Our analysis shows a clear oracle gap between a causal baseline and the best candidate available from a local re-entry search pool, but also shows that many learned leakage-free acceptors and verifiers fail to exploit this gap reliably. We identify pre-occlusion support memory as the dominant causal cue for re-entry selection and propose a simple non-learned Support-Memory Hybrid Selector. On a 3-sequence PointOdyssey validation set with 300 re-entry queries, the selector reduces the re-entry median error from 36.23 px to 32.97 px at full coverage and to 31.30 px at a 75% selective operating point, while increasing the fraction of predictions below 4 px from 2.0% to 6.3% and 6.7%, respectively. These results establish causal re-entry as a concrete tracking problem and show that explicit support memory provides a practical path toward better post-occlusion point recovery.

## Contribution Bullets

1. We define Persistent Re-Entry Tracking, a causal benchmark for first-frame recovery after long invisibility.
2. We show that pre-occlusion support memory is the dominant usable signal for causal re-entry selection.
3. We propose a simple Support-Memory Hybrid Selector that improves re-entry accuracy at both full-coverage and selective operating points.
4. We provide a leakage-free failure analysis showing that the current bottleneck lies in signal quality rather than in decision-boundary complexity.

## Main Table Template

| Method | Median Error (px) | <4 px | Coverage | Better Frac |
|---|---:|---:|---:|---:|
| Baseline only | 36.23 | 0.020 | 1.00 | - |
| Oracle candidate pool | 25.90 | 0.130 | 1.00 | - |
| Hybrid selector (full) | 32.97 | 0.063 | 1.00 | 0.513 |
| Hybrid selector (75%) | 31.30 | 0.067 | 0.75 | 0.423 |

## Feature Analysis Table

| Feature | AUC |
|---|---:|
| support_margin | 0.673 |
| cand_score | 0.483 |
| cand_ncc | 0.502 |

## Claims To Use

1. A substantial causal re-entry oracle gap exists even when full-sequence average metrics may hide it.
2. Support-memory margin is the only clearly predictive cue among the tested candidate-selection signals.
3. A simple explicit selector already improves causal re-entry accuracy without introducing a heavy learned verifier.
4. Learned gating does not outperform explicit thresholding under the current feature space.

## Claims To Avoid

1. Do not claim that long-term point tracking is solved.
2. Do not claim strong universal statistical significance from bootstrap alone.
3. Do not frame the method as a new general-purpose tracking architecture.
4. Do not oversell the selective setting as a strict improvement on every distributional summary.

## Suggested Section Order For Writing

1. Introduction
2. Problem formulation and benchmark definition
3. Support-memory signal analysis
4. Hybrid selector
5. Experiments
6. Limitations and discussion
7. Conclusion

## Immediate Missing Assets

1. Teaser figure with baseline vs hybrid vs ground truth.
2. Coverage-risk curve figure.
3. Feature AUC bar chart.
4. Pooling-ablation mini table.
5. Stratified results by in-frame occlusion vs off-screen return.

## Current Risk Assessment

The paper is strong enough for a CCF-B journal submission if the final draft is clean, honest about scope, and includes a few targeted visualizations and stratified analyses. The main remaining risk is not the quantitative core result, but presentation quality and whether the paper looks like a coherent video-technology story rather than an internal experiment log.
