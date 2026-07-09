# Paper Draft: Experimental Setup Section — 2026-07-04

This document provides a paper-ready Experimental Setup / Evaluation Protocol section. It is designed to accompany `docs/final_paper_tables_2026-07-04.md` and `docs/paper_results_section_draft_2026-07-04.md`.

---

## 3. Experimental Setup

### 3.1 Datasets

We evaluate on standard TAP-Vid datasets and targeted diagnostic variants.

**TAPVid-DAVIS.** We use the 30-video TAPVid-DAVIS evaluation set. We report two local official-style evaluation protocols. The first is a first-query/input-resolution evaluation with 650 queries and 256×256 input-resolution metrics. The second is a stricter strided/original-resolution evaluation with 5,882 queries and original-resolution metrics. These two protocols provide complementary views: first-query evaluation is commonly used for query-first tracker comparison, while strided/original evaluation is denser and more sensitive to global visibility errors.

**TAPVid RGB-Stacking full50.** We evaluate on the complete locally available 50-video TAPVid RGB-Stacking set, covering `rgb_stacking_000000` through `rgb_stacking_000049`, with 60,829 queries. This evaluation is a full50 local standard evaluation. We do not describe it as an official leaderboard submission or as a strict official query-protocol result until an explicit first/strided protocol audit or rerun is completed.

**Re-entry diagnostic splits.** We additionally evaluate on RGB-Stacking fresh20-49 and synthetic stress variants. These include a natural fresh split, a translation stress variant (`translate_L16`), and an occlusion stress variant (`occluder_L16`). These settings are designed to isolate the re-entry failure mode and should be interpreted as diagnostic evaluations rather than official benchmark submissions.

---

### 3.2 Metrics

We report standard TAP-Vid metrics and a re-entry diagnostic metric.

**Average Jaccard (AJ).** AJ measures the joint quality of point localization and visibility prediction over the evaluated frames and thresholds.

**Occlusion Accuracy (OA).** OA measures the accuracy of the predicted visibility/occlusion state.

**Average position accuracy (δ_avg).** δ_avg averages the fraction of visible points within a set of pixel thresholds. In the current ReEntry pipeline, δ_avg often remains unchanged because ReEntry modifies visibility decisions while preserving the base tracker coordinates.

**AJ_RD.** AJ_RD is a re-entry / re-detection diagnostic metric. It focuses on the subset of points after they reappear following occlusion or disappearance. We report AJ_RD because full-trajectory averages such as AJ and OA can dilute failures that happen on sparse re-entry frames. AJ_RD is reported alongside standard TAP-Vid metrics, not as a replacement for them.

---

### 3.3 Baselines and variants

We compare against CoTracker3 offline as the main base tracker. Depending on the protocol, we also include CoTracker3 baseline/online, TrackOn2, and TAPNext where available.

The ReEntry variants are:

**V1 Learned ReEntry-VisCalibrator.** V1 is the AJ_RD-oriented main method. It uses learned confidence to decide when to apply re-entry visibility correction.

**V22Q Interval/Gate.** V22Q adds interval and gate-based post-processing to improve stability and reduce noisy recovery segments.

**V24-DINOScore.** V24 adds an appearance-based DINOScore micro-filter. It is best understood as an AJ-oriented appearance filter that can recover some standard AJ, but it is not a universal replacement for V1 or V22Q.

**V25-safe threshold=0.80.** V25 is a conservative operating point for DAVIS strided/original evaluation. It raises the V1 threshold to preserve global AJ/OA while retaining a small AJ_RD gain.

---

### 3.4 Evaluation types and claim levels

We explicitly separate four evaluation types.

1. **Official-style local evaluation.** Local evaluation using TAP-Vid official query/metric conventions or an equivalent port. This applies to DAVIS first/input and DAVIS strided/original.

2. **Full local standard evaluation.** Full evaluation on all locally available videos of a standard TAP-Vid subset, but not necessarily under an explicitly audited official query mode. This applies to RGB-Stacking full50.

3. **Diagnostic local evaluation.** Local evaluation designed to isolate a target failure mode. This applies to fresh20-49 and stress settings.

4. **Official leaderboard submission.** Server-side result returned by an official leaderboard. We do not claim this.

This distinction is important because TAP-Vid public resources provide benchmark data and metric implementations, but we do not have a confirmed online official submission server result. Therefore, all benchmark-style claims in this paper should be phrased as local evaluations under a specified protocol.

---

### 3.5 Aggregation and paired analysis

For standard metric tables, we report mean performance over videos under the corresponding protocol. We also report paired-video confidence intervals for the key comparisons when available. Paired analysis is especially important because ReEntry modifies visibility decisions on top of a fixed base tracker, so improvements and trade-offs should be measured per video rather than only by a single aggregate score.

For RGB-Stacking full50 and DAVIS official-style evaluations, we report deltas against the CoTracker3 offline base. For DAVIS first/input, we additionally compare against external baselines such as TrackOn2 to contextualize the absolute performance.

---

## Recommended paper wording

```text
We evaluate on TAPVid-DAVIS and TAPVid RGB-Stacking using standard TAP-Vid metrics and a re-entry diagnostic metric, AJ_RD. DAVIS results are reported under official-style first/input and strided/original local protocols. RGB-Stacking full50 is reported as a full local standard evaluation over all 50 RGB-Stacking videos. Unless otherwise stated, all results are local evaluations and not official leaderboard submissions.
```

## Wording to avoid

```text
We submit to the official TAP-Vid leaderboard.
We evaluate on the full official TAP-Vid benchmark.
All official protocols improve.
Do not claim: V24 is universally best.
```
