# Paper Results Section Draft — 2026-07-04

## 4. Experiments and Results

### 4.1 Evaluation protocol

We evaluate ReEntry on standard TAP-Vid datasets and re-entry-focused diagnostic settings. For standard TAP-Vid metrics, we report Average Jaccard (AJ), Occlusion Accuracy (OA), and average position accuracy (δ_avg). These metrics are computed using the official TAP-Vid metric formulation or an equivalent port. Since ReEntry specifically targets failures that occur when a point reappears after occlusion or disappearance, we additionally report AJ_RD as a re-entry diagnostic metric.

AJ_RD is not used as a replacement for standard TAP-Vid metrics. Instead, it isolates the post-reappearance subset where re-entry failures occur and where full-trajectory metrics can dilute the failure signal. Unless otherwise stated, all official-style results are local evaluations under the stated protocol, not official leaderboard/server submissions.

We organize the results into five groups: DAVIS first/input, DAVIS strided/original, RGB-Stacking full50 strided, diagnostic RGB fresh/stress settings, and an official-safe threshold sweep.

---

### 4.2 DAVIS first/input: ReEntry improves standard metrics and AJ_RD over the offline base

We first evaluate on TAPVid-DAVIS using a first-query/input-resolution official-style local protocol with 30 videos and 650 queries. This setting corresponds to the query-first style commonly reported in TAP-Vid comparisons.

The CoTracker3 offline base obtains 62.6566 AJ, 88.1487 OA, and 0.3142 AJ_RD. ReEntry improves all of these metrics. V1 reaches 64.5758 AJ, 91.7274 OA, and 0.3588 AJ_RD. V22Q further improves AJ to 64.7260 and OA to 91.7406, with 0.3556 AJ_RD. V24-DINOScore gives the strongest ReEntry result in this setting, reaching 64.8439 AJ, 91.8851 OA, and 0.3549 AJ_RD.

Compared with CoTracker3 offline, V24 improves by +2.1874 AJ, +3.7363 OA, and +0.0407 AJ_RD. The paired-video check confirms that this improvement is stable across videos: V24 improves AJ by +2.1874 percentage points with a 95% bootstrap confidence interval of [+1.2582, +3.2487], and OA by +3.7363 points with a confidence interval of [+2.1673, +5.7423]. V24 wins/losses/ties are 24/5/1 for AJ and 27/2/1 for OA.

TrackOn2 remains the strongest external baseline in this DAVIS first/input setting, with 67.0406 AJ, 92.0916 OA, and 0.3714 AJ_RD. Therefore, we do not claim that ReEntry outperforms TrackOn2 on DAVIS. The correct conclusion is that ReEntry substantially improves the CoTracker3 offline base and closes part of the gap while targeting a different failure mode.

---

### 4.3 DAVIS strided/original: default ReEntry improves AJ_RD but exposes a stricter-protocol trade-off

We next evaluate on TAPVid-DAVIS using a stricter strided/original-resolution official-style local protocol. This setting contains 30 videos and 5,882 strided queries. It is more demanding for visibility correction because the metric averages across a much denser set of query points and frames.

Under this protocol, CoTracker3 offline obtains 51.5385 AJ, 92.1543 OA, 63.5892 δ_avg, and 0.3870 AJ_RD. The default ReEntry variants improve AJ_RD but reduce standard AJ/OA. V1 reaches 0.4144 AJ_RD, a +0.0274 improvement over offline, but AJ decreases to 50.7303 and OA decreases to 91.2730. V22Q obtains 0.4136 AJ_RD, a +0.0266 improvement, while AJ and OA are 50.8342 and 91.3439.

This result reveals a central trade-off: ReEntry successfully corrects post-reappearance visibility lag, but under dense strided evaluation, aggressive visibility changes can introduce false-visible penalties that reduce global AJ/OA. Since ReEntry does not modify coordinates, δ_avg remains unchanged at 63.5892 for the ReEntry variants.

To test whether the method can be made official-safe, we sweep the V1 confidence threshold. The best conservative point is threshold=0.80. This V25-safe setting obtains 51.5648 AJ, 92.2106 OA, and 0.3900 AJ_RD. Relative to CoTracker3 offline, this is +0.0263 AJ, +0.0563 OA, and +0.0030 AJ_RD. The gain is small and should not be interpreted as a strong leaderboard-style improvement. Instead, it shows that ReEntry can be operated conservatively to preserve standard AJ/OA under the stricter strided/original protocol.

---

### 4.4 RGB-Stacking full50 strided: ReEntry improves AJ_RD and OA on the complete 50-video set

We evaluate on the full TAPVid RGB-Stacking set available locally, covering all 50 videos from `rgb_stacking_000000` through `rgb_stacking_000049`, with 60,829 queries. We report a strided official-style local evaluation with standard TAP-Vid metrics.

The CoTracker3 offline base obtains 79.9345 AJ, 91.6371 OA, 88.5714 δ_avg, and 0.3617 AJ_RD. ReEntry substantially improves AJ_RD and OA. V1 reaches 0.4414 AJ_RD and 93.0758 OA, corresponding to +0.0797 AJ_RD and +1.4387 OA. V22Q reaches 0.4400 AJ_RD and 93.0481 OA. V24-DINOScore reaches 0.4394 AJ_RD and 93.0389 OA.

The paired-video analysis confirms that the OA improvement is stable. V1 improves OA by +1.4387 percentage points with 39/11/0 wins/losses/ties. V22Q improves OA by +1.4110 points with 40/10/0 wins/losses/ties. V24 improves OA by +1.4018 points with 40/10/0 wins/losses/ties.

At the same time, all ReEntry variants show a small AJ trade-off on RGB-Stacking full50. V1 reduces AJ by -0.5043 points, V22Q by -0.3589 points, and V24 by -0.3371 points. The position-only metric δ_avg remains unchanged at 88.5714. This pattern is consistent with the method design: ReEntry modifies visibility, not coordinates. It improves re-entry recovery and occlusion accuracy, while standard AJ can decrease when false-visible penalties outweigh some visibility recovery benefits.

This RGB-Stacking full50 result is the strongest full-standard evidence for the method's intended failure mode. It demonstrates that ReEntry consistently improves AJ_RD and OA over the offline base on a complete 50-video TAP-Vid subset, with an explicit and measured AJ trade-off.

---

### 4.5 Diagnostic RGB fresh/stress evaluations isolate re-entry behavior

To isolate the re-entry failure mode, we also evaluate on RGB-Stacking fresh20-49 and two synthetic stress variants: translate_L16 and occluder_L16. These are diagnostic evaluations rather than official benchmark submissions.

Across the three settings, V24 improves AJ_RD and OA over the base. On fresh20-49 natural, AJ_RD improves from 0.3330 to 0.4027, and OA improves from 91.4636 to 92.9244. On translate_L16, AJ_RD improves from 0.5080 to 0.5564, and OA improves from 90.7805 to 92.1951. On occluder_L16, AJ_RD improves from 0.6417 to 0.6763, and OA improves from 90.8918 to 92.2293.

Averaged across these settings, V24 improves AJ_RD by +0.0509 and OA by +1.4043, while AJ decreases by -0.3006. This mirrors the full50 result: ReEntry improves the targeted re-entry and visibility behavior, while standard AJ can show a small trade-off.

---

### 4.6 Ablation and method positioning

The ReEntry variants play different roles and should not be collapsed into a single “best” method.

V1, the learned ReEntry-VisCalibrator, is the main AJ_RD-oriented method. It gives the strongest AJ_RD among ReEntry variants on RGB full50 and DAVIS strided/original default settings, but it can trade off standard AJ/OA under strict strided evaluation.

V22Q is a stability and interval-extension variant. It is slightly safer than V1 for standard AJ/OA and should be treated as the stable interval-processing version, not as a method replaced by V24.

V24-DINOScore adds an optional appearance micro-filter. It gives small AJ gains over V22Q and is the best ReEntry row on DAVIS first/input, but it introduces a tiny AJ_RD/OA cost in some other settings. Therefore, V24 should be positioned as an optional AJ-oriented appearance filter rather than the universal default.

V25 threshold=0.80 is a conservative official-safe operating point for DAVIS strided/original. It preserves AJ/OA but leaves only a tiny AJ_RD gain. This suggests that thresholding alone is not sufficient for a strong leaderboard-style improvement under dense strided evaluation.

The natural next method direction is V26: an official-metric-aware selective ReEntry mechanism. Such a selector should fire only when a visibility correction is expected to improve re-entry behavior without harming standard AJ/OA.

---

### 4.7 Limitations

There are three important limitations.

First, our results are official-style local evaluations, not official leaderboard/server submissions. We use the official TAP-Vid metric formulation or an equivalent port and explicitly state the query mode and resolution for each setting, but we do not claim an official server-returned leaderboard score.

Second, the current evidence does not cover the full TAP-Vid benchmark including Kinetics and Kubric full evaluations. The current standard results cover TAPVid-DAVIS and TAPVid RGB-Stacking full50, plus diagnostic RGB fresh/stress settings.

Third, ReEntry is not a universal standard-AJ booster. Its main contribution is correcting re-entry visibility failures. This improves AJ_RD and OA in several settings, but dense strided evaluation can expose false-visible penalties that reduce standard AJ. Conservative thresholding can preserve AJ/OA, but stronger leaderboard-style improvement will require an official-metric-aware selector.

---

### 4.8 Summary

Overall, the results support the following conclusion: ReEntry improves re-entry recovery across TAP-Vid official-style local evaluations and diagnostic stress settings. On DAVIS first/input, it improves AJ, OA, and AJ_RD over CoTracker3 offline. On RGB-Stacking full50 strided, it substantially improves AJ_RD and OA, with a small standard-AJ trade-off. On DAVIS strided/original, the default method improves AJ_RD but requires a conservative operating point to preserve standard AJ/OA.

This positions ReEntry as a targeted re-entry failure-mode method rather than a universal leaderboard optimizer. The results are strong enough for a careful paper narrative, and the most promising future direction is an official-metric-aware V26 selector for stronger performance under dense strided protocols.
