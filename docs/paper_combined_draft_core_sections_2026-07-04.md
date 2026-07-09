# Combined Paper Draft Core Sections — 2026-07-04

This file combines the paper-ready abstract-style positioning, experimental setup, result narrative, limitations, and final claim boundary into a single draft package.

---

## Working Title

```text
ReEntry: Diagnosing and Correcting Re-Entry Visibility Failures in Point Tracking
```

Alternative titles:

```text
ReEntry-TAP: Visibility Re-Entry Calibration for Tracking Any Point
ReEntry: Failure-Mode-Aware Visibility Correction for Point Tracking
```

---

## Abstract Draft

Tracking any point through occlusion and reappearance remains challenging because trackers can exhibit visibility lag or false visibility after a point leaves and re-enters the scene. Standard TAP-Vid metrics such as Average Jaccard (AJ) and Occlusion Accuracy (OA) measure overall trajectory quality, but they can dilute failures that occur specifically after reappearance. We introduce ReEntry, a visibility correction pipeline that targets re-entry failure modes on top of existing point trackers. ReEntry applies learned visibility calibration, interval/gate post-processing, and optional appearance-based filtering to improve recovery after occlusion or disappearance.

We evaluate ReEntry on TAPVid-DAVIS and TAPVid RGB-Stacking using standard TAP-Vid metrics and a re-entry diagnostic metric, AJ_RD. Under a DAVIS first/input official-style local evaluation, ReEntry V24-DINOScore improves over the CoTracker3 offline base by +2.19 AJ, +3.74 OA, and +0.0407 AJ_RD. On TAPVid RGB-Stacking full50, ReEntry improves AJ_RD from 0.3617 to 0.4414 with V1 and improves OA from 91.64 to 93.08, with a small standard-AJ trade-off. Under the stricter DAVIS strided/original protocol, default ReEntry improves AJ_RD but reduces global AJ/OA; a conservative V25 threshold preserves AJ/OA while retaining a small AJ_RD gain. These results show that ReEntry improves the targeted re-entry failure mode, while also revealing that metric-aware selection is needed for stronger global benchmark gains under dense strided evaluation.

---

## Contributions Draft

```text
1. We identify and isolate re-entry visibility failures in point tracking, where a point reappears after occlusion or disappearance but the tracker visibility state lags or becomes unstable.

2. We introduce ReEntry, a visibility correction pipeline that can be applied on top of existing point trackers without changing their coordinate trajectories.

3. We evaluate ReEntry using both standard TAP-Vid metrics and AJ_RD, a re-entry diagnostic metric that isolates performance after reappearance.

4. We provide official-style local evaluations on TAPVid-DAVIS first/input and strided/original protocols, plus a full 50-video TAPVid RGB-Stacking local standard evaluation.

5. We show that ReEntry improves first-query DAVIS AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while strict strided/original evaluation requires conservative or metric-aware selection to preserve global metrics.
```

---

## Experimental Setup Draft

We evaluate on standard TAP-Vid datasets and targeted diagnostic variants. On TAPVid-DAVIS, we report two official-style local protocols. The first uses first-query evaluation at input resolution, with 30 videos and 650 queries. The second uses strided queries with original-resolution metrics, providing a stricter test of global visibility behavior. On TAPVid RGB-Stacking, we evaluate the complete locally available full50 set, covering `rgb_stacking_000000` through `rgb_stacking_000049`, with 60,829 queries. We also report RGB-Stacking fresh20-49 and synthetic stress variants to isolate re-entry behavior under translation and occlusion perturbations.

We report standard TAP-Vid metrics: Average Jaccard (AJ), Occlusion Accuracy (OA), and average position accuracy (δ_avg). Since ReEntry targets failures after reappearance, we additionally report AJ_RD as a re-entry diagnostic metric. AJ_RD is reported alongside standard TAP-Vid metrics and is not a replacement for them.

We compare against CoTracker3 offline as the primary base tracker. Where available, we also include CoTracker3 baseline/online, TrackOn2, and TAPNext. ReEntry variants include V1 Learned ReEntry-VisCalibrator, V22Q Interval/Gate, V24-DINOScore, and V25-safe threshold=0.80.

All results are local evaluations under the specified protocol. We do not claim official server or leaderboard submission.

---

## Main Results Draft

### DAVIS first/input official-style evaluation

Under the DAVIS first/input protocol, ReEntry improves both standard metrics and the re-entry diagnostic over the CoTracker3 offline base. CoTracker3 offline obtains 62.6566 AJ, 88.1487 OA, 77.2244 δ_avg, and 0.3142 AJ_RD. ReEntry V1 improves to 64.5758 AJ, 91.7274 OA, and 0.3588 AJ_RD. V22Q reaches 64.7260 AJ, 91.7406 OA, and 0.3556 AJ_RD. V24-DINOScore gives the best ReEntry standard metrics in this protocol, reaching 64.8439 AJ, 91.8851 OA, and 0.3549 AJ_RD.

Compared with CoTracker3 offline, V24-DINOScore improves by +2.1874 AJ, +3.7363 OA, and +0.0407 AJ_RD. Paired-video analysis confirms that these improvements are stable: V24 improves AJ with a 95% confidence interval of [+1.2582, +3.2487] and OA with a confidence interval of [+2.1673, +5.7423]. TrackOn2 remains the strongest external baseline overall, reaching 67.0406 AJ and 92.0916 OA.

### DAVIS strided/original official-style evaluation

The stricter DAVIS strided/original protocol reveals the main trade-off. CoTracker3 offline obtains 51.5385 AJ, 92.1543 OA, 63.5892 δ_avg, and 0.3870 AJ_RD. Default ReEntry V1 increases AJ_RD to 0.4144, but reduces AJ to 50.7303 and OA to 91.2730. V22Q similarly raises AJ_RD to 0.4136 while remaining below the offline base in AJ/OA.

A conservative V25 threshold of 0.80 preserves official-style AJ/OA and retains a small AJ_RD gain. It reaches 51.5648 AJ, 92.2106 OA, 63.5892 δ_avg, and 0.3900 AJ_RD. This corresponds to +0.0263 AJ, +0.0563 OA, and +0.0030 AJ_RD relative to offline. Since paired-video confidence intervals cross zero, this should be interpreted as an official-style safe operating point rather than a strong leaderboard improvement.

### RGB-Stacking full50 local standard evaluation

On RGB-Stacking full50, CoTracker3 offline obtains 0.3617 AJ_RD, 79.9345 AJ, and 91.6371 OA. ReEntry V1 improves AJ_RD to 0.4414 and OA to 93.0758, with a small AJ decrease to 79.4302. V22Q reaches 0.4400 AJ_RD, 79.5756 AJ, and 93.0481 OA. V24-DINOScore reaches 0.4394 AJ_RD, 79.5974 AJ, and 93.0389 OA.

These results show that ReEntry strongly improves re-entry recovery and visibility stability on a full 50-video RGB-Stacking evaluation. The trade-off is a small reduction in standard AJ. Paired-video statistics confirm that V24 improves AJ_RD by +0.074850 and OA by +1.401793 over the offline base, while reducing AJ by -0.337109.

### Diagnostic stress evaluations

On RGB fresh20-49 and synthetic stress variants, V24-DINOScore improves average AJ_RD by +0.0509 and OA by +1.4043, with an average AJ trade-off of -0.3006. This supports the interpretation that ReEntry improves the targeted failure mode rather than only shifting aggregate benchmark scores.

---

## Limitations Draft

ReEntry is not a universal benchmark booster. It targets a specific failure mode: visibility recovery after occlusion or disappearance. Under DAVIS first/input evaluation, this improves both standard metrics and AJ_RD. Under DAVIS strided/original evaluation, however, aggressive visibility correction can hurt global AJ/OA. This demonstrates that re-entry recovery and global benchmark optimization are related but not identical objectives.

The current pipeline primarily modifies visibility and preserves base tracker coordinates. Therefore, coordinate-based metrics such as δ_avg often remain unchanged. Future versions should combine re-entry visibility correction with coordinate refinement or localized re-detection.

We also do not claim official leaderboard submission or full TAP-Vid benchmark coverage. Current official-style local evaluations cover DAVIS first/input and DAVIS strided/original. RGB-Stacking full50 is a full local standard evaluation, but not yet an explicitly audited strict official first/strided query-protocol result. Kinetics and Kubric are not included in the current final result package.

---

## Future Work Draft

The most direct next step is metric-aware selective ReEntry. A future V26 should not select visibility corrections only based on re-entry confidence. Instead, it should estimate whether a proposed correction is likely to preserve or improve standard AJ/OA while also improving AJ_RD. This would directly address the trade-off observed under strided/original evaluation.

A second direction is coordinate-aware recovery. Since ReEntry currently preserves base coordinates, it cannot improve δ_avg. Coupling ReEntry with local coordinate refinement may improve both visibility and localization.

A third direction is broader protocol coverage. The next experimental supplement should audit or rerun RGB-Stacking full50 under explicit official first or strided query mode. Kinetics and Kubric can be considered after the core paper tables are locked.

---

## Conclusion Draft

ReEntry improves the targeted re-entry failure mode across multiple TAP-Vid local evaluations. It yields paired-stable improvements on DAVIS first/input official-style AJ/OA/AJ_RD and strong AJ_RD/OA improvements on RGB-Stacking full50. At the same time, stricter DAVIS strided/original evaluation reveals that aggressive visibility correction can harm global standard metrics. A conservative V25 operating point preserves AJ/OA with a small AJ_RD gain, motivating future metric-aware selective ReEntry.

---

## Final claim boundary

Safe claim:

```text
ReEntry improves re-entry recovery across TAP-Vid local evaluations and diagnostic settings. It improves DAVIS first/input official-style AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter DAVIS strided/original evaluation requires conservative selection to preserve standard metrics.
```

Avoid:

```text
Do not claim: official leaderboard submission
Do not claim: full TAP-Vid official benchmark submission
Do not claim: universal improvement across all official protocols
Do not claim: V24 beats TrackOn2 on DAVIS
Do not claim: V24 is universally best
```
