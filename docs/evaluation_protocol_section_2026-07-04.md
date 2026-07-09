# Paper Draft: Evaluation Protocol Section — 2026-07-04

## Evaluation Protocol

We evaluate ReEntry on standard TAP-Vid datasets and re-entry-focused diagnostic settings. For standard TAP-Vid metrics, we report Average Jaccard (AJ), Occlusion Accuracy (OA), and average position accuracy (δ_avg), computed using the official TAP-Vid metric formulation or an equivalent port. Because the failure mode targeted by ReEntry occurs specifically after a point reappears following occlusion or disappearance, we additionally report AJ_RD as a re-entry diagnostic metric. AJ_RD is not a replacement for standard TAP-Vid metrics; it isolates the re-entry/re-detection subset that can be diluted by full-trajectory averages.

Unless otherwise stated, our evaluations are local evaluations under the specified protocol, not official server/leaderboard submissions.

## Protocol Table

| Evaluation | Dataset | Videos | Queries | Query mode | Resolution | Metrics | Evaluation type |
|---|---:|---:|---:|---|---|---|---|
| DAVIS first/input | TAPVid-DAVIS | 30 | 650 | first | input / 256 | AJ, OA, δ_avg, AJ_RD | official-style local evaluation |
| DAVIS strided/original | TAPVid-DAVIS | 30 | 5,882 | strided | original | AJ, OA, δ_avg, AJ_RD | official-style local evaluation |
| RGB-Stacking full50 strided | TAPVid RGB-Stacking | 50 | 60,829 | strided | input / 256-style local cache | AJ, OA, δ_avg, AJ_RD | official-style local evaluation |
| RGB fresh20-49 | TAPVid RGB-Stacking subset | 30 | local | local re-entry split | 256/local | AJ, OA, AJ_RD | diagnostic local evaluation |
| RGB stress variants | RGB-Stacking derived | 30 each | local | synthetic translate/occluder | 256/local | AJ, OA, AJ_RD | diagnostic stress evaluation |

## DAVIS Official-Style Evaluations

We report two DAVIS official-style local evaluations.

First, under a first-query/input-resolution protocol, ReEntry improves over the CoTracker3 offline base in both standard TAP-Vid metrics and AJ_RD. The strongest ReEntry variant in this setting is V24-DINOScore:

```text
CoTracker3 offline:
  AJ    = 62.6566
  OA    = 88.1487
  δ_avg = 77.2244
  AJ_RD = 0.3142

V24-DINOScore:
  AJ    = 64.8439
  OA    = 91.8851
  δ_avg = 77.2244
  AJ_RD = 0.3549

Deltas:
  ΔAJ    = +2.1874
  ΔOA    = +3.7363
  Δδ_avg = +0.0000
  ΔAJ_RD = +0.0407
```

Second, under the stricter strided/original-resolution protocol, the default ReEntry variants improve AJ_RD but reduce standard AJ/OA. A conservative V25 threshold of 0.80 provides an official-safe operating point with small positive AJ/OA and a tiny AJ_RD gain:

```text
CoTracker3 offline:
  AJ    = 51.5385
  OA    = 92.1543
  δ_avg = 63.5892
  AJ_RD = 0.3870

V25-safe threshold=0.80:
  AJ    = 51.5648
  OA    = 92.2106
  δ_avg = 63.5892
  AJ_RD = 0.3900

Deltas:
  ΔAJ    = +0.0263
  ΔOA    = +0.0563
  Δδ_avg = +0.0000
  ΔAJ_RD = +0.0030
```

These two settings show that ReEntry is protocol-sensitive: it strongly improves first-query DAVIS metrics and the re-entry diagnostic, while strided/original evaluation requires conservative thresholding or future metric-aware selection to avoid global visibility penalties.

## RGB-Stacking Full50 Official-Style Strided Evaluation

We also evaluate on the complete locally available TAPVid RGB-Stacking full50 split, covering videos `rgb_stacking_000000` through `rgb_stacking_000049`. Unlike the earlier local full50 summary, the stricter table below uses a strided official-style local evaluation with standard TAP-Vid metrics.

```text
CoTracker3 offline:
  AJ     = 79.9345
  OA     = 91.6371
  δ_avg  = 88.5714
  δ_4px  = 91.5875
  AJ_RD  = 0.3617

V1:
  AJ     = 79.4302
  OA     = 93.0758
  δ_avg  = 88.5714
  δ_4px  = 91.5875
  AJ_RD  = 0.4414

V22Q:
  AJ     = 79.5756
  OA     = 93.0481
  δ_avg  = 88.5714
  δ_4px  = 91.5875
  AJ_RD  = 0.4400

V24-DINOScore:
  AJ     = 79.5974
  OA     = 93.0389
  δ_avg  = 88.5714
  δ_4px  = 91.5875
  AJ_RD  = 0.4394
```

This full50 official-style result confirms the main trade-off: ReEntry substantially improves AJ_RD and OA, while introducing a small standard-AJ trade-off and leaving position-only δ_avg unchanged.

## Diagnostic Re-entry and Stress Evaluations

To isolate re-entry behavior, we additionally report RGB-Stacking fresh20-49 and synthetic stress variants. These settings are not official benchmark submissions; they are diagnostic evaluations designed to test the re-entry failure mode directly.

Across fresh20-49 natural, translate_L16, and occluder_L16, V24 improves AJ_RD and OA over the base while showing a small AJ trade-off:

```text
Average over three settings:
  ΔAJ_RD = +0.0509
  ΔOA    = +1.4043
  ΔAJ    = -0.3006
```

## Recommended Claim

The safe overall claim is:

```text
ReEntry improves re-entry recovery across standard TAP-Vid official-style local evaluations and diagnostic stress settings. It improves DAVIS first-query AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter strided settings reveal a consistent trade-off between re-entry recovery, occlusion accuracy, and standard AJ.
```

Avoid claiming:

```text
official leaderboard submission
full TAP-Vid benchmark submission
universal improvement across all official protocols
V24 beats TrackOn2 on DAVIS
```
