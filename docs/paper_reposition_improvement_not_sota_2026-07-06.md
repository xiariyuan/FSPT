# Paper Reposition — Improvement Paper Without SOTA Claim — 2026-07-06

## User constraint

The paper does not need to claim SOTA, but it must still read as an improvement/advantage paper. Metrics must not look too weak.

## Decision

This is achievable if the paper is framed as:

```text
A targeted improvement paper for re-entry visibility calibration under point-tracking re-detection failures.
```

Not as:

```text
A generic SOTA point tracker paper.
```

The correct claim is:

```text
ReEntry improves visibility-derived robustness and re-detection quality while preserving tracker coordinates.
```

The paper should emphasize that the method is:

```text
1. coordinate-preserving,
2. lightweight / plug-in,
3. focused on re-entry visibility failure,
4. beneficial on strong-enough baselines without degrading coordinate accuracy,
5. honest about limits on very strong or poorly aligned tracker modes.
```

## Main paper positioning

Recommended title direction:

```text
ReEntry: Coordinate-Preserving Visibility Calibration for Point Re-Detection
```

Alternative:

```text
ReEntry: Improving Point Tracking Re-Detection via Visibility Calibration
```

Main thesis:

```text
Modern point trackers often retain usable coordinates across occlusion/re-entry, but their visibility predictions lag or fail at re-entry. ReEntry targets this separable visibility failure mode and improves re-detection robustness without changing coordinates.
```

This lets the paper be an improvement paper because:

```text
The improvement is not over all possible tracking quality.
The improvement is over a concrete failure mode: re-entry visibility calibration.
```

## Which results can be main-table results

### Main Table 1 — DAVIS first/input CoTracker3 offline

This is the strongest main improvement result.

```text
CoTracker3 offline:
  AJ     62.66
  OA     88.15
  AJ_RD  0.5112

+ ReEntry:
  AJ     64.60
  OA     91.80
  AJ_RD  0.5644

Gain:
  AJ     +1.94
  OA     +3.65
  AJ_RD  +0.0532
```

Why this can be main:

```text
1. Metrics are not weak: AJ is above 60 and improves to 64.60.
2. Coordinate quality remains strong and unchanged.
3. Standard metrics improve, not only AJ_RD.
4. The result is easy to explain: visibility repair improves re-entry without moving points.
```

Important comparison framing:

```text
Compare primarily against the same base tracker's offline visibility stream.
Mention stronger online/TrackOn2 numbers as context, not as the target claim.
```

Do not claim:

```text
ReEntry beats TrackOn2 or SOTA.
```

### Main / Secondary Table 2 — RGB-Stacking full50

Use as robustness / different-dataset evidence, but carefully word it as a trade-off result.

Known frozen numbers:

```text
CoTracker3 offline:
  AJ     79.9345
  OA     91.6371
  AJ_RD  0.3617

ReEntry V24:
  AJ     79.5974
  OA     93.0389
  AJ_RD  0.4394

Delta:
  AJ     -0.3371
  OA     +1.4018
  AJ_RD  +0.0777
```

Use this if the paper says:

```text
ReEntry can improve visibility and re-detection at a small AJ trade-off on held-out RGB-Stacking.
```

Do not present it as unconditional improvement because AJ drops slightly.

### Main / Secondary Table 3 — DAVIS strided/original safe point

Use only as protocol-stress / safety evidence.

Known frozen numbers:

```text
CoTracker3 offline:
  AJ     51.5385
  OA     92.1543
  AJ_RD  0.3870

V25 tau=0.80:
  AJ     51.5648
  OA     92.2106
  AJ_RD  0.3900

Delta:
  AJ     +0.0263
  OA     +0.0563
  AJ_RD  +0.0030
```

Interpretation:

```text
Under strict strided/original protocol, conservative ReEntry preserves standard metrics and provides a small positive re-detection gain.
```

This is not a headline result, but it prevents reviewers from saying the method only works in easy protocol.

## Which results should not be main-table results

### TAPNext++ offline_w8

Do not use as main method improvement because metrics are weak:

```text
TAPNext++ offline_w8:
  AJ     12.49
  delta_avg 25.05
```

Use only in appendix/diagnostic cross-family stress section.

### TrackOn2 offline_batch

Do not use as main method improvement because metrics are extremely weak:

```text
TrackOn2 offline_batch:
  AJ     3.90
  delta_avg 7.36
```

Use only to show a failure mode, not a method advantage.

### TAPNext++ online confidence recovery

Do not claim deployable improvement:

```text
Global threshold best AJ_RD gain: +0.0157 but AJ -3.05 and OA -3.46.
Local recovery success_count = 0.
Best safe-ish local gain: AJ_RD +0.0017 only.
```

Use as honest negative result / limitation.

## Recommended paper contribution list

Use this contribution framing:

```text
1. We identify re-entry visibility lag as a separable failure mode in point tracking.
2. We propose ReEntry, a coordinate-preserving visibility calibration layer that can be applied without changing tracker coordinates.
3. On DAVIS first/input, ReEntry improves CoTracker3 offline from 62.66 to 64.60 AJ, from 88.15 to 91.80 OA, and from 0.5112 to 0.5644 AJ_RD.
4. On RGB-Stacking full50, ReEntry improves OA by +1.40 and AJ_RD by +0.0777 with a small AJ trade-off of -0.34.
5. Under strict DAVIS strided/original protocol, a conservative ReEntry setting preserves AJ/OA and gives a small positive AJ_RD gain.
6. We include cross-family diagnostic and negative results on TAPNext++/TrackOn2 showing both the generality of visibility lag and the limits of simple confidence recovery.
```

This reads as an improvement paper, not just a diagnostic report.

## Main narrative structure

### Abstract angle

```text
Point trackers are usually evaluated by coordinate accuracy, but long occlusion/re-entry also depends on visibility calibration. We show that visibility errors can persist even when coordinates remain usable. ReEntry is a lightweight, coordinate-preserving visibility calibration layer for re-detection. It improves DAVIS first/input CoTracker3 offline by +1.94 AJ, +3.65 OA, and +0.0532 AJ_RD, and improves RGB-Stacking re-detection robustness with only a small AJ trade-off. Additional controlled audits on TAPNext++ show that re-entry visibility lag can substantially affect metrics even with strong coordinates, while also revealing the limits of simple confidence-threshold recovery.
```

### Main claim

```text
ReEntry is not a new SOTA tracker. It is a targeted visibility-calibration improvement for re-entry failures.
```

### Result ordering

Recommended main tables:

```text
Table 1: DAVIS first/input main improvement on CoTracker3 offline.
Table 2: RGB-Stacking held-out robustness/trade-off.
Table 3: DAVIS strided/original conservative safety result.
Table 4 or Appendix: TAPNext++ controlled visibility-lag and confidence-recovery negative result.
Appendix: TrackOn2 and TAPNext++ offline stress results.
```

## Minimum metric-quality rule

For main paper tables:

```text
Do not put rows with AJ below 50 in the main table unless explicitly labeled as stress/diagnostic.
```

Therefore:

```text
CoTracker3 DAVIS first/input: main.
CoTracker3 RGB full50: main/secondary.
DAVIS strided safe point: secondary.
TAPNext++ offline_w8: appendix only.
TrackOn2 offline_batch: appendix only.
```

## Biggest remaining weakness

Even after repositioning, the paper's weakness is:

```text
The method is strongest on CoTracker3 offline and visibility-stress settings, not on normal online SOTA trackers.
```

How to defend:

```text
The paper is scoped to coordinate-preserving re-entry visibility calibration, not replacing tracker backbones.
The goal is to fix a separable failure mode with lightweight post-processing.
```

## Next action

Stop additional SOTA rescue experiments for this paper cycle.

Immediate next action:

```text
Freeze the result set and build the writing package:
1. final_main_tables.md
2. final_claim_boundary.md
3. paper_outline_improvement_position.md
4. abstract_and_intro_draft.md
```

Do not train V26 unless a new paper branch is opened.
