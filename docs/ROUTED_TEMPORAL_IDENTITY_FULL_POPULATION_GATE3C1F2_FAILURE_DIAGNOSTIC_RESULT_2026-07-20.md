# Route-D Gate 3C1F2 full-population failure diagnostic result — 2026-07-20

## Status

```text
COMPLETED_EXPOSED_DIAGNOSTIC
SEALED_GATE3C1F2_PIPELINE_REPRODUCED_EXACTLY
```

This diagnostic reran only the 44 exposed Gate 3C1F2 action videos. Every entry, candidate, shortlist, top-1 action, output candidate, native/modified coordinate, and native/modified visibility digest reproduced the committed exact-replay result. No new confirmation or external data was read.

## Core result

The Gate 3C1F2 AJ bottleneck is a **coordinate--visibility coupling problem**, with a strong category-specific visibility failure on `other` actions.

### Equal-video action-video metrics

| View | AJ | delta_avg | OA |
|---|---:|---:|---:|
| native | 0.25278591 | 0.37392140 | 0.83030555 |
| actual modified | 0.25282712 | 0.37664580 | 0.83493434 |
| modified coordinates + native visibility | 0.25272304 | 0.37664580 | 0.83030555 |
| native coordinates + modified visibility | 0.25059102 | 0.37392140 | 0.83493434 |
| native coordinates + GT visibility oracle | 0.26222635 | 0.37392140 | 1.00000000 |
| modified coordinates + GT visibility oracle | 0.26514752 | 0.37664580 | 1.00000000 |

### Paired AJ decomposition

```text
actual modified - native:
  mean  +0.00004121
  CI    [-0.00099980, +0.00125868]

modified coordinates + native visibility - native:
  mean  -0.00006287
  CI    [-0.00033298, +0.00018545]

native coordinates + modified visibility - native:
  mean  -0.00219489
  CI    [-0.00313342, -0.00133984]

modified - native coordinates under GT visibility:
  mean  +0.00292117
  CI    [+0.00139440, +0.00478191]
```

In percentage-point units, the GT-visibility localization oracle shows a significant `+0.2921` AJ-point coordinate gain on action videos. Native visibility prevents that coordinate gain from contributing to Jaccard. Modified visibility is required to expose recovered coordinates, but applying modified visibility to unrecovered native coordinates causes a significant `-0.2195` AJ-point loss.

## Affected-frame localization

Across action frames 15--23 with GT visibility:

```text
failure rows, visible frames: 335
native within 16 px:           1.19%
modified within 16 px:        65.37%
modified within 8 px:         25.37%
modified within 4 px:         13.13%

ambiguous rows, visible frames:49
native within 8 px:           10.20%
modified within 8 px:         38.78%

other rows, visible frames:    21
native within 16 px:          47.62%
modified within 16 px:        80.95%
```

The coordinate writeback is therefore materially improving affected-frame localization, especially on formal failure rows.

## Visibility transitions

```text
recovered GT-visible false negatives: 268
new GT-occluded false positives:      186
new GT-visible false negatives:         0
removed GT-occluded false positives:    0
```

By sealed GT-defined action category:

| Category | Actions | Recovered visible FN | New occluded FP |
|---|---:|---:|---:|
| failure | 38 | 220 | 2 |
| ambiguous | 7 | 42 | 8 |
| other | 44 | 6 | 176 |

The modified visibility behavior is highly beneficial on failure actions, moderately beneficial on ambiguous actions, and strongly over-visible on `other` actions. Thirty-eight of the 44 `other` actions have no visible GT frame in frames 16--23.

## Pooled affected-frame Jaccard

Average pooled Jaccard across thresholds 1/2/4/8/16 on affected action frames:

| Category | Native | Actual modified | Modified coordinates + GT visibility |
|---|---:|---:|---:|
| all | 0.0000 | 0.09245 | 0.15268 |
| failure | 0.0000 | 0.11759 | 0.14113 |
| ambiguous | 0.0000 | 0.19487 | 0.24810 |
| other | 0.0000 | 0.00603 | 0.17811 |

The largest unrealized Jaccard gap is on `other` actions: coordinates contain useful recovery signal, but predicted visibility activates mostly on GT-occluded frames.

## Scientific interpretation

The result rules out two simplistic diagnoses:

1. **The coordinate selector is not the main failure.** Under GT visibility, coordinate improvement is significant, and affected-frame threshold accuracy rises sharply.
2. **Preserving native visibility is not a solution.** Native visibility is almost always off on affected rows, so it suppresses recovered coordinates and yields approximately zero affected-frame Jaccard.

The next design should preserve the frozen coordinate/memory action and learn a causal, post-writeback visibility decision that distinguishes recovered visible states from over-visible `other` states. It should use modified and native visibility/confidence jointly with sealed entry/top-1 evidence and state-change features. Merely lowering or raising a global visibility threshold is not yet justified.

Any such redesign may use this exposed 128-video population for development, but must be confirmed on a new raw-record-disjoint population. No official TAP-Vid evaluation is authorized.

## Integrity

```text
videos:                       44
sealed actions:               89
sealed pipeline exact:        true
frame-record digest:          6f378af3b98e987b5d77af63198647e5d4d6bc0115686d2a6811d5f910522591
scientific payload SHA256:    8a73a832391c3cfd1cdb017c538a0f1988e6eb97a0ecd5008b059fdb89152f9a
result payload SHA256:        82c653b31bb78c3b5d876bb6a113657f5279c135250bf599d499f8410ce5b881
```

DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.
