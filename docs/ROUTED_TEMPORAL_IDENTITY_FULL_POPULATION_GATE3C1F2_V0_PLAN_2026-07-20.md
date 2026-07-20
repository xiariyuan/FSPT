# Route-D full-population confirmation Gate 3C1F2 v0 plan — 2026-07-20

## Status

```text
PREREGISTERED_NOT_RUN
third-population tracking metrics unread
```

## Question

Does the frozen causal two-stage recovery pipeline improve complete-video TAP metrics on a third 128-video Kubric population that is raw-record-disjoint from every video used to design or validate the entry model and top-1 selector?

## Authorized parent

Gate 3C1F1 completed with exact raw-identity qualification:

```text
videos:                    128
shards:                    8
excluded prior identities:1,024
raw overlap:               zero
selected identity digest:  85193d0aa6381c7d78442e85bf87750ce72a497995a92016001ca0cf51832f28
formal decision:           AUTHORIZE_GATE3C1F2_FULL_POPULATION_CONFIRMATION_PREREGISTRATION
```

No tracking metric from these 128 videos has been read.

## Frozen pipeline

For every video, all 64 original query points are first tracked through frame 15 by frozen CoTracker3. Points with original query frame `<8` enter the causal entry model. Entry features use only observed frames 0--15:

- eight-frame trajectory features;
- native frame-15 visibility and confidence;
- all four native track-feature levels;
- all four native support-memory levels.

The frozen entry rule is:

```text
entry probability >= 0.93
native visibility * confidence <= 0.02
both conditions required
```

Only entry-positive points pay the candidate-generation cost. Their frozen candidate pipeline is:

```text
geometry-preserving four-level score map
native candidate plus 128 nonnative candidates
reverse observed-window tracklets
DINOv3 temporal identity descriptors
native plus eight query-closure shortlist
expected-distance top-1 ranking
support >=0.30
predicted value >=1.0 px
predicted harm <=0.20
```

Only final top-1 actions write frame-15 coordinate plus deterministic four-level feature/support memory. Visibility and confidence are not manually overwritten; the second-window CoTracker state produces them naturally.

## Causal order

For each video:

1. Run native observed frames 0--15.
2. Freeze eligible point identities.
3. Compute entry features, probabilities, and entry mask.
4. Generate candidates only for the frozen entry mask.
5. Freeze top-1 selected slot, support/value/harm predictions, action mask, and output candidate.
6. Run native and modified frames 16--23.
7. Only after all actions are frozen, read GT for complete-video metrics and diagnostic action effects.

Future GT cannot affect entry, candidate coordinates, shortlist, top-1 predictions, action, memory writeback, visibility, or confidence.

## Complete-video metrics

Metrics use the existing project TAP-Vid first-query evaluator at raster 256:

```text
AJ
average points within thresholds (delta_avg)
occlusion accuracy (OA)
```

Native and modified visibility use the same frozen rule:

```text
sigmoid(visibility logit) * sigmoid(confidence logit) > 0.6
```

Primary aggregation is equal-video mean across all 128 videos. Paired 95% CIs resample videos with 10,000 bootstrap samples and frozen seeds `271828`, `271829`, and `271830`.

## Frozen pass gates

All gates must pass in a fresh-process exact replay:

```text
videos exactly:                              128
entry trigger rows:                          >=128
top-1 action rows:                           >=64
action-support videos:                       >=32
actions with visible future GT:              >=64

AJ mean gain:                                >=0.0005  (+0.05 points)
AJ paired-video CI lower:                    >0
AJ nonnegative-video fraction:               >=0.75

delta_avg mean gain:                         >=0.0025  (+0.25 points)
delta_avg paired-video CI lower:             >0

OA mean gain:                                >=0.0025  (+0.25 points)
OA paired-video CI lower:                    >0

action future mean error reduction:          >=8 px
action future positive fraction:             >=0.85
action harmful-by-more-than-4px fraction:     <=0.05
exact replay:                                true
```

The AJ gate is deliberately stronger than the exposed 16-video pilot, whose AJ mean was positive but whose CI crossed zero.

## Replay contract

Primary and replay use separate work roots and rerun CoTracker, DINOv3, reverse tracklets, entry predictions, candidate generation, top-1 predictions, state writeback, complete-video coordinates, visibility, and metrics. Exact replay requires equality of:

- all 128 per-video scientific digests;
- complete metric summaries and CIs;
- aggregate video-record digest;
- final scientific payload digest.

Per-video sidecars may resume an interrupted primary or replay within its own work root, but replay cannot read primary sidecars.

## Decisions

```text
PASS:
AUTHORIZE_GATE3C1F3_OFFICIAL_TAPVID_PREREGISTRATION

FAIL:
STOP_BEFORE_OFFICIAL_TAPVID
```

A pass still does not establish a paper-table result. It authorizes only a separately committed official TAP-Vid protocol. DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.
