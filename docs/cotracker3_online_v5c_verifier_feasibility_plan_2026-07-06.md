# CoTracker3 Online V5-C Lightweight Verifier Feasibility Plan — 2026-07-06

## Motivation

V5-B hard-event candidate-pool oracle showed moderate headroom:

```text
local96_s8_top20 selective oracle on first10:
AJ +0.0856
OA +0.2038
AJ_RD +0.0050
AJ_RD_256 +0.0106
```

But this uses GT to decide whether to accept a candidate. V5-C asks whether a lightweight verifier can recover a useful fraction of this oracle without using GT at inference.

## Experiment status

Diagnostic only. Not a method result until it is validated under video-level split and then re-evaluated without oracle decisions.

## Dataset construction

Base:

```text
CoTracker3 true-streaming native first10 hard-event set from V5-B
```

Candidate pool:

```text
local96_s8 top20 candidates per hard event
```

Each sample:

```text
one candidate point for one hard re-entry event
```

Candidate features must be GT-free:

```text
DINO cosine score
DINO rank
score gap / margin
candidate distance to native
native visibility
native score
support age
candidate y/x normalized
event frame normalized
query index normalized
```

Labels may use GT because this is supervised diagnostic:

```text
candidate_err_px
native_err_px
candidate_better = candidate_err + 2px < native_err
candidate_safe8 = candidate_err <= 8px
candidate_good = candidate_better AND candidate_safe8
```

Forbidden feature leakage:

```text
GT coordinate
candidate_err
native_err
candidate_better
candidate_safe8
candidate_good
```

## Validation protocol

Use video-level split, not random candidate split.

First audit:

```text
leave-one-video-out over first10 videos with hard events
```

Models:

```text
LogisticRegression
ExtraTreesClassifier
RandomForestClassifier
```

## Decision gate

Continue only if verifier achieves:

```text
accepted candidate precision >= 70%
accepted candidates >= 15 on first10 or proportional full30
simulated AJ_RD gain >= +0.0025
simulated AJ_RD_256 gain >= +0.005
AJ/OA non-negative or nearly non-negative
```

If not, stop CoTracker3 online V5 as a mainline and keep the result as appendix/diagnostic.
