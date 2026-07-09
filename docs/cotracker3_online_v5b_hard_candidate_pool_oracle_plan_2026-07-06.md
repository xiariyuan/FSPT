# CoTracker3 Online V5-B Hard-Event Candidate-Pool Oracle Plan — 2026-07-06

## Why V5-B

Previous CoTracker3 true-streaming online attempts show:

```text
V3/V4 state-writeback is safe but has no meaningful AJ_RD gain.
Current re-entry-adjacent visibility candidate oracle gives only AJ_RD +0.0011.
Local DINO patch search around native coordinates did not beat native on easy/near-correct events.
```

Therefore the next question is not whether to tune visibility writeback, but whether a larger active re-detection candidate pool has enough oracle headroom when native CoTracker3 actually fails.

## Diagnostic question

```text
On hard re-entry events where native CoTracker3 is invisible/low-confidence or spatially wrong, can a DINO memory-conditioned candidate pool contain a better point than native?
```

## Protocol

This is an oracle diagnostic only. GT may be used for event selection and top-k oracle evaluation. Results are not deployable method results.

## Smoke design

Baseline:

```text
CoTracker3 true-streaming native full30 cache
```

Event selection:

```text
GT re-entry event
GT visible at event frame
native predicted invisible/low score OR native error >= threshold
```

Candidate pools:

```text
local64_s8:  local grid centered on native coordinate, radius 64 px, stride 8 px
local96_s8:  local grid centered on native coordinate, radius 96 px, stride 8 px
global_s16:  whole-frame coarse grid, stride 16 px
```

Feature:

```text
DINOv3 patch CLS cosine against last reliable visible support patch
```

Outputs:

```text
top-k recall@4/8/16 px
oracle replacement metrics AJ/OA/AJ_RD/AJ_RD_256
per-event error table
```

## Continue gate

Continue to verifier training only if:

```text
hard-event top10 candidate recall@8px clearly exceeds native recall@8px
oracle AJ_RD gain >= +0.01 on the evaluated subset or shows a convincing trend on hard events
AJ/OA does not collapse
```
