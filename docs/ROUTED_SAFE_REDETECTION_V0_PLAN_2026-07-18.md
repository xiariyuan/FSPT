# Route-D safe long-occlusion re-detection v0 — 2026-07-18

## Research question

Can a frozen online point tracker recover a point after a long invisible interval
without paying for that recovery through destructive global jumps on ordinary
visible frames?

The new direction is **risk-controlled global re-detection**, not generic
coordinate correction. The primary event metric is the public TAPNext++
`AJ_RD`, and standard TAP-Vid AJ/OA remain mandatory no-regression metrics.

## Official alignment gate

Before implementing or training the recovery model, the unmodified official
CoTracker3 code was fixed at commit
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d` and evaluated using the exact
scaled-online checkpoint, the official TAP-Vid-DAVIS first-query loader and
evaluator, and `single_point=True`.

```text
DAVIS first-query AJ:            64.4408
occlusion accuracy:              90.8941
average threshold accuracy:      77.1575
```

DAVIS has historical project exposure, so this is a baseline-alignment and
later development benchmark, not a newly untouched final test.

The official full-video call and the official stateful streaming call are two
different execution graphs. On four real DAVIS queries, zero-step Route-D uses
the official streaming call and matches full-call visibility exactly; maximum
coordinate difference is `0.0178833` pixel and mean difference is `0.0013415`
pixel, with zero interventions and zero writes. The frozen streaming baseline
will therefore be evaluated separately and used for every method comparison.

## Official re-detection metric

The metric source is pinned to DeepMind `tapnet` commit
`989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc`, file
`tapnet/tapnextpp/metrics/aj_rd.py`, SHA-256
`4b5f56bc04f4af5905108e999f1a79880cd7b1ae218e76b9a686c640d23b4c1b`.

The Route-D implementation reproduces every scalar from 100 randomized
multi-event comparisons with maximum absolute difference `0.0`.

## Model

The native CoTracker3 trajectory, visibility, confidence and candidate 0 are
frozen. The recovery branch contains:

1. query, last-reliable and episodic appearance anchors;
2. global multi-anchor feature correlations;
3. a learned dense residual proposal map;
4. native plus five deterministic top-K/NMS candidates;
5. monotonic multi-threshold utility, catastrophic risk, reappearance,
   visibility, writeback and abstention heads;
6. two-frame spatial confirmation;
7. exact native fallback;
8. bounded writeback that becomes effective only on a later tracker frame.

Memory updates are allowed only after strong native evidence or a confirmed
recovery. Predicted low-confidence or invisible native positions cannot pollute
appearance memory.

## PointOdyssey training protocol

PointOdyssey supplies real long sequences and visibility transitions. Scene
membership and event membership are selected only by fixed SHA-256 ordering.
Eligible events use the exact AJ_RD record-breaking-duration definition.

```text
fit:              PointOdyssey train scenes; gradients only
model-validation: disjoint PointOdyssey val scenes; checkpoint selection only
internal holdout: remaining val scenes; annotations semantically unread
PointOdyssey test: locked and unread
Kinetics 1,144:   frozen; never rerun or used for tuning
```

Event-centric training windows are 512 or 1024 frames. Ground truth selects the
training event and supplies labels, but runtime memory updates, proposals,
confirmation and writeback are causal and ground-truth-free.

Frozen protocol:

```text
fit:                         24 scenes / 240 events
model-validation:             8 scenes / 160 events
locked internal holdout:      7 scenes
locked PointOdyssey test:    13 scenes
protocol file SHA-256:
f7fad53e06d238025978dac812a6c0ce398ccff1bba1615e4fece27c26eadb44
```

## Formal gates

A model is eligible only when all of the following hold:

```text
official CoTracker3 baseline provenance: exact
AJ_RD parity with official implementation: exact zero difference
zero-step coordinate and visibility equality: exact
candidate 0/native equality: exact
query independence: exact
seed-17 replay: exact
candidate-oracle AJ_RD gain: >= +0.05
learned AJ_RD gain: >= +0.01
standard AJ and OA regression: no worse than -0.002
harmful intervention: <= 1%
false-reacquisition increase: <= 0.2 percentage point
16px severe-error rate: not worse
```

Only after these gates may the locked PointOdyssey holdout be read once. A
second architecturally different backbone is required before claiming a general
plug-in framework.
