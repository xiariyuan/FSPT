# Route-D safe long-occlusion re-detection interface result — 2026-07-18

## 1. Decision

The official alignment, metric-parity, causal-interface, and immutable-data
protocol gates pass. The implemented recovery branch is authorized for cache
export from a clean committed checkout.

```text
ALLOW_COMMITTED_POINTODYSSEY_EVENT_CACHE_EXPORT
```

This milestone is an interface and protocol result. It is not learned
performance and does not authorize reading PointOdyssey internal holdout,
PointOdyssey test, or rerunning the frozen official 1,144-video Kinetics result.

## 2. Correct research scope

The module is a **risk-controlled long-occlusion re-detection branch for a
frozen point tracker**. It is not described as a general coordinate corrector,
and it is not yet claimed as a universal plug-in.

The current implementation is a CoTracker3 instantiation of a
backbone-independent decision contract:

- candidate 0 is always the native tracker output;
- alternative global candidates may be accepted only after risk-aware
  comparison, explicit abstention, and two-frame confirmation;
- coordinate and visibility are recovered jointly;
- appearance memory is updated only from reliable native observations or a
  confirmed recovery;
- state writeback is bounded and becomes effective only on a future tracker
  frame.

A second architecturally different backbone remains mandatory before a general
framework claim.

## 3. Official CoTracker3 alignment

The unmodified official source is pinned to Meta CoTracker commit:

```text
82e02e8029753ad4ef13cf06be7f4fc5facdda4d
```

The official scaled-online checkpoint SHA-256 is:

```text
205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218
```

Exact official configuration:

```text
benchmark:    TAP-Vid-DAVIS first-query
single_point: true
window length: 16
iterations:    6
seed:          0
```

Result:

| Metric | Official replication |
|---|---:|
| Average Jaccard | **64.4408** |
| Occlusion accuracy | **90.8941** |
| Average threshold accuracy | **77.1575** |

This passes the published-range alignment gate. DAVIS has historical project
exposure, so it is an official alignment and frozen development benchmark, not
a newly untouched final test.

Generated evidence:

```text
docs/generated/OFFICIAL_COTRACKER3_DAVIS_FIRST_ALIGNMENT_2026-07-18.json
SHA-256: ccf0eaaab8bd402d6d071a82faa0c12fc6a84263dfdf0bff2a9bd1e72017dc57
```

## 4. Official AJ_RD parity

The public TAPNext++ re-detection metric is pinned to DeepMind `tapnet` commit:

```text
989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc
```

Official source:

```text
tapnet/tapnextpp/metrics/aj_rd.py
SHA-256: 4b5f56bc04f4af5905108e999f1a79880cd7b1ae218e76b9a686c640d23b4c1b
```

The Route-D implementation reproduces every scalar from 100 randomized,
multi-event comparisons exactly:

```text
maximum absolute difference: 0.0
```

Generated evidence:

```text
docs/generated/OFFICIAL_TAPNEXTPP_AJRD_PARITY_2026-07-18.json
SHA-256: e8f4d6f9fdc552108162f49903a6cef023854319ca5834bac408c3d73908050f
```

## 5. Official full-call versus writable streaming runtime

The official full-video evaluation call and the official stateful online call
are different execution graphs. The latter is required for causal state
writeback. On four real DAVIS queries, zero-step Route-D gives:

```text
visibility decisions:       exact
maximum coordinate delta:   0.0178833 pixel
mean coordinate delta:      0.0013415 pixel
interventions:               0
state writes:                0
```

The coordinate discrepancy is below the frozen `0.02`-pixel numerical-equivalence
boundary and far below the strictest one-pixel TAP threshold. Every learned
comparison will use the same streaming runtime and its separately frozen native
baseline.

Generated evidence:

```text
docs/generated/ROUTED_SAFE_REDETECTION_RUNTIME_ALIGNMENT_2026-07-18.json
SHA-256: dbe21f9cfd3f7058086c880e10f05be19f35825a141d94807fa412393e97744a
```

## 6. Architecture

| Component | Parameters |
|---|---:|
| Multi-anchor dense proposal generator | 12,258 |
| Risk-aware candidate comparator | 239,915 |
| **Total recovery branch** | **252,173** |
| Native CoTracker3 | frozen |

Frozen architecture contract:

```text
appearance slots:             4
non-native proposals:         5
hard NMS radius:              1 feature cell
comparator hidden dimension:  96
comparator layers:            2
confirmation length:          2 frames
confirmation radius:          8 pixels
maximum future write step:    32 pixels
```

Memory slots consist of the immutable query anchor, the last reliable appearance
EMA, and two episodic anchors. Low-visibility or low-confidence native positions
cannot update memory. Hard candidate extraction is detached from comparator
backpropagation; the proposal branch is trained through a separate dense loss,
preserving strict CUDA replay.

## 7. PointOdyssey protocol

Scene and event selection are determined only by frozen SHA-256 ordering. Events
follow the exact AJ_RD record-breaking invisibility-duration rule.

| Partition | Scenes | Events | Use |
|---|---:|---:|---|
| Fit | 24 | 240 | gradients only |
| Model validation | 8 | 160 | checkpoint selection only |
| Locked internal holdout | 7 | unread | one-time gated evaluation |
| Locked PointOdyssey test | 13 | unread | untouched synthetic test |

Each fit and model-validation scene contributes fixed quotas from all five
invisibility buckets:

```text
1–3, 4–15, 16–63, 64–255, and >=256 frames
```

Protocol evidence:

```text
configs/routeD_safe_redetection_pointodyssey_protocol_v0.json
file SHA-256:
f7fad53e06d238025978dac812a6c0ce398ccff1bba1615e4fece27c26eadb44
```

Ground truth selects training events and supplies labels. Runtime memory,
proposal, confirmation, abstention, visibility recovery, and writeback remain
causal and ground-truth-free.

## 8. Interface and determinism tests

The interface tests verify:

- exact zero-step native coordinate and visibility fallback;
- query independence under perturbation of another query;
- reliable-only appearance-memory updates;
- two-frame recovery confirmation;
- bounded, future-only state writeback;
- record-breaking reappearance labels;
- finite, separated proposal and comparator gradients;
- official internal support-grid construction;
- official AJ_RD exact parity;
- deterministic cache tensors across independent CUDA processes.

Regression result:

```text
new interface tests: 15 passed
complete Route-D tests: 132 passed
```

## 9. Formal training gates

Before any locked data may be read, the fit/model-validation result must pass:

```text
candidate-oracle AJ_RD gain:        >= +0.05
learned AJ_RD gain:                 >= +0.01
paired event AJ gain CI lower:      > 0
standard event AJ regression:       >= -0.002
visibility-accuracy regression:     >= -0.002
harmful intervention rate:          <= 1%
false-reacquisition increase:       <= 0.2 percentage point
16px severe-error rate:             not worse
seed-17 replay:                      exact
```

The official Kinetics 1,144-video result remains frozen and will not be rerun or
used for tuning.

## 10. Next authorized operation

Formal PointOdyssey event-cache export must begin only from a clean committed
checkout. Every sidecar records and is verified against:

```text
Git HEAD
clean-worktree state
cache-builder SHA-256
cache-module SHA-256
model-module SHA-256
protocol-module SHA-256
protocol SHA-256
official checkpoint SHA-256
```

Only after complete fit and model-validation caches pass exact replay may formal
seed-17 training start.
