# Route-D CSRR teacher-cache smoke result — 2026-07-19

## Formal decision

```text
ALLOW_COMPLETE_GATE2_TEACHER_CACHE_BUILD
```

The four fixed Gate 2 cache anchors pass primary/replay construction and exact
nested-artifact comparison.

```text
source indices: [8,31,32,47]
completed:      4 / 4
failure rows:   12
clean rows:     28
```

This result authorizes complete cache construction only for training indices
`8–31` and fit-internal validation indices `32–47`. It is not a learned-model or
performance result.

## Cache contract

Each sidecar contains two separate views:

```text
exact native rollout state: float32
CSRR model/teacher view:     float16
continuation frames 8–23:    uint8
```

The exact view stores all 64 point slots, original predictor queries, four track
feature levels, four 7 x 7 support-memory levels, coordinates, visibility logits,
and confidence logits. The model view stores normalized coordinates,
visibility/confidence probabilities, nine-dimensional trajectory features,
selected native/teacher memories, frame-15 feature pyramid, fixed row identities,
and future scoring metadata.

## Quantization gate

Across all four anchors:

```text
maximum absolute error:       0.0004826784  < 0.001
minimum cosine similarity:    0.9999974966  > 0.99999
```

The first implementation attempted to apply the same absolute-error gate to raw
0–255 coordinates and large logits. That invalid source-8 sidecar failed before
a report or index was written. Commit `906cc6d` corrected the model view to the
already intended normalized-coordinate/probability representation while keeping
the float32 raw state exact. The invalid sidecar was deleted and all anchors were
rebuilt from scratch after the fix.

## Row availability

```text
source 8:   8 failure, 8 clean
source 31:  0 failure, 16 clean
source 32:  3 failure, 3 clean
source 47:  1 failure, 1 clean
```

Source 31 genuinely contains no point satisfying the frozen natural-failure
threshold. No below-threshold point was added. The protocol balances classes only
when both are available.

## Independent replay

All replay gates pass:

```text
index exact excluding paths/serialization fields: true
all nested artifacts exact:                       true
all reports exact excluding paths:                true
per-video sidecar SHA exact:                       true
```

Hashes:

```text
primary index SHA256:
8012a7bb89cd58b6520f4862b191d880bc93f203e0d73618badaf4393432ac9f

replay index SHA256:
7362688f6902d1c10c02deb5bcbc024f5f5e9197d0776161826fe67a95c38b48

combined tensor digest:
1c2f5234aa458d8377aba56fc6f30310a6e14fc4b3a5bb240e26fff1fcef71de

canonical summary SHA256:
584ef5e6be13efced9181a9488819b493acf69859393b06ff30ae2316405860c
```

## Locked data

No source index `48–63` was read. Calibration, final holdout, DAVIS, and Kinetics
remain locked. The official 1,144-video Kinetics evaluation was not rerun.
