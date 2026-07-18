# ReQueryTAP Model-Level Feasibility Result — 2026-07-18

## Decision

```text
Gate 0 PyTorch backward: PASS
Gate A fresh-query oracle: PASS
ALLOW_REQUERYTAP_ARCHITECTURE_IMPLEMENTATION
NO PERFORMANCE TRAINING YET
NO MODEL VALIDATION OR LOCKED DATA
```

## Gate 0 — real backward pass

A four-frame PointOdyssey fit clip was processed by the official TAPNext++
PyTorch model in training mode.  The final recurrent block, encoder norm,
coordinate head, visibility head, and point-query token were trainable.

```text
trainable parameters:             16,711,937
loss:                              0.246292
last-block gradient norm sum:      2.518443
coordinate-head gradient sum:      4.262914
visibility-head gradient sum:      3.940147
point-query-token gradient norm:   0.363735
changed tensors after AdamW:       54 / 54
peak allocated GPU memory:         2.685 GiB
```

All loss, gradients, and parameter updates were finite.

## Gate A — fresh query after synthetic long gap

All 17 qualified PointOdyssey fit scenes were evaluated with 8 visible frames,
32 deterministic corruption frames, and 16 post-gap frames.  The query frame
was excluded from the primary score; only the following 15 future frames were
measured.

### Fresh query at native coordinate

```text
positive scenes:       0 / 17
median gain:           -8.8406 px
mean gain:             -24.5079 px
95% bootstrap CI:      [-43.1318, -10.0457] px
```

Resetting the token at the already-drifted native coordinate is uniformly
harmful.  State reset alone is rejected.

### Fresh query at clean-teacher coordinate

```text
positive scenes:       13 / 17
median gain:           +1.7874 px
mean gain:             +18.0981 px
95% bootstrap CI:      [+5.7439, +32.9847] px
```

A strong coordinate estimate usually enables recovery but remains harmful in
four scenes.  Coordinate quality and exact point identity remain the bottleneck.

### Fresh query at GT coordinate — primary oracle

```text
positive scenes:       16 / 17
median gain:           +6.8431 px
mean gain:             +20.0055 px
95% bootstrap CI:      [+7.9793, +34.3044] px
median improved frames: 100%
worst scene:           -0.0550 px
```

Every frozen gate passed.

## Scientific conclusion

The prior state-repair experiments failed because recurrent hidden state is
scene-conditioned and non-transferable.  The fresh-query result shows a cleaner
mechanism: once the exact physical point is re-localized, creating a new
transient query state restores future tracking across scenes.

Therefore the model must learn:

```text
persistent exact-point identity
→ global re-localization after reappearance
→ retire corrupted transient track token
→ instantiate a fresh transient token
→ optimize its future rollout
```

It must not learn:

```text
copy old state
blend hidden state
restart at native drifted coordinate
post-hoc replace outputs
```

## Novelty boundary

Persistent queries, global image matching, streaming memory, and long-sequence
occlusion training already exist in recent literature.  The intended
contribution is narrower:

> An online arbitrary-point tracker with explicit transient query-token
> lifecycle, where persistent point identity globally rebinds to the image and
> instantiates a fresh recurrent track token after long occlusion.

The identity representation must distinguish exact nearby points on the same
surface, not merely semantic object parts.  Training must use paired clean and
occluded streams, same-surface hard negatives, a differentiable rebind
location loss, and future fresh-token rollout supervision.
