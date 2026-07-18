# ReQueryTAP Architecture Integrity Result — 2026-07-18

## Architecture

ReQueryTAP is a strict single-query lifecycle model around a trainable TAPNext++
backbone:

```text
immutable exact-point identity memory
+ current image patch tokens
→ global differentiable rebind locator
→ rebind coordinate + respawn logit
→ retire old recurrent track state
→ initialize a completely fresh TAPNext query state
```

No hidden-state tensor is copied, blended, or injected into the fresh branch.
The same backbone remains part of the model and can be trained end to end.

## Synthetic interface tests

```text
native-off exactness:                    PASS
forced fresh equals standalone fresh:   PASS
predicted fresh path differentiable:     PASS
multi-query input rejected:              PASS
```

## Real TAPNext++ checkpoint smoke

Using a real PointOdyssey fit clip and the 2.4 GB official checkpoint:

```text
native-off tracks max difference:          0
native-off track logits max difference:    0
native-off visibility max difference:      0
native-off recurrent-state max difference: 0

forced fresh tracks max difference:          0
forced fresh track logits max difference:    0
forced fresh visibility max difference:      0
forced fresh recurrent-state max difference: 0
```

The locator received finite nonzero gradients through the fresh TAPNext query:

```text
identity projection gradient norm sum: 628.272
image projection gradient norm sum:    642.179
offset head gradient norm sum:           9.975
respawn head gradient norm sum:           0.123
```

```text
locator parameters:        263,427
peak allocated GPU memory: 2.443 GiB
Gate B:                    PASS
```

## Decision

```text
ALLOW_FIT_ONLY_LOCATOR_TRAINING
NO POINTODYSSEY MODEL-VALIDATION
NO HOLDOUT / TEST / DAVIS METHOD EVAL / KINETICS
```

The next stage must train global exact-point re-localization on a frozen fit-only
scene split, then evaluate whether predicted-coordinate respawn recovers future
tracking.  Locator coordinate metrics alone cannot promote the method.
