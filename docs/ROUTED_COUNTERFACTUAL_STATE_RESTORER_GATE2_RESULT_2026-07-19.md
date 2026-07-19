# Route-D CSRR Gate 2 learned-restorer result — 2026-07-19

## Formal decision

```text
STOP_LEARNED_STATE_RESTORER_BEFORE_MODEL_VALIDATION
```

The frozen Gate 2 training protocol completed in primary and independent replay.
The selected epoch, complete training history, fit-internal validation outputs,
checkpoint metadata, model-state digest, and every nested checkpoint tensor are
exactly reproducible. The learned branch does not pass the preregistered
fit-internal scientific and action gates, so source indices `48–63` remain
locked.

This is a fit-only negative result for the current CSRR v0 learned module. It is
not evidence against the Gate 1 oracle state-restoration mechanism, which remains
strong, and it is not a model-validation, holdout, DAVIS, or Kinetics result.

## Frozen execution identity

```text
config SHA256:                    b294498e1ea8c668c6197d3adbf747a1bb54794b4e6839c76cbaaf2f806609c2
seed:                             17
train source indices:             8–31
fit-internal validation indices:  32–47
train cache index SHA256:         ab93e070adc4d623aeada811a88b16def3f4bb2137946f3728327806ae331517
validation cache index SHA256:    4216898d32c65a60360087bd7c4026d2db7d6877bfa06a934932a429112e4bef
trainable parameters:             19,685
selected epoch:                   6
checkpoint SHA256:                537f1819e923bdc2a68d5ebc15d647e60ea39afb4225d4bd04e2bb8c68913257
model-state digest:               c5d7b126ff58507c3003a84d171d344e77d489de9ab30c996ff798a38925c462
```

Early stopping triggered after epoch 10 with patience 4. The selected checkpoint
is epoch 6 under the frozen score and tie-break rules.

## Independent replay

```text
reports exact excluding output paths:               true
checkpoint nested tensors and metadata exact:       true
model state exact:                                   true
model-state digest exact:                            true
primary checkpoint SHA == replay checkpoint SHA:    true
```

Primary and replay report files have different serialization hashes because the
embedded output paths differ, but the formal packager confirms all scientific
content is exact.

## Forced failure action

The forced-action audit isolates restoration quality from the learned action
gate. It applies each learned variant to the same 80 preregistered natural
failure rows from all 16 fit-internal validation videos.

| Variant | Mean future error | Severe >=16 px | Threshold utility |
|---|---:|---:|---:|
| native | 55.36197 px | 0.99040 | 0.00192 |
| learned coordinate only | 54.29321 px | 0.98750 | 0.00250 |
| learned coordinate + probability | 53.33695 px | 0.98594 | 0.00281 |
| learned full structured state | 47.44938 px | 0.84375 | 0.04719 |

Full structured state versus native:

```text
mean-error reduction:            +7.91259 px
95% bootstrap CI:                [+2.13986, +13.32411] px
threshold-utility gain:          +0.04527
95% bootstrap CI:                [+0.02250, +0.07156]
positive-point fraction:          0.70000
severe >=16 px rate reduction:   +0.14665
```

This is a real learned signal: the error-reduction and utility-gain confidence
intervals are strictly positive, 70% of failure points improve, and full state
substantially outperforms both learned coordinate controls.

Full state versus learned coordinate only:

```text
mean-error reduction:            +6.84383 px
threshold-utility gain:          +0.04469
better-video fraction:            0.62500
```

Full state versus learned coordinate + probability:

```text
mean-error reduction:            +5.88756 px
threshold-utility gain:          +0.04438
```

These controls confirm that re-extracted support memory contributes material
future-rollout value beyond committing a learned coordinate or probability.

## Failed preregistered restoration gates

```text
required mean-error reduction >= 8.0 px:       observed 7.91259      FAIL
required utility gain >= 0.12:                 observed 0.04527      FAIL
required utility CI lower >= 0.03:             observed 0.02250      FAIL
required severe-rate reduction >= 0.15:        observed 0.14665      FAIL
```

The mean-error and severe-rate gates are narrowly missed, but the utility gates
are missed by a substantial margin. The learned correction reduces large errors
more often than it converts failures into genuinely accurate tracks under the
`1/2/4/8/16 px` utility metric.

## Learned action gate collapse

On the fixed union of 80 failure and 80 clean rows:

```text
failure apply recall:                    0.00000
clean false-apply rate:                  0.00000
clean harmful rate:                      0.00000
union threshold-utility gain:            0.00000
union mean-error reduction:              0.00000077 px
```

The gate selected no intervention on any row. It therefore achieved perfect
clean safety by collapsing to exact native no-op rather than learning a useful
risk-controlled action policy.

The zero-action native state and future parity check remains exact. This rules
out an evaluator or fallback-state corruption explanation for the null union
result.

## Passed controls and integrity gates

```text
validation failure support:                         PASS
validation clean support:                           PASS
error-reduction CI lower >= 2 px:                   PASS
positive-point fraction >= 0.65:                    PASS
full state > coordinate-only error/utility:         PASS
full state > coordinate-probability error/utility:  PASS
full state better-video fraction >= 0.60:           PASS
clean false-apply safety:                           PASS
clean harmful safety:                               PASS
zero-action native parity exact:                    PASS
independent replay exact:                           PASS
```

## Scientific interpretation

Gate 2 separates the current failure into two components:

1. **Structured restoration is learnable but still too inaccurate.** The
   predicted location plus frozen feature re-extraction provides statistically
   positive future improvement and materially beats coordinate-only controls,
   but the remaining coordinate errors are too large to cross the strict
   utility thresholds reliably.
2. **The action objective is not usable.** A single symmetric BCE action head,
   trained jointly with reconstruction and strong clean no-op penalties, finds a
   stable all-negative solution. The fixed 0.5 threshold then prevents the
   learned restoration signal from being exercised at all.

The result rejects the current CSRR v0 architecture/loss/action design. It does
not reject complete-state restoration as a research direction.

## Formal boundary

No authorization is granted to read source indices `48–63`. Calibration, final
holdout, TAP-Vid DAVIS, TAP-Vid Kinetics, and the official 1,144-video Kinetics
run remain untouched.

Canonical summary SHA256:

```text
fcb0219293b39c0d8842c74a3a0ffe405687f34837fcf8403a448c64934ca1ff
```
