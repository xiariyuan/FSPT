# Route-D deployable causal future rollout Gate 3C1E v0 result — 2026-07-20

## Formal decision

```text
COMPLETED_PASS
EXACT_REPLAY_PASS
AUTHORIZE_GATE3C1F_OFFICIAL_TAPVID_BENCHMARK_PREREGISTRATION
```

The runner reconstructed all `1,379` Gate 3C1D model-validation decisions before
CoTracker initialization. Selected shortlist slot, action, output candidate,
causal-record digest, and frozen policy match the sealed parent replay exactly.
No teacher or future tensor influenced action identity.

## Frozen selector

```text
support probability >= 0.30
predicted value      >= 1.0 px
predicted harm       <= 0.20
```

The policy acts on `387/1,379` rows (`28.0638%`). Commit-level results reproduce
Gate 3C1D exactly:

```text
native mean commit error: 34.3669 px
policy mean commit error: 30.8805 px
mean reduction:           +3.4864 px
action precision <=12px:  62.2739%
```

## Future trajectory result, frames 16--23

| Action | Mean future error | Severe >16 px | Threshold utility |
|---|---:|---:|---:|
| native | 37.2469 px | 98.3579% | 0.003345 |
| coordinate only | 37.1768 px | 97.8866% | 0.004287 |
| coordinate + four-level memory | **33.5796 px** | **79.3654%** | **0.061299** |

### Coordinate plus memory versus native, all failure rows

```text
mean future-error reduction:       +3.6673 px
video-cluster 95% CI:              [+3.2492,+4.6715] px
threshold-utility gain:             +0.05795
utility 95% CI:                     [+0.05515,+0.08475]
severe-16px rate reduction:         +18.9925 percentage points
positive point fraction:            55.8376%
nonnegative point fraction:         71.9362%
harmful future fraction (>+4 px):   0.8702%
nonnegative source-video fraction:  96.0%
```

### Frozen action rows only

```text
rows:                               387
mean future-error reduction:        +13.0150 px
video-cluster 95% CI:               [+11.4466,+14.1075] px
threshold-utility gain:             +0.20586
positive point fraction:            94.0568%
harmful future fraction (>+4 px):   3.1008%
```

## Memory contribution beyond coordinate-only

Coordinate-only changes future mean error by only `+0.0700 px`. Deterministic
four-level memory reinstatement contributes the dominant improvement:

```text
all-row incremental error reduction:     +3.5973 px
video-cluster 95% CI:                    [+3.1859,+4.5559] px
incremental threshold-utility gain:       +0.05701
incremental severe-16px reduction:        +18.5212 percentage points
memory-better source-video fraction:      96.0%
```

On action rows, memory contributes `+12.7589 px` beyond coordinate-only, with
video-cluster 95% CI `[+10.9739,+13.6579]` px. This establishes that the future
benefit is not a coordinate-overwrite artifact.

## Replay integrity

```text
native replay maximum absolute drift: 0.0 px
selector consistency:                 exact
primary/replay scientific payload:    exact
future coordinate digests:            exact
point/video record digests:           exact
```

Artifacts:

```text
primary file SHA256:
5c68cc02e7097ab1bd21078186caceb79dde1b5f077f9189535f491c33a86555
primary payload SHA256:
c9d6655b9db1201fa0e60b366decce18b71cbfb3bf6c2227342929d7c51f953a

replay file SHA256:
97a4c4972ff2db03ad18b8a2c0e6a2bd477bd1c1e03b89bba19eb87e4d58c406
replay payload SHA256:
9518f96c1cc19b37b02348d9d559c07898ce76c8dc1d3fadf6f2f1e3729ad173

scientific payload SHA256:
6147063d80ffc2a92cfb802b8a4415afbdd3cd229c9626ae201a00c73c3195dc
```

## Claim boundary

Allowed:

```text
On the renewed raw-record-disjoint Kubric natural-failure population, the frozen
causal selector plus deterministic four-level memory writeback reproducibly
improves visible future trajectories. Memory reinstatement is the dominant
mechanism.
```

Not yet allowed:

```text
The method improves CoTracker3 paper AJ, delta-average, or OA on complete
TAP-Vid videos.
```

The current natural-failure population was defined using future ground truth for
scientific isolation. A complete-video paper-table evaluation must not reuse that
membership oracle. Gate 3C1F must therefore preregister a fully causal
full-population entry contract and an official evaluator before opening DAVIS or
Kinetics for this route.

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
unread by Gate 3C1E.
