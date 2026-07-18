# BridgeTrack Persistent Query-State Feasibility Result — 2026-07-18

## Decision

```text
PERSISTENCE_ONLY_CLOSED
SAME_TIME_QUERY_STATE_RECONSTRUCTION_HAS_CAUSAL ORACLE SIGNAL
MID-LAYER JOINT STATE IS THE ONLY CURRENT COMPACT TARGET
NO TRAINING
NO LOCKED DATA
```

The frozen TAPNext++ backbone and PointOdyssey fit scenes were used only for
oracle feasibility. PointOdyssey internal holdout/test, method-version DAVIS,
and Kinetics 1,144 were not read.

## 1. Static Gate A

Protocol: one real fit image repeated for 8 visible, 32 corrupted, and 16 visible
frames.

| Branch | Future mean error @256 |
|---|---:|
| Native student | 1.0649 px |
| Pre-gap query state | 0.3707 px |
| Same-time teacher query state | 0.3691 px |
| Full teacher state | 0.2907 px |

Teacher query-state improvement was `+0.6958 px`, and it retained `89.87%` of
the full-state improvement. The frozen `>=2 px` gate failed because even the
full-state oracle ceiling was only `+0.7742 px`.

Interpretation:

```text
STATIC_GATE_CEILING_TOO_SMALL
```

The threshold was not relaxed. A moving fit-only redesign was preregistered.

## 2. Dynamic Gate A2

Protocol: a real 56-frame moving fit clip, with 8 original frames, 32 fixed
mean-RGB corruption frames, and 16 original rollout frames. The point was
selected by the frozen 90th-percentile gap-motion rule from 232 points that
remained valid and interior throughout.

Selected gap motion: `25.2255 px` at original raster.

| Branch | Future mean error @256 | Delta vs native |
|---|---:|---:|
| Native student | 4.3145 px | — |
| Pre-gap query state | 9.9435 px | **−5.6290 px** |
| Same-time teacher query state | 0.8751 px | **+3.4394 px** |
| Full teacher state | 0.7917 px | **+3.5227 px** |

Same-time teacher query state:

- improved all `16/16` future frames;
- retained `97.63%` of full-state oracle improvement;
- reduced future mean error from `4.3145` to `0.8751 px`.

However, stale pre-gap state was strongly harmful. Therefore direct persistence
or snapshot restoration is falsified.

Interpretation:

```text
PERSISTENCE_ONLY_FAIL
SAME_TIME_STATE_RECONSTRUCTION_SIGNAL
```

The full frozen gate did not pass, so training remains forbidden.

## 3. Query-state factorization

The same dynamic fit-only protocol was used to replace selected recurrent cache
components/layers with the same-time teacher query state.

| Variant | State replaced | Mean error | Retained full-query gain |
|---|---:|---:|---:|
| Layers 4–7, RG + Conv | 33.33% | **0.6134 px** | **107.61%** |
| Layers 6–11, RG + Conv | 50.00% | 0.8209 px | 101.58% |
| Layers 0–5, RG + Conv | 41.66% | 0.8895 px | 99.58% |
| Layers 6–11, Conv only | 37.50% | 1.0693 px | 94.35% |
| All Conv only | 68.75% | 1.9572 px | 68.54% |
| All RG only | 22.91% | 2.6758 px | 47.64% |
| Layers 8–11, RG + Conv | 33.33% | 4.9372 px | −18.11% |

The best compact branch, layers 4–7 with both cache components, improved all
16 future frames and outperformed replacement of the entire query state. This
indicates causal layer synergy rather than a monotonic “replace more state”
effect.

The preregistered single-component `>=70%` gate narrowly failed because Conv
alone retained `68.54%`. It is not valid to round this up or remove the gate.

Interpretation:

```text
MID_LAYER_JOINT_STATE_HYPOTHESIS
SINGLE_COMPONENT_SIMPLIFICATION_FAIL
```

## 4. Model implication

The next model must not:

- restore a stale pre-occlusion state;
- predict coordinates directly;
- reconstruct all 12 recurrent layers;
- modify image-token state;
- use a Conv-only shortcut based only on state-distance magnitude.

The only currently justified target is a bounded reconstruction of the joint
RG-LRU and Conv1D query state in middle recurrent layers, conditioned on the
current reappearance observation and causal damaged state.

## 5. Next allowed step

Before any training, preregister a multi-fit-scene verification of the layers
4–7 joint-state hypothesis. It must test sequence consistency, include lower
motion points, and retain the same no-locked-data boundary.

Only a successful multi-scene fit-only gate may authorize a small learned
reconstructor. No validation, holdout, test, DAVIS method evaluation, or
Kinetics execution is allowed at this stage.
