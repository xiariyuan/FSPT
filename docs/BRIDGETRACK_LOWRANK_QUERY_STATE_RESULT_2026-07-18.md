# BridgeTrack Low-Rank Query-State Result — 2026-07-18

## Status

The frozen low-rank repair-subspace oracle is complete on ten previously
unexposed PointOdyssey fit scenes.  This is an oracle architecture audit, not a
learned or deployable result.

## Basis

```text
basis scenes:                    7 already exposed fit scenes
basis clips:                     21
query-state dimension:           36,864
primary PCA rank:                8
rank-8 normalized energy:        69.42%
verification scenes:             10 previously unused fit scenes
```

## Result

Rank-8 projection:

```text
positive scenes:                 1 / 10
median future-error gain:       -6.7984 px
mean scene-bootstrap 95% CI:   [-26.8860, -4.4897] px
median improved-frame fraction:  12.5%
worst scene regression:         -58.3170 px
```

All frozen gates failed except completion and the diagnostic full-query count.

The complete rank curve was also negative:

```text
rank 0 median gain:  -4.3234 px
rank 1 median gain:  -5.2647 px
rank 2 median gain:  -6.4906 px
rank 4 median gain:  -6.3122 px
rank 8 median gain:  -6.7984 px
```

Therefore this is not a rank-selection problem.

## Full-query oracle boundary

Same-time full teacher query-state replacement remained useful in 8/10 scenes,
but it was harmful in two scenes, including a `-15.6716 px` regression on
`animal_rabbit`.  Full recurrent-state replacement was also slightly harmful in
some already-correct cases.

This demonstrates that query state is not an independently transferable point
identity variable.  Its meaning is coupled nonlinearly to the current image-token
state and scene-specific recurrent trajectory.  L2/PCA closeness in state space
does not preserve causal output behavior.

## Decision

```text
CLOSE_BRIDGETRACK_QUERY_STATE_REPAIR
NO COEFFICIENT PREDICTOR TRAINING
NO MODEL VALIDATION
NO LOCKED DATA
```

The following are closed:

- stale-state restoration;
- fixed-layer state replacement;
- single-component state repair;
- low-rank cross-scene state-delta projection;
- frozen TAPNext++ query-state repair as the paper mainline.

Any continuation must be a genuinely model-level architecture that learns
persistent identity and transient tracking state jointly during training.  It
cannot be another post-hoc state injection or frozen-tracker adapter.
