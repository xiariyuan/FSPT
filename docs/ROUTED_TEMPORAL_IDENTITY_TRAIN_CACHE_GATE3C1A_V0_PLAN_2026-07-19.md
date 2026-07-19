# Route-D temporal identity train-cache Gate 3C1A v0 — 2026-07-19

## Status

```text
COMPLETED_PASS
```

## Why this gate precedes selector preregistration

The single-row probe fixed a feasible causal feature contract, but it did not
measure how many natural failures exist across the new gradient-train
population or how often the frozen top-128 bank contains a 12-pixel positive.
Those quantities determine whether a learned listwise selector has enough
training support.

Gate 3C1A therefore builds only the gradient-train cache on indices 64--383.
It does not read checkpoint-selection indices 384--447 or fit-only audit
indices 448--511. Selector architecture and loss remain unfrozen until this
training-only cache is complete.

## Frozen construction

For every one of the 320 gradient-train videos:

1. run frozen CoTracker3 through frames 0--23;
2. select at most 16 natural failures using the frozen Gate 2 rule;
3. use only frames 0--15 and native state to form mandatory native plus 128 M1
   candidates;
4. freeze and hash candidate coordinates;
5. run each candidate list backward through frames 15--0 with frozen
   CoTracker3, one failure row per microbatch;
6. encode frames 0--15 once per video with frozen DINOv3;
7. store only the 9 temporal channels, 14 static channels, candidate metadata,
   and masks;
8. after feature freezing, read frame-15 teacher coordinates to create
   candidate-distance and within-12-pixel labels.

Raw 384-dimensional DINO descriptors are not persisted.

Videos with zero eligible failures remain required cache members and receive
valid empty tensors. They may not be silently dropped.

## Completion

Passing requires exactly 320 validated sidecars in source order, exact
64--383 membership, finite feature tensors, exact per-row shapes, verified
sidecar hashes, candidate hashes recorded before teacher labels, and every
locked-data flag false.

Training-set row counts and top-128 support statistics are descriptive and may
be used to set a later model capacity and minimum-evidence rule. They are not
selector performance and cannot be reported as an improvement.

```text
pass -> AUTHORIZE_GATE3C1B_SELECTOR_PREREGISTRATION
fail -> STOP_AND_REPAIR_GRADIENT_TRAIN_CACHE
```

## Final result

The preregistered build completed on exactly source indices 64--383. All 320
sidecars and their tensor hashes validated, including 12 videos with valid
empty failure tensors. The cache contains 3,325 frozen natural-failure rows.

Training-only descriptive support is 3,185/3,325 (95.7895%) for the complete
native-plus-128 bank, versus 1,725/3,325 (51.8797%) for native plus the first
eight non-native M1 candidates. Native alone covers 54/3,325 (1.6241%).

The formal completion decision is:

```text
AUTHORIZE_GATE3C1B_SELECTOR_PREREGISTRATION
```

This is candidate-set headroom on gradient-training data. It is not learned
ranking, a deployable restoration gain, model validation, or external
performance improvement. Exact hashes and locked-data flags are recorded in
`docs/generated/ROUTED_TEMPORAL_IDENTITY_TRAIN_CACHE_GATE3C1A_V0_2026-07-19.json`.
