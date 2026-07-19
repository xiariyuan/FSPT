# Route-D strong-backbone DAVIS external transfer smoke result — 2026-07-19

## 1. Decision

The P0m external wrapper and frozen model path pass the preregistered Kubric
fit-index-0 smoke:

```text
ALLOW_P0M_COMPLETE_DAVIS_CACHE_BUILD
```

No DAVIS sample was loaded and no DAVIS performance was computed.

## 2. Scope

The smoke compares two independent loaders of the same frozen variant-C
checkpoint on the existing formal Kubric fit-0 feature cache:

- the established P0l final-holdout model loader;
- the new P0m external-transfer model loader.

Both use the same finalized-state output-only prediction function and point batch
size.

## 3. Exact checks

The following tensors are exact:

```text
selected coordinates
selected candidate indices
candidate coordinates
candidate validity masks
oracle coordinates
oracle candidate indices
```

Comparator behavior summaries, normalization statistics and the combined
model-state SHA are also exact.

This proves that P0m does not alter the frozen variant-C graph while adapting the
cache and dataset interface for DAVIS.

## 4. Loader-mutation guard

The pinned DAVIS loader scales its underlying point array in place during
`__getitem__`. P0m wraps every access by copying the raw normalized points and
restoring them in a `finally` block. Unit tests verify exact restoration, so the
fixed extraction replays at indices 0, 14 and 29 cannot be corrupted by repeated
point scaling.

## 5. Evidence

```text
docs/generated/ROUTED_STRONG_BACKBONE_DAVIS_EXTERNAL_SMOKE_SUMMARY_2026-07-19.json
SHA-256: e29a99b0cb8a4446f7ce1b45d09d3435cec6a940af47ed6f3eedefb117366aeb
```

## 6. Claim boundary

This is implementation integrity only. It authorizes construction of the full
30-video sealed DAVIS cache after the implementation commit is frozen. It does
not authorize a DAVIS performance claim or any Kinetics rerun.
