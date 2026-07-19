# Route-D strong-backbone final-holdout cache smoke result — 2026-07-19

## 1. Decision

The P0l sealed-cache implementation passes the preregistered fit-source-0
compatibility gate:

```text
ALLOW_P0L_FINAL_HOLDOUT_CACHE_BUILD
```

No final-holdout identity was read. No performance metric was computed.

## 2. Scope

```text
partition: implementation_smoke
source index: 0
video: 1680
use: schema, parity, quantization and deterministic compatibility only
```

The new cache format stores only the tensors required by finalized-state
output-only variant C:

- frozen CoTracker native coordinates, visibility and confidence;
- first-visible queries and sealed GT tensors;
- candidate 0 equal to native;
- frozen float16 CoTracker feature maps.

It does not generate P0j-C candidates or calculate AJ during cache creation.

## 3. Exact reference parity

The sealed base artifact exactly matches the existing formal fit cache for:

```text
native coordinates
native visibility probability
native confidence probability
native joint probability
native visibility decision
query points
GT tracks and occlusion
candidate-0/native parity
```

The float16 feature tensor is also byte-identical to the existing formal CMCP
feature cache.

## 4. Variant-C output parity

Using the frozen variant-C checkpoint on the sealed artifact reproduces the
existing formal fit artifact exactly for:

```text
candidate coordinates
candidate validity masks
selected candidate indices
selected coordinates
dynamic decision summaries
```

This verifies that the new base schema, feature schema and evaluator preserve the
P0j-C numerical contract without using performance to select an implementation.

## 5. Quantization

```text
max absolute error: 0.0001220703125
minimum cosine:     0.999999463558197
```

Both remain inside the frozen P0l gates.

## 6. Evidence

```text
docs/generated/ROUTED_STRONG_BACKBONE_FINAL_HOLDOUT_SMOKE_SUMMARY_2026-07-19.json
SHA-256: 3ac292a50fd28b487130a1c613ea01312a3869909bd18a1b17bc336f21c04abf
```

## 7. Claim boundary

This is implementation integrity only. It authorizes building the complete
16-video sealed final-holdout cache after the implementation commit is frozen.
It does not authorize a performance claim, DAVIS access or a Kinetics rerun.
