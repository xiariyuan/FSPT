# Route-D counterfactual structured state restorer interface result — 2026-07-19

## Formal decision

```text
ALLOW_GATE2_TEACHER_CACHE_BUILD
```

The frozen Gate 2 CSRR interface passes on source index 8 from the gradient-train
partition. Primary and independent replay are exact after excluding output-path
fields, including every nested sidecar tensor.

This result authorizes only construction of the sealed teacher caches for source
indices `8–31` and `32–47`. It is not a learned-model, model-validation,
holdout, DAVIS, Kinetics, or paper-level result.

## Frozen identities

```text
Gate 2 config SHA256:
31a62db62d609e28acb9a7ac8cc5740866134d92ee3564e99a0bd7f847a33f6b

CoTracker3 checkpoint SHA256:
205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218

Kubric fit manifest SHA256:
bedc1678fccfeb89ea081bf006dc28b4fb37deae1f226d39cb37a68195d0eb14

interface source index: 8
video name: 7137
selected point indices: [0,2,5,6,7,8,10,16]
input raster: 256 x 256
CoTracker model raster: 384 x 512
CSRR trainable parameters: 19,685
```

## Schema-corrected rerun

Before cache construction, the enumerated trajectory representation was found to
contain nine scalar channels while the module default was six and the config did
not record an explicit dimension. No cache, checkpoint, training result, or
validation metric existed. Commit `1ed05c6` corrected the module and explicitly
froze `model.input.trajectory_dim: 9`.

The primary/replay results and hashes in this document are from the corrected
configuration. The earlier interface summary is superseded and must not be used.

## Structured re-extraction parity

The complete observed clip `0–15` is encoded using the same frozen CoTracker fnet
batch path as official online inference. At the exact frame-15 GT coordinate,
CSRR's re-extraction helper reconstructs all four fresh-query memory levels.

Across the four levels:

```text
track-feature maximum absolute error: <= 2.98e-8
support-memory maximum absolute error: <= 2.98e-8
minimum cosine similarity:             >= 0.99999994
```

These values are far inside the preregistered cache interface gate:

```text
maximum absolute error <= 1e-3
cosine similarity >= 0.99999
```

The remaining `1e-8` differences are floating-point operation-order effects and
not a coordinate-system or feature-sampling mismatch.

## Float16 cache feasibility

The largest observed float16 round-trip errors are:

```text
commit feature pyramid:       1.2204e-4
teacher track feature:        6.0245e-5
teacher support memory:       1.1718e-4
minimum observed cosine:      0.99999988
```

Therefore the frozen float16 sidecar representation passes both quantization
gates with substantial margin.

## Safety and causality checks

All required checks pass:

```text
source belongs to gradient-train partition: true
24-frame / 64-point identities exact:       true
parameter ceiling:                          true
teacher re-extraction gate:                 true
float16 quantization gate:                   true
false apply gives exact native state:        true
model reads future frames:                  false
model reads GT at inference:                false
model-validation read:                      false
calibration read:                           false
final holdout read:                         false
DAVIS read:                                 false
Kinetics read or rerun:                     false
```

The no-op branch clones state without aliasing while preserving every tensor
bit-for-bit.

## Reproducibility

```text
primary report SHA256:
217bf83b5eee296f25dce65bc81a206dcde500eec0ccce3c9215355a1af56b70

replay report SHA256:
652c1119d02587be617ac1366543e2a832de0e1f97d9fdf47997db172cd75799

primary sidecar SHA256:
49617a75838ac3103aec1b6a2d8c43be0016466ea5bf61611e69c494725241b5

replay sidecar SHA256:
59a05856fc7fedafe1d6efa1d0a6b95b6e8fe2e5a4172e9ff8b7940671878ab1

canonical summary SHA256:
84287c8ad739df418d8a2158ff36a0e2e69a4c87b9545b062c2e6eda4dfa9bfb
```

## Authorized next step

Build complete sealed caches in this order:

```text
train cache:               source indices 8–31
fit-internal validation:   source indices 32–47
fixed replay anchors:      8, 31, 32, 47
```

Each row must include frozen frame-15 feature pyramid, native commit state,
failure fresh-query teacher state, clean no-op rows, future GT scoring metadata,
source hashes, tensor hashes, and quantization audit. Training may begin only
after both cache indexes are complete and all fixed replay anchors match.

Source indices `48–63`, calibration, final holdout, DAVIS, and Kinetics remain
locked.
