# Route-D CMCP frozen feature-map cache result — 2026-07-17

The frozen CoTracker3 feature-map cache is complete for the authorized CMCP
development partitions:

| Partition | Videos | Complete |
|---|---:|---:|
| fit | 48 | yes |
| model-validation | 16 | yes |

Each sidecar stores one `float16` tensor of shape `24×128×96×128`. Native
coordinate, visibility, confidence, joint probability, and visibility tensors
were regenerated and checked against the already qualified stage-0 sidecars.
All match exactly.

Quantization audit:

```text
global worst max_abs:    0.000235856
global worst min cosine: 0.999999404
gates: max_abs <= 0.0005; min cosine >= 0.99999
```

On fit video 0, reconstructing the three query/previous/EMA correlation fields
from the float16 cache changes correlation values by at most `0.00012365`
(mean `0.00001634`). The motion prior and causal validity mask are exact.

Float32 feature extraction is bit-identical at frozen anchors 0, 47, 48, and 63.
All 64 cache sidecars pass schema, base-sidecar, protocol, and tensor-hash
verification.

Decision:

```text
ALLOW_CMCP_FIT_ONLY_DENSE_PROPOSAL_TRAINING
```

This is cache integrity evidence, not learned proposal performance. Calibration,
final_holdout, DAVIS, Kinetics, MUSR selection, and state writeback remain locked.
