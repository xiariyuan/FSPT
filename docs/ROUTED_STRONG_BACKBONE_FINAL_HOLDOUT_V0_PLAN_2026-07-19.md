# Route-D strong-backbone final synthetic holdout v0 plan — 2026-07-19

## 1. Decision being preregistered

P0j freezes the recommended strong-backbone model as variant C:

```text
frozen formal P0g CMCP
+ frozen rank-32 LMRA checkpoint
+ frozen local safety comparator
+ frozen CoTracker3 finalized native trajectory
```

P0k closes coordinate-only state writeback before model-validation access. The
complete learned system is therefore now frozen as **finalized-state output-only
variant C**. P0l performs one identity-disjoint Kubric final-holdout evaluation.

This is the last synthetic decision gate before any new external protocol can be
preregistered.

## 2. Why calibration is not opened

Variant C contains no post-training threshold or operating point:

- abstention is part of the frozen comparator graph;
- candidate extraction, NMS, EMA and top-K are fixed;
- checkpoint selection ended at P0j;
- state writeback was rejected and removed from the final system.

The calibration partition therefore has no legitimate remaining decision to
make. P0l explicitly uses **no calibration**. This prevents an unnecessary
post-selection adaptation opportunity and preserves a cleaner final-holdout
claim.

## 3. Frozen identities

Implementation smoke:

```text
train source index 0
schema/parity/replay only
no performance decision
```

Final holdout:

```text
validation source indices 16--31
16 videos
```

The following validation-source identities remain excluded:

```text
index 0: historical interface pilot
indices 1--15: candidate qualification
```

No identity overlap is permitted among pilot, qualification and final holdout.
The validation manifest is pinned at:

```text
ae31d7a2d6c9c31c487afcb0d15e89f1cc4ee51323f8673abdeabacfb307333e
```

## 4. Cache-before-metrics rule

P0l first builds immutable native-state and feature-map artifacts for all 16
holdout videos. The cache builder may report only:

```text
membership and completion
native parity
feature quantization
hashes and deterministic replay
runtime
```

It may not compute, print or save selected-model performance while the partition
is incomplete. Ground truth may be stored in the sealed artifact for the later
frozen evaluator, but it cannot affect native state, feature extraction,
proposal generation or model actions.

The evaluator is unavailable until the cache index declares exactly 16/16
members complete.

## 5. Frozen model and execution

```text
variant-C checkpoint SHA-256:
7babb76e3407497832d0bc0fca4557df64b2d09450ebc60766e32a70816ab52f

combined model-state SHA-256:
64c3f6dae6ae34dc0223754aca3c76f78a1f48137d08bab6c3e01f2a806cd074

CoTracker3 checkpoint SHA-256:
205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218
```

Execution uses:

```text
true streaming
first-visible queries
256x256 input and metric raster
all points jointly
no support grid
finalized native overlap state
output-only correction
no state writeback
```

## 6. Determinism and quantization

The full extraction path is independently replayed at final-holdout source
indices `16`, `23`, and `31`. Native tensors and float32 feature maps must be
bit-identical.

Frozen float16 feature gates remain:

```text
max absolute reconstruction error <= 5e-4
minimum cosine similarity >= 0.99999
```

Formal variant-C evaluation runs twice from the same sealed 16-video cache. Model
state, candidate-coordinate hashes, complete metrics, per-video rows and gates
must be exact.

## 7. Final gates

All gates are required:

```text
complete 16-video partition
candidate-0/native parity exact
candidate-oracle AJ gain >= +3.0 points
direct AJ gain >= +0.5 point
paired-video direct AJ CI lower bound > 0
delta-average gain > 0
16px severe-error rate not worse
harmful non-native rate <= 1%
positive videos >= 12 / 16
exact primary/replay
```

The direct magnitude and safety gates are the same as the prior strong-backbone
model-selection contract. The positive-video gate is added before holdout access
to prevent a pooled gain dominated by a small subset.

## 8. Decision

Pass:

```text
AUTHORIZE_FROZEN_EXTERNAL_PROTOCOL_PREREGISTRATION
```

This authorizes writing an external protocol; it does not itself open DAVIS or
permit a Kinetics rerun.

Fail:

```text
STOP_STRONG_BACKBONE_EXTERNAL_ROUTE_AND_RETAIN_INTERNAL_EVIDENCE_ONLY
```

A failure may be reported as final synthetic evidence, but cannot trigger a
checkpoint, threshold, candidate, calibration or policy rescue.

## 9. External lock

Until P0l formally passes:

```text
TAP-Vid-DAVIS: unread
official TAP-Vid-Kinetics 1,144: unread and not rerun
```

## 10. Completed implementation smoke — 2026-07-19

The fit-source-0 sealed-cache smoke passes exact parity for all native/query/GT
tensors, the float16 feature map, and all frozen variant-C candidate/selection
outputs. No performance metric or final-holdout sample was read.

```text
ALLOW_P0L_FINAL_HOLDOUT_CACHE_BUILD
```

Canonical evidence:
`docs/ROUTED_STRONG_BACKBONE_FINAL_HOLDOUT_SMOKE_RESULT_2026-07-19.md`.

## 11. Completed final result — 2026-07-19

The sealed 16-video final holdout and independent replay are complete. Every
preregistered gate passes:

```text
AJ gain: +0.7679
paired-video AJ 95% CI: [+0.4532,+0.9184]
delta-average gain: +0.8889
harmful non-native rate: 0.9462%
positive videos: 14 / 16
```

Formal decision:

```text
AUTHORIZE_FROZEN_EXTERNAL_PROTOCOL_PREREGISTRATION
```

See `docs/ROUTED_STRONG_BACKBONE_FINAL_HOLDOUT_RESULT_2026-07-19.md`.
External datasets remain unread until a separate protocol is committed.
