# Route-D temporal identity data Gate 3C0 v0 result — 2026-07-19

## Status

```text
COMPLETED_PASS
```

## Formal decision

```text
AUTHORIZE_GATE3C1_CAUSAL_FEATURE_CACHE_ON_EXPANDED_KUBRIC
```

This decision authorizes only the causal feature-shape probe and feature cache
on the expanded Kubric development population. It does not authorize a
performance claim, model-validation access, external evaluation, or a Kinetics
rerun.

## Qualified population

| Quantity | Result |
|---|---:|
| Samples | 512 |
| Output shards | 32 |
| Hashed source TFRecords | 54 |
| Existing-protocol overlap verification | indices 0--63 |
| Gradient train | indices 64--383, 320 videos |
| Checkpoint selection | indices 384--447, 64 videos |
| Fit-only internal audit | indices 448--511, 64 videos |

## Mandatory checks

Every preregistered check passed:

- expanded sample count is exactly 512;
- expanded shard count is exactly 32;
- all 54 independently recomputed source-file hashes match the manifest;
- the first 64 raw identities exactly match the frozen existing protocol;
- the first 64 sample tensors and metadata digests exactly reproduce the
  frozen existing protocol in order;
- expanded indices 64--511 have no raw-record overlap with the existing 64;
- all 512 raw identities are unique;
- the three development partitions cover exactly indices 64--511 and are
  disjoint;
- all locked-data flags remained false.

## Authority and hashes

- Config SHA-256:
  `15143e297ad93a2a392ef6dabca5324b09cf07545d9bbfd607d5b06e270bde0f`
- Expanded manifest SHA-256:
  `afd3af1b2af07f8c3483907c94539dc40a7dff5bb290a587eb7db5083953f961`
- Combined expanded sample digest:
  `65e2d1610e572e7f7ca798a62623e11108172151894b944c4553a67cb70ac786`
- Combined existing-64 sample digest:
  `895491e20ca1cbf7cd3624e374ced14921f72ff1b6ef6b768806f1e0a4fb5315`
- Summary payload SHA-256:
  `94cdbc46cf7284659f8cf80b6d60a3783e52465e01288e4bb34d7e4111cdd20b`

Partition raw-identity digests:

- gradient train: `dd558713cea1e83695169e2942f8883860944255d6009de10118b2a33780cd17`
- checkpoint selection: `710a4bb0dfd54277ab770e3e86d840daf83febbd27e72d64f04811e184be938f`
- fit-only internal audit: `f317f920bd161cf00e7c6a541d952706e42345dbf5e57ac86e9a3b46724880a7`

Machine-readable authority:

`docs/generated/ROUTED_TEMPORAL_IDENTITY_DATA_GATE3C0_V0_SUMMARY_2026-07-19.json`

## Claim boundary

Gate 3C0 is a data-identity and split-integrity result, not a tracking
improvement. The currently observed M1 top-128 oracle support remains a
fit-only diagnostic. A real method gain still requires a causal temporal
identity selector to pass its separately preregistered top-eight support gate
and then produce positive future rollout utility under frozen state commit and
risk-control rules.

## Next authorized action

Run a single-sample feature-shape and memory probe on gradient-train index 64.
The probe may read frames 0--15 and frozen CoTracker3/DINOv3 observations. It
must not read checkpoint-selection indices, fit-only audit indices, original
model-validation indices 48--63, future model features, DAVIS, or Kinetics.

Only after the probe fixes feature shapes and feasible candidate batching may
Gate 3C1 architecture, loss, checkpoint selection, and audit thresholds be
preregistered.
