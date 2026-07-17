# Route-D MUSR Kubric cache protocol v0 — 2026-07-17

## Decision being preregistered

The one-video CoTracker3 interface audit established causal candidate headroom,
but it used `validation_source index 0` during method design. That sample is now
permanently quarantined. It cannot enter training, calibration, model selection,
qualification, final holdout reporting, or a learned-result claim.

The next question is deliberately narrower than training:

> Does the frozen native-plus-local-correlation candidate generator retain
> material coordinate-oracle headroom across a complete, identity-disjoint
> multi-video Kubric qualification partition?

MUSR training remains forbidden until this gate is complete and passed.

## Frozen sources and identities

Three manifests from the already frozen 2026-07-14 Kubric protocol are pinned by
SHA-256:

| Alias | Samples | SHA-256 |
|---|---:|---|
| train source | 64 | `bedc1678fccfeb89ea081bf006dc28b4fb37deae1f226d39cb37a68195d0eb14` |
| calibration source | 32 | `958eed38daabc040932d1496c4bfa61a4c9f3d2be5641e386350725bb9cad4d7` |
| validation source | 32 | `ae31d7a2d6c9c31c487afcb0d15e89f1cc4ee51323f8673abdeabacfb307333e` |

Identity is the exact tuple:

```text
(source_tfrecord, source_record_index, video_name)
```

All 128 source samples are unique and the three source manifests have zero
identity overlap.

## Frozen partitions

| Partition | Membership | Count | Permitted use |
|---|---|---:|---|
| fit | train `0–47` | 48 | gradient updates only |
| model validation | train `48–63` | 16 | architecture selection and early stopping |
| calibration | calibration `0–31` | 32 | post-training calibration only |
| pilot excluded | validation `0` | 1 | historical interface pilot only |
| candidate qualification | validation `1–15` | 15 | pre-training candidate oracle gate |
| final holdout | validation `16–31` | 16 | locked until the complete learned system is frozen |

The final holdout is rejected by the cache builder before model freeze.

## Frozen candidate interface

```text
backbone: CoTracker3 scaled online, true streaming, frozen
checkpoint SHA-256: 205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218
query mode: first visible
raster: 256 × 256
candidate 0: exact native CoTracker3 continuation
candidates 1–5: query-memory local-correlation top-K
search radius: 64 px
candidate feature dimension: 64
state feature dimension: 32
state source: real online_track_feat level 0
```

Ground truth is unavailable to candidate generation. It is used only after the
candidate cache is frozen to compute oracle labels and audit metrics.

## Qualification gates

All checks must pass on the complete 15-video candidate-qualification partition:

```text
routing-disabled native parity: exact on every video
pooled coordinate-oracle AJ gain: at least +3.0 points
pooled delta-average gain: at least +4.0 points
median per-video AJ gain: at least +2.0 points
fraction of videos with at least +1.0 AJ: at least 2/3
each pooled 1/2/4/8/16-px hit-rate gain: strictly positive
complete partition: required
```

A partial run is progress only. It cannot authorize training regardless of its
observed metrics.

Adapter-export replay is repeated at validation source indices `1`, `8`, and
`15`. The candidate coordinates, 64-dimensional candidate features, candidate
scores, validity masks, source IDs, 32-dimensional state features, and native
state tensors must be bit-identical between the two exports from the same frozen
native backbone state.

## Artifact layout

Each video is written independently so interrupted GPU runs are resumable:

```text
outputs/routeD_musr_kubric_cache_20260717/<partition>/video_<index>.pt
outputs/routeD_musr_kubric_cache_20260717/<partition>/video_<index>.json
outputs/routeD_musr_kubric_cache_20260717/<partition>/cache_index.json
```

The per-video sidecar stores candidate tensors, state tensors, GT labels,
provenance, hashes, and the per-video oracle audit. The index stores exact
membership, completion state, pooled official TAP-Vid metrics, gate checks, and
whether training is authorized.

## External-data lock

The corrected 1,144-sample TAP-Vid-Kinetics result is not read, rerun, or used
for candidate/model/policy selection. TAP-Vid-DAVIS is also locked. No external
result is opened during this phase.

## Commands

Protocol and CPU-only integrity check:

```bash
python -m pytest -q tests/test_routeD_cotracker3_kubric_cache.py
```

Resumable full candidate qualification:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python scripts/build_routeD_cotracker3_kubric_cache.py \
  --protocol configs/routeD_musr_kubric_cache_protocol_v0.json \
  --partition candidate_qualification \
  --resume \
  --device cuda
```

A bounded progress run may add `--max-videos N`, but the generated index must
remain `complete: false` and `training_authorized: false`.
