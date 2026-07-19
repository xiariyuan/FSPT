# Route-D geometry representation audit Gate 3A v1 — 2026-07-19

## Status

```text
PREREGISTERED_NOT_RUN
```

## Frozen question

Gate 2.5 established that deterministic full-state reinstatement has a real,
contiguous recovery basin through 12 input pixels. Gate 3A v1 asks whether a
frozen causal correlation representation places at least one discrete proposal
inside that basin after a long-occlusion failure.

This is a representation-support oracle audit. It does not train a selector and
does not estimate deployable performance.

## Why v0 is not run as the route decision

Gate 3A v0 was written before the reinstatement basin was measured and tested
only the Gate 2 pooled-native map at an unjustified 8-pixel primary radius.
Gate 2.5 makes 12 pixels the empirically supported primary radius and motivates
two representation controls. V0 remains frozen and unrun; its map is M0 below.

## Frozen fit-only population

- source indices: 32--47;
- 16 already fit-exposed videos and 80 frozen Gate 2 failure points;
- failure membership was defined by the earlier Gate 2 diagnostic labels;
- no new point or video may be added after seeing Gate 3A output;
- source indices 48--63, calibration, final holdout, DAVIS, and Kinetics stay
  unread.

Conditioning on known failures makes this an oracle headroom diagnosis, not a
causal failure detector evaluation.

## Teacher/future isolation boundary

The candidate process is not allowed to deserialize the teacher-bearing Gate 2
artifact or the raw Kubric sample.

A separate allowlist materializer writes only:

1. video frames 0--15;
2. the legitimate original tracking queries;
3. frozen failure point identities;
4. native commit coordinates and Gate 2 causal model inputs;
5. frame-15 feature pyramid and native 49-token support.

Its tensor namespace rejects fields containing `teacher`, `future`, `gt_`,
`continuation`, `target_points`, or `occluded`. The downstream candidate
builder reads only these sanitized sidecars. Frame-15 teacher coordinates and
future outcomes are first read by the audit process, after a complete primary
candidate cache, an independent fresh replay, and exact sidecar qualification.

## Frozen representations

All representations use four CoTracker3 levels, a 64 x 64 fused map, and no new
trainable parameters.

```text
M0 pooled-native Gate 2:
  frozen epoch-6 Gate 2 fused logits; control for Gate 3A v0.

M1 geometry-native:
  cosine-match every current native 7x7 support token at its corresponding
  spatial offset; symmetrically trim 6/49 token scores per tail; equal-average
  four levels.

M2 geometry-immutable-query:
  the identical geometry matcher using 7x7 supports re-extracted at each
  point's original query event. Feature sequences are physically truncated
  after the latest query frame before anchor extraction.
```

The representations are judged independently. If more than one passes, the
formal selection order is M0, then M1, then M2; this prefers the smallest method
change and prevents post-result cherry-picking.

## Shared candidate extractor

For every representation and point:

1. candidate 0 is the frozen native coordinate;
2. extract eight nonnative peaks;
3. square Chebyshev NMS radius is 3 grid cells;
4. refine each peak with a 5 x 5 local softmax at temperature 0.05;
5. reject candidates within 4 input pixels of an earlier candidate;
6. suppress rejected peaks before deterministic backfill;
7. ties use descending score, then ascending y, then ascending x.

Candidate coordinates, validity, scores, ranks, maps, and hashes are frozen
before teacher access.

## Coordinate oracle and state action

After candidate qualification, the audit reveals the exact frame-15 teacher
coordinate and selects the nearest valid frozen candidate. It writes only that
commit coordinate and deterministic four-level re-extracted memory. Native
visibility/confidence probabilities remain unchanged, following Gate 2.5.
Frames 16--23 are then rolled out identically for all representations.

The primary commit-support radius is 12 pixels. Recall at 4 and 8 pixels is
reported as a nested diagnostic.

## Independent pass gates

Each representation passes only if all conditions hold:

```text
candidate recall within 12 px                 >= 0.75
median nearest-candidate error                <= 8 px
future mean-error reduction vs native         >= 8 px
error-reduction equal-video CI lower          >= 2 px
future threshold-utility gain vs native       >= 0.12
utility-gain equal-video CI lower             >= 0.03
positive failure-point fraction               >= 0.65
severe >=16 px rate reduction                 >= 0.15
```

Bootstrap sample count is 10,000 with frozen seed 193721. The unit resampled is
video, not point.

## Formal branch

```text
M0 passes
  -> AUTHORIZE_GATE3B_SELECTOR_ON_FROZEN_GATE2_DISCRETE_MAP

M0 fails, M1 passes
  -> AUTHORIZE_GATE3B_SELECTOR_ON_GEOMETRY_NATIVE_MAP

M0 and M1 fail, M2 passes
  -> AUTHORIZE_GATE3B_SELECTOR_ON_IMMUTABLE_QUERY_GEOMETRY_MAP

all three fail
  -> STOP_CURRENT_CANDIDATES_AND_REDESIGN_TEMPORAL_IDENTITY_MATCHING

any cache/replay integrity failure
  -> STOP_BEFORE_TEACHER_AUDIT
```

A representation pass authorizes only fit-only causal selector development.
It does not authorize model-validation or external evaluation.

## Exact replay requirements

- Primary and replay candidate sidecars must be nested-exact for every tensor
  and metadata value.
- Candidate membership and combined candidate digest must be exact.
- The entire teacher audit JSON must be exactly equal across independent runs.
- Any replay mismatch overrides metric results.

## Frozen implementation hashes

```text
config:
  817d194d9541394ca953852cd4eb5cf0556b4662cb4963696f0ba8c4cb44d478
discrete candidate extractor:
  af7d5a0bf0619be851c5fdf2ab3e22e257bbbe07f22226fb6388a1a64ddd10c7
geometry matcher:
  784e14d35a9b8e797800d80be4f330eba33d3a21071dd39f4a23b2f7ac350a47
causal anchor extractor:
  73b524583316716cd0ea81623fe17d7458c92b30f77f030d0e6691ecfcd1fefd
causal-input materializer:
  d83a5c32fd97ec8005efa2756e2f4db1af9e07a1162cef3b064c5273fcb3024a
candidate builder:
  e7a587f98ebe98f2fe0c7e08cad06dad29a2ccc12b9d94fba556bddc4fa281a5
candidate replay packager:
  4b426709045d18f36c7e5337d0aaf765f6abda3b5704c4c38d7a1bf78f3e7e0f
teacher audit:
  d4a43c57814b8b49b21b7f7575cfd01fbf7a664855324be0874a58cecc3de16b
final replay packager:
  639741640e2cae26e127ebea91a5b864712969aeb81ec8b7a0101201ce3a2524
```

## Frozen execution order

```bash
python scripts/build_routeD_geometry_causal_input_cache_gate3a_v1.py

python scripts/build_routeD_geometry_representation_cache_gate3a_v1.py \
  --output-root outputs/routeD_geometry_representation_cache_gate3a_v1_20260719

python scripts/build_routeD_geometry_representation_cache_gate3a_v1.py \
  --output-root outputs/routeD_geometry_representation_cache_gate3a_v1_replay_20260719

python scripts/package_routeD_geometry_candidate_cache_gate3a_v1.py \
  --config configs/routeD_geometry_representation_audit_gate3a_v1.yaml \
  --primary-index outputs/routeD_geometry_representation_cache_gate3a_v1_20260719/cache_index.json \
  --replay-index outputs/routeD_geometry_representation_cache_gate3a_v1_replay_20260719/cache_index.json \
  --output outputs/routeD_geometry_representation_gate3a_v1_20260719/candidate_qualification.json

python scripts/audit_routeD_geometry_representation_gate3a_v1.py \
  --candidate-index outputs/routeD_geometry_representation_cache_gate3a_v1_20260719/cache_index.json \
  --candidate-qualification outputs/routeD_geometry_representation_gate3a_v1_20260719/candidate_qualification.json \
  --output outputs/routeD_geometry_representation_gate3a_v1_20260719/primary.json

python scripts/audit_routeD_geometry_representation_gate3a_v1.py \
  --candidate-index outputs/routeD_geometry_representation_cache_gate3a_v1_20260719/cache_index.json \
  --candidate-qualification outputs/routeD_geometry_representation_gate3a_v1_20260719/candidate_qualification.json \
  --output outputs/routeD_geometry_representation_gate3a_v1_20260719/replay.json

python scripts/package_routeD_geometry_representation_gate3a_v1.py \
  --config configs/routeD_geometry_representation_audit_gate3a_v1.yaml \
  --candidate-qualification outputs/routeD_geometry_representation_gate3a_v1_20260719/candidate_qualification.json \
  --primary-report outputs/routeD_geometry_representation_gate3a_v1_20260719/primary.json \
  --replay-report outputs/routeD_geometry_representation_gate3a_v1_20260719/replay.json \
  --output docs/generated/ROUTED_GEOMETRY_REPRESENTATION_GATE3A_V1_SUMMARY_2026-07-19.json
```

No additional dataset or pretrained weight is required for this gate.
