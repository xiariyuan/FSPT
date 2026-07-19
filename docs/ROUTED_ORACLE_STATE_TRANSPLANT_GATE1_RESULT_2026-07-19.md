# Route-D oracle state transplant Gate 1 result — 2026-07-19

## 1. Formal decision

```text
AUTHORIZE_FIT_ONLY_COUNTERFACTUAL_STATE_RESTORER_TRAINING_PROTOCOL
```

The preregistered fit-only oracle state transplant gate passes. Primary and
independent replay are exact over all nested tensors and scalars after excluding
output-path fields.

This is an oracle teacher and causal-mechanism result. It is not an
inference-time method result, external-transfer result, or paper-level tracking
claim.

## 2. Frozen protocol

```text
config:
  configs/routeD_oracle_state_transplant_gate1_v0.yaml

config SHA256:
  84e730acd6acac74f983d7858971fa88bd14ca128944ce8d245353dea3804034

fit source videos:
  0--7 from the frozen 64-video Kubric fit manifest

selected natural-failure points:
  63 points from 8 / 8 videos

selection rule:
  future native mean error >= 16 px
  at least the frozen visible-frame requirements
  maximum 8 points per video
  no below-threshold backfill

future evaluation interval:
  frames 16--23
```

No model-validation, calibration, final holdout, TAP-Vid-DAVIS, or
TAP-Vid-Kinetics data were read. The completed official 1,144-video Kinetics
result was not rerun or used for selection.

## 3. Compared interventions

All variants preserve the original point identity and original query-time
contract.

```text
N  native continuation
C  fresh-query overlap coordinates only
P  fresh-query overlap coordinates + visibility/confidence
F  fresh-query coordinates + visibility/confidence + track feature/support memory
```

The fresh branch is created using a fit-only GT query at the last visible frame
inside the frozen overlap. It is a teacher available only for this oracle audit.
No candidate generator or selector is involved.

## 4. Aggregate results

| Variant | Mean future L2 error | Severe >16 px rate | Threshold utility |
|---|---:|---:|---:|
| Native | 60.7129 px | 99.7732% | 0.000454 |
| Coordinate only | 42.3839 px | 97.8175% | 0.012302 |
| Coordinate + probability | 43.2158 px | 96.6270% | 0.019444 |
| Full state | **5.8543 px** | **10.2041%** | **0.638492** |

### Full state versus native

```text
mean error reduction:       +54.8586 px
95% paired-point CI:         [45.9531, 63.9536]
threshold utility gain:      +0.638039
95% paired-point CI:         [0.547222, 0.729707]
positive point fraction:     63 / 63 = 100%
```

### Full state versus coordinate only

```text
mean error reduction:       +36.5296 px
95% paired-point CI:         [33.1271, 40.0367]
threshold utility gain:      +0.626190
95% paired-point CI:         [0.535714, 0.717857]
positive point fraction:     63 / 63 = 100%
better video fraction:      8 / 8 = 100%
```

### Full state versus coordinate + probability

```text
mean error reduction:       +37.3614 px
95% paired-point CI:         [33.4359, 41.2438]
threshold utility gain:      +0.619048
95% paired-point CI:         [0.527778, 0.711111]
positive point fraction:     63 / 63 = 100%
```

All preregistered gates pass.

## 5. Main causal interpretation

The result rejects the hypothesis that recovery is primarily an output-coordinate
problem.

Coordinate-only transplantation reduces the aggregate error relative to native,
but almost all future rows remain severe failures. Adding visibility and
confidence does not materially solve the problem. The decisive improvement
appears only when the track feature and 49-position support memory are also
transplanted.

Therefore the useful teacher target is not merely a corrected coordinate or a
binary commit action. It is a coherent latent tracking state containing:

```text
coordinates;
visibility and confidence;
per-level track feature;
per-level 7 x 7 support memory.
```

This explains why prior output-only candidate selection, late metric adaptation,
and coordinate writeback could improve aligned synthetic cases but failed to
provide stable strong-backbone external transfer.

## 6. Support-memory structure

Pooled support-delta spectral energy is:

```text
rank 1:   25.3733%
rank 2:   39.4313%
rank 4:   55.9874%
rank 8:   73.6226%
rank 16:  89.3524%
```

The target correction is structured but not extremely low rank. A rank-1 or
rank-2 adapter would discard most required energy. Rank 8 captures about 74%,
while rank 16 captures about 89%.

This does not authorize an arbitrary 1.6-million-value dense predictor. The next
model should exploit the 7 x 7 support geometry and shared feature channels,
rather than flattening the entire state into an unconstrained MLP.

## 7. Reproducibility

```text
primary report SHA256:
  403ff7bb35d546c77c06f7c1f1dff6fa9d549616015ba0f9e9c1cc65c88299f3

replay report SHA256:
  e3fac872880ca8c0499eddc7b7edb185f64c6b3e0a87dba7a50d1ad49b47dfc1

primary sidecar SHA256:
  9b87b074ba65ceed8e4786d8f2563b5045b93ae25311c70f6d80b057bc75c5cf

replay sidecar SHA256:
  90b28e16e2d6b7865b81b9e6e655caca88eca8e7fa8a54613e5c6f4c3c52afdb

canonical summary SHA256:
  45f69a7e7fee651ffa7353963e4e8cdb8f798d52ca3f907abc747f568dd8d45e
```

Independent replay checks:

```text
report exact excluding paths:              true
all nested tensors and scalars exact:      true
native continuation replay exact:          true
fresh teacher replay exact:                true
```

## 8. Authorized next step

Gate 1 authorizes only a separately preregistered, fit-only learned restorer
experiment.

The next protocol must:

1. freeze CoTracker3 completely;
2. train only on fit data and use a disjoint fit-internal validation split;
3. predict bounded state residuals or structured re-extraction actions;
4. include explicit no-op examples from clean states;
5. compare against coordinate-only and coordinate-probability learned controls;
6. evaluate future rollout, not only state reconstruction loss;
7. stop before model-validation if fit-internal future-rollout gates fail;
8. keep calibration, final holdout, DAVIS, and Kinetics locked.

A learned restorer must infer a useful fresh-like state from causal observations.
It may not use the GT fresh-query teacher at inference.

## 9. Claim boundary

The evidence currently supports:

> On frozen Kubric fit videos, a GT-derived fresh-query latent state transplanted
> into the original CoTracker3 point slot causally repairs natural future failures,
> and complete support-memory transplantation is necessary beyond coordinate and
> probability correction.

It does not support:

- learned inference-time recovery;
- model-validation or holdout improvement;
- real-video transfer;
- DAVIS rescue;
- tracker-agnostic generalization;
- paper-level performance claims.
