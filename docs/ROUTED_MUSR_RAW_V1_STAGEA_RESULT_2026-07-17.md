# Route-D MUSR raw-v1 Stage-A result — 2026-07-17

## 1. Scope and claim boundary

This is a selector-only result on the frozen Kubric `fit` and
`model_validation` partitions. Candidate coordinates, validity masks, source
IDs, CoTracker3 native outputs, query mode, search radius, top-K, loss,
optimizer, seed, and gates are unchanged from the structured-64 Stage-A run.
Only the candidate/state representation changes from `64/32` to `601/160`.

Calibration, final holdout, TAP-Vid-DAVIS, and the corrected 1,144-video
TAP-Vid-Kinetics result were not read.

## 2. Representation and cache integrity

```text
candidate token: 601 dimensions
state token:     160 dimensions
fit videos:       48 / 48 complete
validation:       16 / 16 complete
```

The raw-v1 representation contains the frozen 64-D structured token, full
128-D query support and candidate features, their difference and product, and a
candidate-centered 5x5 correlation patch. The state token contains the frozen
32-D structured state plus the full 128-D level-0 `online_track_feat`.

Integrity checks:

- all 64 development videos preserve the exact candidate-coordinate hash;
- structured 64-D and 32-D prefixes are byte-identical to the frozen cache;
- frozen native state is reproduced exactly from the backbone;
- replay anchors at fit indices 0 and 47 and validation indices 48 and 63 are
  byte-identical for both raw candidate and raw state features;
- pre-query appended raw features are zero;
- GT is used only for the unchanged training labels and oracle audit, never to
  construct the representation.

```text
config SHA-256:
a02675499599913184503ecf7d228713f2d78675573313dee70b3daee042375c

fit cache-index SHA-256:
5c331737dbbd81579d4b2a9169309d459a8198db329fe2c5725334616f0d78f6

model-validation cache-index SHA-256:
8848921016725899b7520e9fb9cc03fb8ba2a2e4ffba270eb35980fa9f455a10
```

## 3. Training contract

```text
stage: selector-only Stage A
seed: 17
optimizer: AdamW
learning rate: 3e-4
weight decay: 1e-4
batch size: 512
epoch limit: 10
patience: 4
best epoch: 3
executed epochs: 8
state-write heads trained: no
calibration read: no
```

Native-safe initialization retains exact zero-step metric parity. The formal
run and an independent replay have identical epoch history, best epoch, final
metrics, gate decision, and model-state hash.

```text
model-state SHA-256:
0c82c755ac6f7bbdcdc1bf09bf4877105c681c0bd2db010e0c0f0c3c34c74870

checkpoint SHA-256:
4fc38b8c17347a3d750381d46396b226787ba3bd09a4a936dc47006e1233e113
```

## 4. Model-validation result

| Metric | Native | Raw-v1 selector | Gain |
|---|---:|---:|---:|
| AJ | 25.0135 | 24.9985 | **-0.0149 point** |
| Delta average | 37.1739 | 37.3029 | +0.1290 point |
| OA | 84.9784 | 84.9784 | 0 |

Paired-video AJ gain:

```text
mean: -0.0190 point
95% CI: [-0.0806, +0.0502]
positive videos: 6 / 16
median per-video gain: -0.0285 point
minimum: -0.2215 point
maximum: +0.3581 point
```

Threshold hit-rate changes:

| Threshold | Gain |
|---|---:|
| <1 px | -0.1290 point |
| <2 px | +0.0368 point |
| <4 px | +0.1965 point |
| <8 px | +0.2825 point |
| <16 px | +0.2579 point |

Selector behavior:

```text
non-native selection rate:       7.29%
oracle-utility match rate:      59.62%
mean utility regret:             0.0906
harmful global selection rate:   0.84%
16px severe-error-rate change:  -0.264 percentage point
```

The method becomes slightly more conservative in the severe tail but loses
strict 1-pixel accuracy, leaving AJ unchanged to slightly negative.

## 5. Comparison with structured-64

| Stage-A representation | AJ gain | Delta gain | Oracle match |
|---|---:|---:|---:|
| Structured 64/32 | +0.0173 | +0.2432 | 61.83% |
| Raw-v1 601/160 | -0.0149 | +0.1290 | 59.62% |
| Raw-v1 minus structured | **-0.0322** | -0.1142 | -2.21 pp |

The raw representation does not recover more of the frozen `+6.8463 AJ`
coordinate-oracle headroom.

## 6. Failure diagnosis

### 6.1 Not a simple normalization or covariate-shift failure

Across all raw feature groups, fit-to-validation standardized mean-shift RMS is
only approximately `0.02–0.06`; validation-to-fit standard-deviation ratios are
near one. No candidate dimension is near constant and only one state dimension
is constant by construction. Therefore the negative result is not explained by
an obvious normalization error or large marginal distribution drift.

### 6.2 Single-frame recall is the bottleneck

On complete model validation:

```text
rows with a truly beneficial non-native candidate: 37.63%
selector non-native selection rate:                  7.37%
beneficial-event precision:                         42.25%
beneficial-event recall:                             8.27%
beneficial-event F1:                                13.84%
```

The selector remains safe but misses more than 91% of the rows where the frozen
candidate pool could improve utility.

### 6.3 Beneficial events are temporally persistent

```text
P(beneficial at t+1 | beneficial at t):      69.31%
P(beneficial at t+1 | not beneficial at t): 18.65%
oracle-rank lag-1 agreement:                 59.13%
median beneficial run length:                 2 frames
90th percentile run length:                   6 frames
runs lasting at least 2 frames:              56.92%
runs lasting at least 3 frames:              32.66%
```

This is direct evidence that independent single-frame classification discards a
useful temporal signal.

## 7. Gate

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| AJ gain | >= +0.5 | -0.0149 | FAIL |
| Paired AJ CI lower | > 0 | -0.0806 | FAIL |
| Delta gain | > 0 | +0.1290 | PASS |
| Severe 16px rate | not worse | -0.264 pp | PASS |
| Harmful selection | <= 1% | 0.837% | PASS |
| Zero-step native parity | exact | exact | PASS |
| Seed-17 replay | exact | exact | PASS |

```text
STOP_RAW_V1_AND_START_CAUSAL_TEMPORAL_REPRESENTATION
```

State-write Stage B, calibration, final holdout, DAVIS, and Kinetics remain
closed.

## 8. Artifacts

```text
projects/mmp_tracker/mmp_tracker/routeD_raw_representation.py
scripts/build_routeD_cotracker3_raw_v1_cache.py
scripts/package_routeD_musr_raw_v1_result.py
configs/routeD_musr_cotracker3_raw_v1.yaml
docs/generated/ROUTED_MUSR_RAW_V1_STAGEA_SUMMARY_2026-07-17.json
```

Versioned summary SHA-256:

```text
874844af7fb2f898e8428ed1159bd7c4f4c0fcb108179fb1467637817b975e73
```
