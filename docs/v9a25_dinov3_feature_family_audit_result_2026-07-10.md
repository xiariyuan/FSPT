# V9-A2.5 DINOv3 Feature-Family Audit

Date: 2026-07-10

## Protocol

```text
Exactly 557 frozen W16-extension rows; 5-fold video-group-heldout OOF.
Class-balanced logistic regression; event_max + OOF-F1; preserve W8 common.
No DINO re-encoding and no trajectory threshold sweep.
```

## Integrity

- Row alignment: PASS (557 rows, 57 features).
- Finite rate: 1.000000.
- DINO NPZ SHA256: `9116c02ef2a088a2f882f22cfb4c4be9229913a29e91c68a7c99202df918646f`.
- Model weights SHA256: `208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2`.

## Feature-family OOF and trajectory apply-back

| Family | Dim | AP | AUC | Brier | OOF-F1 threshold | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| history_candidate_direct | 17 | 0.3240 | 0.6420 | 0.2164 | 0.517872 | +0.1059 | +0.9940 | +0.0288 | 327 | 15/5/5 |
| history_native_direct | 17 | 0.2391 | 0.5647 | 0.2306 | 0.411427 | +0.0870 | +0.9788 | +0.0286 | 430 | 15/5/5 |
| candidate_native_contrast | 4 | 0.3605 | 0.6435 | 0.2192 | 0.398517 | +0.0962 | +0.9864 | +0.0290 | 408 | 16/4/5 |
| anchor_consistency | 3 | 0.2375 | 0.5576 | 0.2412 | 0.469455 | +0.1090 | +0.9771 | +0.0296 | 375 | 15/5/5 |
| local_distinctiveness | 16 | 0.3967 | 0.6537 | 0.2070 | 0.347567 | +0.1205 | +0.9818 | +0.0290 | 394 | 16/4/5 |
| history_candidate_identity | 36 | 0.3703 | 0.6532 | 0.1980 | 0.341304 | +0.1138 | +0.9933 | +0.0292 | 417 | 16/4/5 |
| full_without_candidate_native_contrast | 53 | 0.3404 | 0.6230 | 0.2081 | 0.339183 | +0.1100 | +0.9765 | +0.0287 | 404 | 16/4/5 |
| full_dino | 57 | 0.3637 | 0.6429 | 0.1990 | 0.325552 | +0.1068 | +0.9967 | +0.0287 | 408 | 16/4/5 |

## Paired against frozen V9-A2 fixed0.05

| Family | Aggregate Δ | Mean video Δ | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |
|---|---:|---:|---|---|---:|
| history_candidate_direct | -0.000547 | +0.000721 | [-0.001833, +0.004591] | 4/8/13 | 0.9272 |
| history_native_direct | -0.000805 | -0.000668 | [-0.001747, +0.000127] | 3/5/17 | 0.2266 |
| candidate_native_contrast | -0.000347 | -0.000259 | [-0.000756, +0.000137] | 2/3/20 | 0.3750 |
| anchor_consistency | +0.000228 | -0.000180 | [-0.001303, +0.000716] | 6/3/16 | 0.8008 |
| local_distinctiveness | -0.000350 | -0.000466 | [-0.001519, +0.000330] | 3/4/18 | 0.4219 |
| history_candidate_identity | -0.000132 | -0.000291 | [-0.001303, +0.000422] | 3/4/18 | 0.7031 |
| full_without_candidate_native_contrast | -0.000647 | -0.000557 | [-0.001586, +0.000158] | 2/5/18 | 0.3125 |
| full_dino | -0.000699 | -0.000586 | [-0.001599, +0.000131] | 2/5/18 | 0.2344 |

## Decision

**REAL_SEMANTIC_RANKING_SIGNAL_NO_ROBUST_TRAJECTORY_GAIN**

Historical candidate identity and local distinctiveness provide real video-heldout ranking signal, so the DINO result is not explained only by candidate-native disagreement. However, none of the predeclared feature families establishes a robust trajectory improvement over frozen V9-A2. Stop DINO concatenation/fusion/family tuning and move to TrackOn2 internal matching/memory features.
