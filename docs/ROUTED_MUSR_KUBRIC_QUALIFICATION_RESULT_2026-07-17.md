# Route-D MUSR Kubric candidate qualification result — 2026-07-17

## 1. Decision

```text
PASS the complete frozen 15-video candidate-qualification gate.
ALLOW export of fit/model-validation/calibration caches and MUSR training.
DO NOT call the oracle gain a learned result.
DO NOT read final_holdout, TAP-Vid-DAVIS, or TAP-Vid-Kinetics.
```

The candidate pool contains material and video-consistent coordinate headroom on
a frozen strong backbone. This closes P0c and opens P0d, but does not establish
that MUSR can learn to select the candidates safely.

## 2. Frozen membership

The qualification partition is validation-source indices `1–15`, exactly 15
videos. Validation-source index `0`, used during the earlier interface pilot, is
permanently excluded. Indices `16–31` remain the untouched final synthetic
holdout.

```text
protocol SHA-256:
317a9ad2dbf36c1ed9a2c10eaa408967682f6f95a3870dfbaf61329077ed2fb0

qualification identity SHA-256:
3e357f840799589231d57ca3c80225492e46f9314518a8e3c3371bf48b5c3183
```

## 3. Aggregate official first-query metrics

Visibility is the unchanged native CoTracker3 visibility. The oracle changes
only coordinates by selecting the minimum-GT-error candidate from the frozen
six-candidate pool.

| Metric | Native CoTracker3 | Coordinate oracle | Gain |
|---|---:|---:|---:|
| AJ | 28.5946 | 35.1443 | **+6.5497** |
| Delta average | 42.0255 | 50.6433 | **+8.6178** |
| OA | 86.3058 | 86.3058 | 0.0000 |
| <1 px | 16.4939 | 23.0861 | **+6.5922** |
| <2 px | 24.8671 | 34.8884 | **+10.0213** |
| <4 px | 38.3506 | 47.5744 | **+9.2238** |
| <8 px | 55.3628 | 64.2477 | **+8.8849** |
| <16 px | 75.0532 | 83.4197 | **+8.3666** |

All five threshold-hit gains are strictly positive.

## 4. Cross-video consistency

```text
videos: 15 / 15 complete
positive AJ-gain videos: 15 / 15
videos with AJ gain >= +1.0: 15 / 15
minimum per-video AJ gain: +4.2208
median per-video AJ gain: +6.4035
mean per-video AJ gain: +6.4915
maximum per-video AJ gain: +9.4179

minimum per-video delta gain: +6.9355
median per-video delta gain: +8.4568
mean per-video delta gain: +8.6157
```

The result is therefore not driven by one or two favorable videos.

## 5. Integrity and determinism

Every video passed exact routing-disabled parity:

```text
candidate 0 == native CoTracker3 coordinates
candidate 0 valid everywhere
max absolute coordinate difference = 0
```

The adapter export was independently repeated at source indices `1`, `8`, and
`15`. Candidate coordinates, 64-dimensional candidate features, candidate
scores, validity masks, source IDs, 32-dimensional state features, native
coordinates, visibility probabilities, confidence probabilities, and native
visibility were bit-identical in all three checks.

```text
cache-index canonical payload SHA-256:
eac5ac62a406fc2f246f583770ab599b5240d1052723e480de23bfefaf4d3ab8

cache-index file SHA-256:
76d759444e5a88e3c25af9a7f3c26cad71de14ee73bd7d42a76ce2c9f07a6517

ordered 15-sidecar-hashes SHA-256:
c2618fd61f323fe9e322cb9aa705c057fef3617d55345569591a01a07bf6b4ff

sidecar count: 15
sidecar total bytes: 43,059,780
```

## 6. Gate evaluation

| Check | Requirement | Result |
|---|---:|---:|
| Complete partition | 15 videos | PASS |
| Exact native parity | all videos | PASS |
| Pooled AJ gain | >= +3.0 | **+6.5497 PASS** |
| Pooled delta gain | >= +4.0 | **+8.6178 PASS** |
| Median per-video AJ gain | >= +2.0 | **+6.4035 PASS** |
| Fraction with AJ gain >= +1.0 | >= 2/3 | **1.0000 PASS** |
| Every threshold gain | > 0 | PASS |
| Replay indices 1/8/15 | exact | PASS |

Final decision emitted by the cache index:

```text
ALLOW_FIT_MODEL_CACHE_EXPORT_AND_MUSR_TRAINING
```

## 7. Claim boundary

This is a GT-only candidate oracle. It establishes candidate reachability, not
learned selection, calibration, abstention, closed-loop safety, or external
transfer. The `+6.55 AJ` number must never be described as MUSR performance.

## 8. Next gate

P0d is now active:

1. export the exact `fit`, `model_validation`, and `calibration` caches;
2. implement a streaming cache dataset and MUSR training harness;
3. train only on `fit`;
4. select checkpoints only on `model_validation`;
5. choose calibration/abstention operating points only on `calibration`;
6. keep `final_holdout`, DAVIS, and Kinetics unread;
7. require learned Kubric AJ gain of at least `+1.0`, positive paired CI, no
   severe-tail regression, and closed-loop improvement before opening the final
   synthetic holdout.

## 9. Versioned artifacts

```text
configs/routeD_musr_kubric_cache_protocol_v0.json
projects/mmp_tracker/mmp_tracker/routeD_kubric_cache.py
scripts/build_routeD_cotracker3_kubric_cache.py
scripts/package_routeD_musr_kubric_qualification.py
docs/ROUTED_MUSR_KUBRIC_CACHE_PROTOCOL_V0_2026-07-17.md
docs/generated/ROUTED_MUSR_KUBRIC_QUALIFICATION_SUMMARY_2026-07-17.json
```

Runtime sidecars remain under `outputs/` and are intentionally not committed.
