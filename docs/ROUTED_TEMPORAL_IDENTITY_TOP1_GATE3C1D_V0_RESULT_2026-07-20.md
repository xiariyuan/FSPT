# Route-D causal top-1 selector Gate 3C1D v0 result — 2026-07-20

## Formal status

```text
COMPLETED_FAIL
EXACT_REPLAY_PASS
STOP_GATE3C1D_TOP1_SELECTOR
```

The preregistered checkpoint-selection gate was run on exactly indices
`384--447`. No threshold in the frozen grid passed all checkpoint gates. Per the
frozen protocol, fit-only top-1 audit `448--511` and original-model top-1
confirmation `48--63` were not run.

## Integrity

```text
preregister commit:          db83a73
config SHA256:               d3cc35914e277cce6014b5bbc437c0c2536522799e1ca1005ad862ea1941d28f
primary result SHA256:       f404413f8dd74a1c4298cf45a8fdb12797c7b2ea0ed785d093edce08c95e1c07
replay result SHA256:        7ad30da43a9a0772af9ae75e73714b8d32f6b90e35895388b5bb1a517f9f1170
primary model SHA256:        39c944d94881b690476f36b5d754a6fcf581c327fb15d5d2d8a1039aa995bb69
replay model SHA256:         39c944d94881b690476f36b5d754a6fcf581c327fb15d5d2d8a1039aa995bb69
scientific payload SHA256:   9f952d2931623ce9e9d91e9cf5b5aa6a864359afff06e6796ba0f44c266930d1
summary payload SHA256:      23e6c3f1a55e49b42790f71e484eb7ebb516962de8c94592a70388d220394503
```

Primary and replay models are byte-identical. Feature, shortlist, candidate
probability, output slot, threshold-grid, point-record, and complete scientific
payload digests replay exactly.

## Candidate discrimination

The model contains real candidate-level signal but does not produce a usable
native-safe action policy under the frozen gate:

| Metric | Value |
|---|---:|
| candidate AUC | 0.827372 |
| candidate AP | 0.422708 |
| raw top-1 support within 12 px | 35.8162% |

## Frozen threshold sweep

| Threshold | Coverage | Action precision <=12 px | Mean error reduction | Video CI lower | Harmful all-row rate | Pass |
|---:|---:|---:|---:|---:|---:|---|
| 0.25 | 46.12% | 51.55% | +4.663 px | +2.968 px | 4.28% | no |
| 0.30 | 31.70% | 56.00% | +3.205 px | +2.127 px | 2.22% | no |
| 0.35 | 23.45% | 60.81% | +2.445 px | +1.498 px | 1.11% | no |
| 0.40 | 16.64% | 64.76% | +1.849 px | +0.990 px | 0.63% | no |
| 0.45 | 11.25% | 70.42% | +1.286 px | +0.592 px | 0.48% | no |
| 0.50 | 7.61% | 75.00% | +0.951 px | +0.634 px | 0.00% | no |
| 0.55 | 4.91% | 83.87% | +0.591 px | +0.313 px | 0.00% | no |
| 0.60 | 2.22% | 85.71% | +0.222 px | +0.090 px | 0.00% | no |

The failure is structural rather than a single bad threshold:

- lower thresholds provide useful coverage and error reduction but violate
  precision and/or harmful-rate gates;
- higher thresholds become safe and precise but lose the required coverage and
  aggregate error reduction;
- the nearest near-boundary point is threshold `0.40`, where precision is
  `64.76%` versus the required `65%`, coverage is `16.64%` versus `20%`, mean
  reduction is `1.849 px` versus `2.0 px`, and the CI lower bound is `0.990 px`
  versus `1.0 px`.

These near misses are not grounds for relaxing or rounding the preregistered
thresholds.

## Scientific interpretation

The HGB candidate classifier improves raw causal top-1 support to `35.82%`, but
one scalar candidate-success probability cannot simultaneously solve candidate
ranking and row-level action safety. Probability magnitude is not sufficiently
calibrated across videos to identify when a non-native commit is both correct and
better than native.

The next design should separate:

1. **candidate ranking**, trained to choose the best shortlist candidate; and
2. **row-level action value/risk**, trained on out-of-fold candidate predictions
   to estimate improvement over native and harmful-action probability.

Because checkpoint-selection results are now observed, indices `384--447` can no
longer serve as an independent confirmation partition for a redesigned v1. A new
raw-record-disjoint Kubric expansion is required before another formal top-1
confirmation sequence.

## Claim boundary

Gate 3C1D v0 failed at commit-level top-1 selection. It does not invalidate the
query-closure shortlist or the coordinate-plus-memory recovery mechanism. It does
reject the current single-stage HGB probability plus scalar-threshold policy.
Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
unread.
