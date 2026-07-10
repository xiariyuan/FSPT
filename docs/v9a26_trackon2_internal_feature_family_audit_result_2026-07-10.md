# V9-A2.6 TrackOn2 Internal Feature-Family Audit

Date: 2026-07-10

## Protocol

```text
Exactly 557 frozen W16-extension rows; 5-fold video-group-heldout OOF.
Class-balanced logistic regression; event_max + OOF-F1; preserve W8 common.
Feature families are defined by TrackOn2 module semantics; no trajectory threshold sweep.
```

## Family OOF and trajectory apply-back

| Family | Dim | AP | AUC | Brier | Threshold | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| correlation_global | 10 | 0.3087 | 0.6830 | 0.2129 | 0.365020 | +0.1015 | +0.9816 | +0.0289 | 388 | 16/4/5 |
| correlation_alignment | 12 | 0.2679 | 0.5883 | 0.2476 | 0.546086 | +0.1088 | +0.9736 | +0.0293 | 238 | 16/4/5 |
| rerank_scores | 10 | 0.2730 | 0.5981 | 0.2420 | 0.527897 | +0.0964 | +0.9701 | +0.0288 | 325 | 15/5/5 |
| topk_geometry | 8 | 0.2651 | 0.4639 | 0.2525 | 0.040962 | +0.0810 | +0.9701 | +0.0286 | 557 | 15/5/5 |
| visibility_uncertainty | 4 | 0.4368 | 0.6304 | 0.2180 | 0.512518 | +0.1215 | +0.9806 | +0.0296 | 204 | 16/4/5 |
| offsets_proxy_geometry | 5 | 0.2263 | 0.4821 | 0.2457 | 0.071001 | +0.0810 | +0.9701 | +0.0286 | 557 | 15/5/5 |
| query_update | 3 | 0.4230 | 0.6740 | 0.2193 | 0.538482 | +0.1295 | +0.9864 | +0.0299 | 162 | 16/4/5 |
| memory_consistency | 11 | 0.2499 | 0.5436 | 0.2527 | 0.556694 | +0.1192 | +0.9644 | +0.0303 | 237 | 16/4/5 |
| intrinsic_matching | 24 | 0.3116 | 0.6718 | 0.2178 | 0.461603 | +0.0994 | +0.9763 | +0.0285 | 345 | 16/4/5 |
| candidate_alignment | 25 | 0.3167 | 0.5468 | 0.2366 | 0.021317 | +0.0810 | +0.9701 | +0.0286 | 557 | 15/5/5 |
| state_only | 23 | 0.2998 | 0.5519 | 0.2464 | 0.065015 | +0.0876 | +0.9729 | +0.0291 | 548 | 16/4/5 |
| matching_all | 40 | 0.2455 | 0.5571 | 0.2453 | 0.145558 | +0.0851 | +0.9694 | +0.0286 | 526 | 16/4/5 |
| all_internal | 63 | 0.3262 | 0.5972 | 0.2282 | 0.292621 | +0.0664 | +0.9711 | +0.0271 | 394 | 16/4/5 |

## Paired against frozen V9-A2 fixed0.05

| Family | Aggregate Δ | Mean video Δ | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |
|---|---:|---:|---|---|---:|
| correlation_global | -0.000491 | -0.000361 | [-0.001047, +0.000236] | 3/6/16 | 0.3047 |
| correlation_alignment | -0.000125 | +0.001484 | [-0.000555, +0.005006] | 4/4/17 | 0.6562 |
| rerank_scores | -0.000608 | -0.000140 | [-0.001316, +0.000984] | 4/4/17 | 0.7031 |
| topk_geometry | -0.000792 | -0.000665 | [-0.001680, +0.000047] | 2/4/19 | 0.1562 |
| visibility_uncertainty | +0.000188 | +0.002101 | [-0.000501, +0.006112] | 6/5/14 | 0.4131 |
| offsets_proxy_geometry | -0.000792 | -0.000665 | [-0.001680, +0.000051] | 2/4/19 | 0.1562 |
| query_update | +0.000545 | +0.002079 | [-0.000571, +0.006085] | 6/4/15 | 0.3730 |
| memory_consistency | +0.000912 | +0.002261 | [-0.000400, +0.006235] | 8/4/13 | 0.2803 |
| intrinsic_matching | -0.000830 | -0.000560 | [-0.001674, +0.000279] | 3/6/16 | 0.3516 |
| candidate_alignment | -0.000792 | -0.000665 | [-0.001683, +0.000047] | 2/4/19 | 0.1562 |
| state_only | -0.000309 | -0.000249 | [-0.000711, +0.000133] | 2/3/20 | 0.3125 |
| matching_all | -0.000798 | -0.000493 | [-0.001148, +0.000044] | 2/5/18 | 0.1250 |
| all_internal | -0.002302 | -0.001741 | [-0.004757, +0.000143] | 3/6/16 | 0.2109 |

## Decision

**INTERNAL_SIGNAL_REAL_SELECTOR_ROUTE_SATURATED**

TrackOn2 internal matching/state families contain heldout row-level signal, but no predeclared family produces a paired trajectory improvement with a positive lower confidence bound over frozen V9-A2. Stop selector/fusion/family tuning and move to V9-A3 multi-hypothesis candidate generation or trainable reacquisition.
