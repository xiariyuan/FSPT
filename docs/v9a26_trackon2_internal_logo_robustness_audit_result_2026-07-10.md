# V9-A2.6 TrackOn2 Internal LOGO Robustness Audit

Date: 2026-07-10

## Protocol

```text
Leave-one-video-out OOF over all 20 extension-bearing videos.
All 13 predeclared internal feature families; class-balanced logistic regression.
event_max + global LOGO-OOF-F1; W8 common preserved; no trajectory threshold sweep.
```

## LOGO results

| Family | AP | AUC | Threshold | AJ Δ | OA Δ | AJ_RD_256 Δ | Aggregate vs frozen | Paired mean | 95% CI | Better/Worse/Equal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| correlation_global | 0.3082 | 0.6807 | 0.353440 | +0.0977 | +0.9816 | +0.0288 | -0.000575 | -0.000419 | [-0.001119, +0.000204] | 3/6/16 |
| correlation_alignment | 0.2946 | 0.6301 | 0.536126 | +0.1085 | +0.9684 | +0.0295 | +0.000125 | +0.002110 | [-0.000546, +0.006130] | 5/5/15 |
| rerank_scores | 0.2737 | 0.6108 | 0.503532 | +0.1035 | +0.9701 | +0.0291 | -0.000323 | +0.001644 | [-0.001223, +0.005712] | 5/5/15 |
| topk_geometry | 0.2712 | 0.4879 | 0.046763 | +0.0810 | +0.9701 | +0.0286 | -0.000792 | -0.000665 | [-0.001680, +0.000047] | 2/4/19 |
| visibility_uncertainty | 0.4390 | 0.6312 | 0.504616 | +0.1192 | +0.9806 | +0.0295 | +0.000137 | +0.002123 | [-0.000492, +0.006128] | 7/5/13 |
| offsets_proxy_geometry | 0.2270 | 0.4808 | 0.054126 | +0.0810 | +0.9701 | +0.0286 | -0.000792 | -0.000665 | [-0.001680, +0.000047] | 2/4/19 |
| query_update | 0.4216 | 0.6824 | 0.533657 | +0.1318 | +0.9864 | +0.0300 | +0.000659 | +0.002153 | [-0.000499, +0.006168] | 7/4/14 |
| memory_consistency | 0.2812 | 0.5671 | 0.597827 | +0.0992 | +0.9772 | +0.0284 | -0.001026 | +0.000343 | [-0.003201, +0.004647] | 6/5/14 |
| intrinsic_matching | 0.3482 | 0.6983 | 0.532396 | +0.1015 | +0.9763 | +0.0285 | -0.000899 | -0.000549 | [-0.001581, +0.000192] | 2/7/16 |
| candidate_alignment | 0.3661 | 0.5944 | 0.420173 | +0.1053 | +0.9836 | +0.0288 | -0.000556 | -0.000211 | [-0.001517, +0.001081] | 4/6/15 |
| state_only | 0.3104 | 0.5638 | 0.074827 | +0.0810 | +0.9701 | +0.0286 | -0.000792 | -0.000665 | [-0.001684, +0.000047] | 2/4/19 |
| matching_all | 0.2738 | 0.6044 | 0.337098 | +0.0936 | +0.9763 | +0.0287 | -0.000678 | -0.000452 | [-0.001126, +0.000135] | 2/6/17 |
| all_internal | 0.3874 | 0.6415 | 0.467601 | +0.0911 | +0.9829 | +0.0273 | -0.002047 | -0.000500 | [-0.004194, +0.003047] | 4/6/15 |

## Five-fold versus LOGO stability

| Family | 5-fold AP | LOGO AP | 5-fold AJ_RD_256 Δ | LOGO AJ_RD_256 Δ |
|---|---:|---:|---:|---:|
| correlation_global | 0.3087 | 0.3082 | +0.0289 | +0.0288 |
| correlation_alignment | 0.2679 | 0.2946 | +0.0293 | +0.0295 |
| rerank_scores | 0.2730 | 0.2737 | +0.0288 | +0.0291 |
| topk_geometry | 0.2651 | 0.2712 | +0.0286 | +0.0286 |
| visibility_uncertainty | 0.4368 | 0.4390 | +0.0296 | +0.0295 |
| offsets_proxy_geometry | 0.2263 | 0.2270 | +0.0286 | +0.0286 |
| query_update | 0.4230 | 0.4216 | +0.0299 | +0.0300 |
| memory_consistency | 0.2499 | 0.2812 | +0.0303 | +0.0284 |
| intrinsic_matching | 0.3116 | 0.3482 | +0.0285 | +0.0285 |
| candidate_alignment | 0.3167 | 0.3661 | +0.0286 | +0.0288 |
| state_only | 0.2998 | 0.3104 | +0.0291 | +0.0286 |
| matching_all | 0.2455 | 0.2738 | +0.0286 | +0.0287 |
| all_internal | 0.3262 | 0.3874 | +0.0271 | +0.0273 |

## Decision

**LOGO_CONFIRMS_NO_ROBUST_SELECTOR_GAIN**

Leave-one-video-out OOF does not produce any internal feature family whose paired mean 95% CI has a positive lower bound over frozen V9-A2. The promising 5-fold memory-consistency aggregate result is not sufficient to reopen selector tuning. Proceed to V9-A3 candidate-generation/reacquisition.
