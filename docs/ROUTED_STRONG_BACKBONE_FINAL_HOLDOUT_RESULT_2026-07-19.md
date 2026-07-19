# Route-D strong-backbone final synthetic holdout result — 2026-07-19

## 1. Formal decision

The frozen output-only P0j variant C passes every preregistered P0l gate on the
complete identity-disjoint 16-video Kubric final holdout:

```text
AUTHORIZE_FROZEN_EXTERNAL_PROTOCOL_PREREGISTRATION
```

This authorizes writing a frozen external protocol. It does not itself open
TAP-Vid-DAVIS and does not authorize rerunning the completed official Kinetics
1,144-video evaluation.

## 2. Claim boundary

```text
model: finalized-state output-only variant C
fit/model selection: completed before holdout access
calibration: not read
final holdout: validation source indices 16--31
checkpoint selection on holdout: none
state writeback: disabled
DAVIS: unread
Kinetics official 1,144: unread and not rerun
```

The holdout was first sealed into a complete 16/16 cache without computing any
performance. Metrics were revealed only after cache completion. Primary and
replay evaluations were then run in separate processes.

## 3. Cache integrity

```text
cache index SHA-256:
6c12db0addf5bfcd0f83e8d515acc7c9249e8a8143bc18aacfbaef5ee7d99009

cache payload SHA-256:
762420b96eae598396a6c9ab10e7f22a3b642408182182ea0e0dadeb78d4394d

completed identities: 16 / 16
source indices: 16--31
full extraction replay: indices 16, 23, 31 exact
maximum fp16 error: 0.0002134442
minimum cosine: 0.999999404
```

No partial AJ, per-video gain or gate was computed during cache construction.

## 4. Final aggregate result

| Metric | Native | Variant C | Coordinate oracle | C gain | Oracle gain |
|---|---:|---:|---:|---:|---:|
| AJ | 29.2580 | 30.0259 | 46.0069 | **+0.7679** | +16.7489 |
| Delta average | 42.9457 | 43.8347 | 60.5092 | **+0.8889** | +17.5635 |
| OA | 86.5721 | 86.5721 | 86.5721 | +0.0000 | +0.0000 |

Paired-video direct AJ gain:

```text
mean:   +0.6837 points
95% CI: [+0.4532, +0.9184]
positive / negative videos: 14 / 2
median video gain: +0.6371
minimum / maximum: -0.0414 / +1.6511
```

Paired-video delta-average gain:

```text
mean:   +0.9067 points
95% CI: [+0.5550, +1.2625]
```

## 5. Threshold profile

| Threshold | Variant-C gain | Oracle gain |
|---|---:|---:|
| <1px | -0.2739 | +3.6541 |
| <2px | +0.3113 | +13.9318 |
| <4px | +1.9111 | +28.0565 |
| <8px | +2.0045 | +24.7697 |
| <16px | +0.4918 | +17.4054 |

Variant C retains the previously observed trade-off: 1px precision decreases
slightly, while 2/4/8/16px accuracy improves, with the largest gains at 4px and
8px. The paper must report this profile rather than describing the gain as
uniform across localization scales.

Error summaries also improve:

```text
mean visible error:   13.2963px -> 13.0608px
median visible error: 6.4089px -> 5.8758px
```

## 6. Safety and behavior

```text
visible post-query rows:          16064
selected non-native rate:         5.7333%
harmful non-native rate:          0.9462%
beneficial candidate availability:45.6549%
beneficial candidate recall:      9.3128%
```

Severe 16px error rate:

```text
native:   25.9462%
selected: 25.4669%
change:   -0.4793 percentage points
oracle:   8.6591%
```

The pooled harmful rate passes the frozen 1% ceiling. Per-video harmful rates are
heterogeneous and can exceed 1%; the maximum is approximately 3.0790% on source
index 29. The method claim is therefore pooled low-risk improvement, not a
per-video hard safety guarantee.

## 7. Per-video audit

| Source index | Video | AJ gain | Delta gain | Harmful rate |
|---:|---|---:|---:|---:|
| 16 | 179 | +0.9666 | +1.2150 | 0.8679% |
| 17 | 981 | +0.4405 | +0.4204 | 1.2763% |
| 18 | 502 | +0.9778 | +1.3910 | 0.5639% |
| 19 | 559 | +0.4117 | +0.5004 | 0.9830% |
| 20 | 522 | +0.5525 | +0.3613 | 0.3188% |
| 21 | 954 | +1.1759 | +1.4565 | 0.7993% |
| 22 | 711 | +1.6511 | +2.1998 | 1.2109% |
| 23 | 515 | +0.1341 | +0.1661 | 0.4613% |
| 24 | 609 | +1.0297 | +1.6393 | 1.7654% |
| 25 | 504 | +1.1584 | +1.6413 | 1.9930% |
| 26 | 868 | +0.4159 | +0.3070 | 0.4723% |
| 27 | 545 | +0.2266 | +0.2023 | 0.0000% |
| 28 | 500 | -0.0414 | -0.0371 | 0.5561% |
| 29 | 249 | +0.7218 | +1.0977 | 3.0790% |
| 30 | 605 | -0.0070 | +0.0000 | 1.4347% |
| 31 | 286 | +1.1242 | +1.9457 | 0.5613% |

Fourteen videos improve AJ. The two regressions are small (`-0.0414` and
`-0.0070` points), while the maximum gain is `+1.6511` points. No per-video row
was used for checkpoint or policy selection.

## 8. Exact replay

Primary and replay structured result files are byte-identical:

```text
ae4290a55b2ea07cc1921a04cd701ac85ce1e7bd69ad27f7ebc5272a23c00dd9
```

The two runs reproduce exactly:

- checkpoint and component-state hashes;
- normalization;
- all candidate-coordinate hashes;
- pooled and per-video metrics;
- bootstrap intervals;
- behavior and severe-tail statistics.

## 9. Formal gate

| Gate | Result |
|---|---|
| Complete 16-video partition | PASS |
| Native candidate parity exact | PASS |
| Candidate oracle AJ >= +3.0 | PASS — +16.7489 |
| Direct AJ >= +0.5 | PASS — +0.7679 |
| Paired direct AJ CI lower > 0 | PASS — +0.4532 |
| Delta gain > 0 | PASS — +0.8889 |
| Severe 16px not worse | PASS |
| Harmful non-native <= 1% | PASS — 0.9462% |
| Positive videos >= 12/16 | PASS — 14/16 |
| Exact primary/replay | PASS |

## 10. Scientific interpretation

The final holdout supports the central strong-backbone mechanism:

> A frozen learned multi-memory proposal core, late metric adaptation and a local
> safety comparator produce a statistically consistent, low-risk improvement on
> unseen synthetic identities without calibration or holdout tuning.

The result also preserves two explicit limitations:

1. the gain is output-only, not closed-loop;
2. tight 1px precision and per-video harmful rates are not uniformly improved.

## 11. Next authorized step

P0l authorizes only **preregistration** of a frozen external protocol. Before any
external read, the protocol must pin:

- exact checkpoint and model-state hashes;
- exact DAVIS sample/query/raster contract;
- no calibration or threshold selection;
- one aggregate evaluation and exact replay;
- success/failure gates and claim boundary;
- official Kinetics 1,144 result remains frozen and is not rerun.

## 12. Canonical artifact

```text
docs/generated/ROUTED_STRONG_BACKBONE_FINAL_HOLDOUT_SUMMARY_2026-07-19.json
SHA-256: 918d75ee72aa868676dacb04d5d066fa80696a46328649bf7fe84e302c46b05c
```

## 13. External follow-up boundary — 2026-07-19

P0l authorized only the preregistration of a frozen external protocol. It did not
establish external transfer. The subsequent complete P0m DAVIS audit fails:

```text
DAVIS direct AJ gain:       -1.9170
paired-video 95% CI:        [-2.4472,-1.4449]
DAVIS oracle AJ gain:       +0.4936
harmful non-native rate:    2.2148%
positive videos:            0 / 30
```

Therefore the P0l claim remains restricted to identity-disjoint synthetic
Kubric generalization. It cannot be extended to DAVIS, broad domain transfer, or
tracker-agnostic safety. This negative follow-up does not invalidate the frozen
Kubric result; it defines its external boundary.

Canonical P0m result:
`docs/ROUTED_STRONG_BACKBONE_DAVIS_EXTERNAL_RESULT_2026-07-19.md`.
