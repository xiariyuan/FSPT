# Standard Benchmark Comparison: TAPVid RGB-Stacking Full50 — 2026-07-04

## 1. Scope

This document evaluates the frozen ReEntry pipeline on the reconstructed TAPVid RGB-Stacking full50 split.

Protocol:

```text
Dataset: TAPVid RGB-Stacking
Split: rgb_stacking_000000 -- rgb_stacking_000049
Videos: 50
Queries: 60,829
Re-entry queries: 10,585
Base cache: CoTracker3 offline RGB-Stacking full50
Override cache: CoTracker3 online RGB-Stacking full50
```

This is stronger than the earlier fresh20-49-only result because it covers the complete locally available RGB-Stacking 0-49 evaluation set assembled from existing caches:

```text
dev0-6 + dev7-9 + heldout10-19 + fresh20-49 = full50
```

---

## 2. Main artifacts

Merged base/override caches:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/offline_rgb_stacking_full50.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/online_rgb_stacking_full50.pt
```

Generated method caches:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v1_thr010/reentry_v1_rgb_full50_thr010.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v22Q/reentry_v22Q_rgb_full50.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_dinoscore/reentry_v24_dinoscore_rgb_full50.pt
```

DINOScore support artifacts:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/rgb_full50_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/rgb_full50_video_cache.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_chunks_final_nonoverlap/
```

Paired stats:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/paired_stats_rgb_full50_v24_vs_all.json
docs/standard_benchmark_rgb_full50_paired_stats_2026-07-04.md
```

---

## 3. Query-weighted full50 results

These are the direct evaluation-script outputs over all 50 videos and 60,829 queries.

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline Base | 0.3617 | 79.9345 | 91.6371 |
| online | 0.4028 | 44.8432 | 55.8092 |
| V1 | 0.4414 | 79.4302 | 93.0758 |
| V22Q | 0.4400 | 79.5756 | 93.0481 |
| V24-DINOScore | 0.4394 | 79.5974 | 93.0389 |

Key deltas vs offline Base:

| Method | ΔAJ_RD | ΔAJ | ΔOA |
|---|---:|---:|---:|
| V1 - Base | +0.0797 | -0.5043 | +1.4387 |
| V22Q - Base | +0.0783 | -0.3589 | +1.4110 |
| V24 - Base | +0.0777 | -0.3371 | +1.4018 |

Interpretation:

```text
On RGB-Stacking full50, all ReEntry variants substantially improve AJ_RD and OA over the offline base.
The cost is a small standard-AJ decrease, because the method changes visibility decisions while preserving offline coordinates.
```

---

## 4. Video-weighted paired-stat means

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline | 0.318334 | 79.934460 | 91.637145 |
| online | 0.344838 | 44.843239 | 55.809194 |
| V1 | 0.394862 | 79.430179 | 93.075843 |
| V22Q | 0.393572 | 79.575607 | 93.048122 |
| V24 | 0.393184 | 79.597351 | 93.038938 |

Paired differences of V24:

| Comparison | ΔAJ_RD | ΔAJ | ΔOA | Interpretation |
|---|---:|---:|---:|---|
| V24 - offline | +0.074850 | -0.337109 | +1.401793 | Large AJ_RD/OA gain with small AJ cost |
| V24 - online | +0.048346 | +34.754112 | +37.229744 | V24 dominates raw online because online destroys AJ/OA |
| V24 - V1 | -0.001678 | +0.167172 | -0.036905 | V24 is more AJ-oriented than V1 |
| V24 - V22Q | -0.000388 | +0.021744 | -0.009184 | V24 gives tiny AJ gain over V22Q, tiny AJ_RD/OA cost |

Detailed paired statistics:

| Comparison | Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |
|---|---|---:|---:|---:|---:|
| V24 - offline | AJ_RD | +0.074850 | [0.060516, 0.090088] | 48 / 0 / 2 | 0.000000 |
| V24 - offline | AJ | -0.337109 | [-0.624740, -0.048121] | 11 / 39 / 0 | 0.000090 |
| V24 - offline | OA | +1.401793 | [0.940280, 1.873843] | 40 / 10 / 0 | 0.000024 |
| V24 - V1 | AJ_RD | -0.001678 | [-0.003068, -0.000370] | 12 / 34 / 4 | 0.001641 |
| V24 - V1 | AJ | +0.167172 | [0.107396, 0.253932] | 49 / 1 / 0 | 0.000000 |
| V24 - V1 | OA | -0.036905 | [-0.118835, 0.025332] | 20 / 30 / 0 | 0.202639 |
| V24 - V22Q | AJ_RD | -0.000388 | [-0.000726, -0.000108] | 8 / 20 / 22 | 0.035698 |
| V24 - V22Q | AJ | +0.021744 | [0.009674, 0.038240] | 39 / 6 / 5 | 0.000001 |
| V24 - V22Q | OA | -0.009184 | [-0.028577, 0.005473] | 19 / 25 / 6 | 0.451381 |

---

## 5. DINOScore full50 decision statistics

DINOScore fixed policy:

```text
target = V22Q
feature = last_candidate_cosine
threshold = 0.395
direction = low_is_bad
```

Full50 DINOScore decision stats:

```text
frame-keep rows = 609,090
selected visible rows = 574,871
fires = 6,403
flipped = 6,104
already invisible at apply = 299
flipped / scanned rows = 0.01002
fires / selected visible rows = 0.01114
```

The final chunk directory is non-overlap and covers the full range:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_chunks_final_nonoverlap
n_chunks = 30
coverage = 0 -- 609090
no gaps
no overlaps
```

---

## 6. Final interpretation

The RGB-Stacking full50 result is now the strongest standard/full evidence for the current ReEntry pipeline.

Main claim supported:

```text
On TAPVid RGB-Stacking full50, ReEntry improves re-entry recovery and visibility stability over the CoTracker3 offline base:
  AJ_RD: 0.3617 -> 0.4394 / 0.4414
  OA:    91.6371 -> 93.0389 / 93.0758
```

But the trade-off remains:

```text
Standard AJ is slightly lower than the offline base:
  Base AJ = 79.9345
  V1 AJ   = 79.4302
  V22Q AJ = 79.5756
  V24 AJ  = 79.5974
```

Final method positioning remains unchanged:

```text
V1 Learned:
  Best AJ_RD-oriented main method on full50.

V22Q:
  Stability/interval extension; recovers some AJ over V1 while keeping most AJ_RD/OA gain.

V24-DINOScore:
  Optional AJ-oriented appearance micro-filter; improves AJ over V22Q but introduces tiny AJ_RD/OA costs.
```

---

## 7. Recommended paper statement

Recommended wording:

```text
On TAPVid RGB-Stacking full50, our ReEntry pipeline improves AJ_RD from 0.3617 to 0.4414 with V1, and to 0.4394 with V24-DINOScore. It also improves OA from 91.64 to about 93.04--93.08. This validates the method on a full 50-video RGB-Stacking evaluation, while revealing a small standard-AJ trade-off.
```

Avoid:

```text
All metrics improve.
V24 is universally best.
DINOScore should replace V1/V22Q as the default method.
```
