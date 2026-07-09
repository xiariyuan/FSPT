# ReEntry V2.4-DINOScore Fresh Stress Results — Partial Natural Pending — 2026-07-03

## 1. Fixed policy

The dev-selected DINOScore policy is frozen:

```text
target = V22Q
feature = last_candidate_cosine
threshold = 0.395
direction = low_is_bad
```

No threshold tuning was done on fresh.

---

## 2. Engineering updates

### 2.1 Fresh frame-keep datasets

Generated fresh20-49 V2.3 frame-keep datasets:

```text
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/rgb_fresh20_49_natural_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/fresh20_49_translate_L16_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/fresh20_49_occluder_L16_frame_keep_v1thr010_labels_v2.npz
```

Dataset summaries:

| Setting | n_samples | safe16 positive | utility/safe4 positive |
|---|---:|---:|---:|
| natural | 363,629 | 0.6001 | 0.3380 |
| translate_L16 | 348,013 | 0.5887 | 0.3359 |
| occluder_L16 | 342,766 | 0.5944 | 0.3162 |

### 2.2 Stress video support

`build_local_vit_patch_features.py` was extended with:

```text
--stress-dataset
--video-source auto|natural|stress
```

Stress smoke tests were successful for both translate and occluder, confirming that stress videos are read from the corresponding `stress_dataset.pt` rather than natural RGB-Stacking video.

### 2.3 Streaming DINOScore decisions

A memory/compute-efficient streaming decision builder was added:

```text
scripts/build_dinoscore_decisions_stream.py
```

It computes only the fixed-policy score:

```text
last_candidate_cosine
```

and only for rows that are currently visible in the V22Q target cache. This avoids full 3-crop feature extraction on all fresh samples.

Decision chunks are applied/evaluated by:

```text
scripts/apply_dinoscore_decision_chunks.py
```

---

## 3. Fresh translate_L16 result

Target V22Q cache:

```text
outputs/paper_discovery_2026-06-27/reentry_interval_v22/protocol_clean_fresh/fresh20_49_translate_L16_v22Q/reentry_viscalibrator_v22Q_block001_min2_edge015.pt
```

DINOScore output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/translate_v24_micro.pt
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/translate_v24_micro/manifest.json
```

Decision summary:

```text
scanned_rows = 348013
selected_visible_rows = 328325
fires = 1455
flipped = 1384
flipped_over_scanned = 0.00398
```

Metrics:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.5350 | 74.9919 | 92.1943 |
| V24-DINOScore 0.395 | 0.5348 | 74.9987 | 92.1951 |
| Delta | -0.0002 | +0.0068 | +0.0008 |

Interpretation:

```text
Translate fresh shows a very small AJ/OA gain with a tiny AJ_RD decrease.
The effect is much smaller than dev7-9 but directionally mostly consistent.
```

---

## 4. Fresh occluder_L16 result

Target V22Q cache:

```text
outputs/paper_discovery_2026-06-27/reentry_interval_v22/protocol_clean_fresh/fresh20_49_occluder_L16_v22Q/reentry_viscalibrator_v22Q_block001_min2_edge015.pt
```

DINOScore output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/occluder_v24_micro.pt
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/occluder_v24_micro/manifest.json
```

Decision summary:

```text
scanned_rows = 342766
selected_visible_rows = 328562
fires = 2087
flipped = 2019
flipped_over_scanned = 0.00589
```

Metrics:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.6669 | 77.2218 | 92.2326 |
| V24-DINOScore 0.395 | 0.6667 | 77.2276 | 92.2293 |
| Delta | -0.0002 | +0.0058 | -0.0033 |

Interpretation:

```text
Occluder fresh shows a small AJ gain, but AJ_RD and OA decrease very slightly.
This is not a clean all-metric improvement, but the trade-off is much smaller than aggressive RiskScore/DINO thresholds.
```

---

## 5. Natural status

Natural fresh is not complete yet.

Completed natural decision chunks include at least:

```text
0-10000
10000-30000
30000-50000
50000-70000
```

There is also an overlapping `0-50000` chunk from an earlier timed-out run, which must be excluded from clean merging.

The blocker is intermittent I/O failure when reading the large RGB-Stacking pickle:

```text
OSError: [Errno 5] Input/output error
```

This is an environment/file-system issue, not a DINOScore logic error. Stress datasets do not hit this issue because they are read from `stress_dataset.pt`.

---

## 6. Current interpretation

Fresh stress results are weaker than dev7-9:

```text
translate: AJ +0.0068, OA +0.0008, AJ_RD -0.0002
occluder:  AJ +0.0058, OA -0.0033, AJ_RD -0.0002
```

Therefore:

```text
DINOScore 0.395 is not yet a decisive new main method on fresh stress.
It remains a promising, very low-impact appearance micro-filter.
```

Current position:

```text
V1 remains AJ_RD-oriented main method.
V22Q remains the strongest stability-oriented extension.
V24-DINOScore is promising but needs natural fresh and paired-video analysis before promotion.
```

---

## 7. Recommended immediate next step

Resolve natural RGB loading so the same fixed policy can be evaluated on fresh natural.

Recommended options:

```text
1. Continue retrying remaining natural chunks if the file-system I/O stabilizes.
2. Create a small local per-video RGB cache for fresh20-49 natural to avoid repeated loading of the large RGB-Stacking pickle.
3. If natural remains blocked, do not claim fresh completeness; report stress-only partial result honestly.
```

After natural is complete, run paired-video stats for:

```text
V24-DINOScore vs V22Q
V24-DINOScore vs V1
V24-DINOScore vs Base
```
