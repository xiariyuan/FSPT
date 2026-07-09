# ReEntry V2.4-DINOScore Fresh20-49 Full Evaluation — 2026-07-03

## 1. Fixed policy

The dev-selected policy was frozen before fresh evaluation:

```text
target = V22Q
feature = last_candidate_cosine
threshold = 0.395
direction = low_is_bad
```

No threshold tuning was performed on fresh20-49.

---

## 2. Implementation notes

For fresh evaluation, full DINO feature NPZ construction was too expensive because each fresh setting contains more than 340k frame-keep samples.

A streaming decision builder was used instead:

```text
scripts/build_dinoscore_decisions_stream.py
```

It computes only the fixed DINOScore required by the policy:

```text
last_candidate_cosine
```

and only for rows that are currently visible in the V22Q target cache. This reduces compute compared with the earlier 3-crop full feature builder.

A chunk apply/eval script was used:

```text
scripts/apply_dinoscore_decision_chunks.py
```

It merges non-overlapping decision chunks, flips selected visible frames to invisible, and evaluates AJ_RD / AJ / OA.

Important natural-video fix:

```text
The RGB-Stacking dataset only existed as a 2.35GB pickle.
To avoid repeated I/O failures, a fresh20-49 video cache was extracted:
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/rgb_fresh20_49_video_cache.pt
```

For translate/occluder, stress_dataset.pt was used as the video source.

---

## 3. Fresh20-49 frame-keep datasets

Generated datasets:

```text
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/rgb_fresh20_49_natural_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/fresh20_49_translate_L16_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/fresh20_49_occluder_L16_frame_keep_v1thr010_labels_v2.npz
```

Summary:

| setting | n samples | safe16 pos rate | utility/safe4 pos rate |
|---|---:|---:|---:|
| natural | 363629 | 0.6001 | 0.3380 |
| translate_L16 | 348013 | 0.5887 | 0.3359 |
| occluder_L16 | 342766 | 0.5944 | 0.3162 |

---

## 4. Natural fresh20-49

Target cache:

```text
outputs/paper_discovery_2026-06-27/reentry_interval_v22/protocol_clean_fresh/rgb_fresh20_49_natural_v22Q/reentry_viscalibrator_v22Q_block001_min2_edge015.pt
```

V24 output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/natural_v24_micro_clean.pt
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/natural_v24_micro_clean/manifest.json
```

Chunking:

```text
clean chunks = 20
scanned_rows = 363629
selected_visible_rows = 347352
fires = 2294
flipped = 2195
already_invisible_at_apply = 99
selected_visible_over_scanned = 0.9552
fires_over_selected_visible = 0.00660
flipped_over_scanned = 0.00604
```

Metrics:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.4520 | 79.2775 | 92.9233 |
| V24-DINOScore 0.395 | 0.4514 | 79.2884 | 92.9244 |
| Δ | -0.0006 | +0.0109 | +0.0011 |

---

## 5. Translate_L16 fresh20-49

Target cache:

```text
outputs/paper_discovery_2026-06-27/reentry_interval_v22/protocol_clean_fresh/fresh20_49_translate_L16_v22Q/reentry_viscalibrator_v22Q_block001_min2_edge015.pt
```

V24 output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/translate_v24_micro.pt
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/translate_v24_micro/manifest.json
```

Chunking:

```text
n_chunks = 18
scanned_rows = 348013
selected_visible_rows = 328325
fires = 1455
flipped = 1384
already_invisible_at_apply = 71
selected_visible_over_scanned = 0.9434
fires_over_selected_visible = 0.00443
flipped_over_scanned = 0.00398
```

Metrics:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.5350 | 74.9919 | 92.1943 |
| V24-DINOScore 0.395 | 0.5348 | 74.9987 | 92.1951 |
| Δ | -0.0002 | +0.0068 | +0.0008 |

---

## 6. Occluder_L16 fresh20-49

Target cache:

```text
outputs/paper_discovery_2026-06-27/reentry_interval_v22/protocol_clean_fresh/fresh20_49_occluder_L16_v22Q/reentry_viscalibrator_v22Q_block001_min2_edge015.pt
```

V24 output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/occluder_v24_micro_clean.pt
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/occluder_v24_micro_clean/manifest.json
```

A first occluder apply accidentally included overlapping chunks and was discarded. The final result uses:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_fresh20_49/occluder_chunks_clean
```

Clean chunking:

```text
n_chunks = 18
scanned_rows = 342766
selected_visible_rows = 328562
fires = 2087
flipped = 2019
already_invisible_at_apply = 68
selected_visible_over_scanned = 0.9586
fires_over_selected_visible = 0.00635
flipped_over_scanned = 0.00589
```

Metrics:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.6669 | 77.2218 | 92.2326 |
| V24-DINOScore 0.395 | 0.6667 | 77.2276 | 92.2293 |
| Δ | -0.0002 | +0.0058 | -0.0033 |

---

## 7. Fresh20-49 summary

| setting | ΔAJ_RD vs V22Q | ΔAJ vs V22Q | ΔOA vs V22Q | flipped rows |
|---|---:|---:|---:|---:|
| natural | -0.0006 | +0.0109 | +0.0011 | 2195 |
| translate_L16 | -0.0002 | +0.0068 | +0.0008 | 1384 |
| occluder_L16 | -0.0002 | +0.0058 | -0.0033 | 2019 |

Interpretation:

```text
V24-DINOScore 0.395 consistently improves AJ over V22Q on all three fresh settings.
The AJ gain is small: roughly +0.006 to +0.011.
AJ_RD shows a very small negative mean shift: -0.0002 to -0.0006.
OA is mixed: slightly positive on natural/translate, slightly negative on occluder.
```

---

## 8. Current conclusion

V24-DINOScore 0.395 is not a new dominant main method.

It is best described as:

```text
A conservative DINOv3 appearance micro-filter on top of V22Q.
```

Compared with V22Q, it gives a consistent but tiny AJ improvement on fresh20-49, with tiny AJ_RD cost and near-neutral OA.

This is weaker than the dev7-9 result, where AJ and OA improved while AJ_RD stayed unchanged.

Therefore:

```text
V22Q remains the safest stability-oriented extension.
V24-DINOScore is a promising analysis/optional micro-filter, but needs paired-video stats before promotion.
```

---

## 9. Recommended next step

Run paired-video statistics for V24-DINOScore vs V22Q on:

```text
natural
translate_L16
occluder_L16
```

Decision rule after paired stats:

```text
If paired AJ wins are stable and AJ_RD/OA losses are negligible, keep V24-DINOScore as an optional extension.
If paired gains are mixed, leave it as appearance diagnostic and keep V22Q as the final stability method.
```
