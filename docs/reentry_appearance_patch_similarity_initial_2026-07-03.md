# ReEntry Appearance Patch Similarity Initial Diagnostic — 2026-07-03

## 1. Goal

After V2.3 oracle showed a large keep/drop upper bound and numeric RiskScore reached a tradeoff boundary, we tested whether lightweight RGB patch similarity provides useful additional information.

The first objective was diagnostic only:

```text
Can RGB patch consistency separate safe recovery frames from harmful V1-proposed recovery frames?
```

No training or method replacement was attempted.

---

## 2. Frame access inspection

New script:

```text
scripts/inspect_rgb_frame_access.py
```

Outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/inspect_rgb_frame_access.json
docs/reentry_appearance_frame_access_inspection_2026-07-03.md
```

Findings:

```text
Natural RGB-Stacking frames can be read via TAPVidRGBStackingDataset.
Translate_L16 and occluder_L16 frames can be read from stress_dataset.pt.
All inspected videos have shape [250, 256, 256, 3].
record original_size matches [256, 256].
frame_count matches records.
patch extraction for 9/17/33 sizes works and inspected patches had valid_ratio=1.0.
```

Frame access is therefore usable for appearance work.

---

## 3. Patch-similarity feature builder

New script:

```text
scripts/build_reentry_patch_similarity_dataset.py
```

For each V1 proposed recovery frame sample, it extracts:

```text
query-frame patch at query point
last reliable base-visible patch before trigger_t
candidate patch at base coordinate at frame_t
```

Patch sizes:

```text
9x9, 17x17, 33x33
```

Features per patch pair:

```text
valid_min
mean absolute RGB difference
mean squared RGB difference
cosine similarity
normalized cross correlation
mean color L2
std color L2
gradient magnitude mean absolute difference
```

Extra features:

```text
last_visible_age_norm
has_last_visible_ref
query_candidate_age_norm
```

Total patch feature dimension:

```text
51
```

Important implementation note:

```text
The first builder attempt produced a feature-name order mismatch; this was corrected.
The builder now uniformly samples across the full NPZ when --max-samples is used, rather than taking a prefix.
```

---

## 4. Smoke test: first 5000 samples

Output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/patch_similarity/rgb_dev7_9_patch_similarity_5000.npz
docs/reentry_patch_similarity_5000_analysis_2026-07-03.md
```

Result:

```text
n_samples = 5000
videos_loaded = 1
patch_feature_dim = 51
```

Top y_safe16 patch AUCs were very high in this prefix-only subset:

| Feature | AUC |
|---|---:|
| query_base_s33_grad_mad | 0.9216 |
| query_base_s17_grad_mad | 0.9174 |
| last_visible_base_s17_grad_mad | 0.9015 |
| query_base_s9_ncc | 0.8983 |
| query_base_s17_ncc | 0.8982 |

However, this subset loaded only one video, so it was not representative. It was treated as a smoke test only.

---

## 5. Uniform dev7-9 sample: 9000 samples over 3 videos

Output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/patch_similarity/rgb_dev7_9_patch_similarity_uniform9000.npz
outputs/paper_discovery_2026-06-27/reentry_appearance/patch_similarity/rgb_dev7_9_patch_similarity_uniform9000_analysis.json
docs/reentry_patch_similarity_uniform9000_analysis_2026-07-03.md
```

Result:

```text
n_samples = 9000
videos_loaded = 3
patch_feature_dim = 51
```

### y_safe16: patch features

Top RGB patch features after uniform sampling:

| Feature | AUC | Direction | AP |
|---|---:|---:|---:|
| query_base_s17_ncc | 0.5493 | negative | 0.7796 |
| query_base_s33_ncc | 0.5459 | negative | 0.7799 |
| last_visible_base_s9_ncc | 0.5459 | negative | 0.7733 |
| last_visible_base_s17_ncc | 0.5413 | negative | 0.7777 |
| query_base_s9_ncc | 0.5405 | negative | 0.7736 |

### y_safe16: numeric reference features on the same 9000 samples

| Feature | AUC | Direction | AP |
|---|---:|---:|---:|
| frame_override_border_dist_norm | 0.8315 | positive | 0.9224 |
| frame_override_y_norm | 0.8308 | positive | 0.9171 |
| frame_base_border_dist_norm | 0.7990 | positive | 0.9105 |
| frame_base_y_norm | 0.7987 | positive | 0.9077 |
| event_gate_prob | 0.7852 | positive | 0.8856 |

Interpretation:

```text
Across multiple videos, raw RGB patch similarity is much weaker than existing numeric features for y_safe16.
```

---

## 6. Other labels

### y_gt_visible

Patch features:

```text
best AUC ≈ 0.5467
```

Numeric features:

```text
best AUC ≈ 0.8393
```

### y_safe8

Patch features:

```text
best AUC ≈ 0.5458
```

Numeric features:

```text
best AUC ≈ 0.8586
```

### y_utility / y_safe4

Patch features:

```text
best AUC ≈ 0.5399
```

Numeric features:

```text
best AUC ≈ 0.9155
```

---

## 7. Conclusion

Frame access and patch extraction are working, but raw RGB patch similarity does not provide a strong cross-video signal in this first diagnostic.

Key conclusion:

```text
The very high prefix-only AUC was a single-video artifact.
Uniform sampling across dev7-9 reduces patch-similarity AUC to about 0.54--0.55.
Existing numeric features are much stronger on the same samples.
```

Therefore:

```text
Do not build a V2.4 model using only simple RGB patch similarity.
```

RGB patch similarity may still be useful as a weak auxiliary feature, but it is not enough to close the V2.3 oracle gap.

---

## 8. Recommended next step

The next useful appearance step should use stronger representation than raw RGB patches:

```text
DINO / ViT patch feature similarity
or
external tracker teacher agreement
```

Most practical next route:

```text
1. Build a small DINO/ViT patch embedding diagnostic if pretrained features are available locally.
2. If not available, test external tracker teacher agreement first, e.g. TAPIR / LocoTrack / TAPNext++ if checkpoints/code are available.
3. Continue to keep V1 as the AJ_RD main method and V22Q as the current best stability extension.
```

Current method positioning remains unchanged:

```text
V1 Learned = AJ_RD-oriented main method.
V22Q = best current stability-oriented extension.
V23-RiskScore = useful numeric diagnostic but not final replacement.
RGB patch-similarity = frame access works, but raw RGB similarity is too weak across videos.
```
