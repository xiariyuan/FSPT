# ReEntry V2.4-DINOScore Full Dev7-9 Evaluation — 2026-07-03

## 1. Goal

Move DINOScore from label-level diagnostics to full downstream TAP-style evaluation.

The target is to test whether a very conservative DINOv3 appearance micro-filter can improve the current stability-oriented method V22Q without hurting AJ_RD / OA.

---

## 2. Full DINOv3 feature generation

The DINOv3 builder was extended with chunking support:

```text
scripts/build_local_vit_patch_features.py
  --sample-mode slice
  --start-index
  --end-index
```

All dev7-9 V2.3 frame-keep samples were embedded with independent DINOv3:

```text
input frame-keep NPZ:
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/rgb_dev7_9_frame_keep_v1thr010_labels_v2.npz

base cache:
outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_offline_rgb_dev7_9.pt

model:
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

Chunk outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/dev7_9_chunks/
```

Generated chunks:

```text
00000_05000
05000_10000
10000_15000
15000_20000
20000_25000
25000_30000
30000_35000
35000_40000
40000_45000
45000_46041
```

Merged output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev7_9_dinov3_all_crop33.npz
```

Merge summary:

```text
n_chunks = 10
n_samples = 46041
feature_dim = 13
first_index = 0
last_index = 46040
```

---

## 3. Apply/eval script

New script:

```text
scripts/apply_reentry_dinoscore_from_npz.py
```

It loads:

```text
1. target cache, e.g. V22Q dev7-9
2. full DINOv3 feature NPZ
3. row metadata mapping each DINO feature row to (video_id, query_idx, frame_t)
```

Then it applies a pure visibility micro-filter:

```text
if DINOScore rule fires and target cache currently marks this frame visible:
    set target visibility to False
```

Coordinates are never changed.

No GT is used at inference.

---

## 4. Baseline target: V22Q dev7-9

Target cache:

```text
outputs/paper_discovery_2026-06-27/reentry_interval_v22/sweep_dev7_9/v22_Q_block001_min2_edge015/v22_Q_block001_min2_edge015.pt
```

V22Q dev7-9 metrics:

```text
AJ_RD = 0.3212
AJ    = 79.9763
OA    = 94.0033
```

---

## 5. Policy A: ultra-conservative micro-filter

Policy:

```text
feature = last_candidate_cosine
drop if last_candidate_cosine < 0.395
```

Output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_dev7_9/v24_micro.pt
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_dev7_9/v24_micro/manifest.json
```

Counts:

```text
considered_rows = 46041
fired_rows = 409
fired_over_considered = 0.00888

target_visible_rows = 251
flipped_rows = 251
flipped_over_considered = 0.00545
flipped_over_fired = 0.61369
already_invisible_rows = 158
missing_record_rows = 0
```

Label composition among fired rows:

```text
y_safe16:      16 positive / 393 negative
y_gt_visible: 112 positive / 297 negative
y_safe8:       16 positive / 393 negative
y_utility:      7 positive / 402 negative
y_safe4:        7 positive / 402 negative
```

Label composition among actually flipped rows:

```text
y_safe16:      12 positive / 239 negative
y_gt_visible:  62 positive / 189 negative
y_safe8:       12 positive / 239 negative
y_utility:      6 positive / 245 negative
y_safe4:        6 positive / 245 negative
```

Metrics:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.3212 | 79.9763 | 94.0033 |
| V24-DINOScore 0.395 | 0.3212 | 79.9950 | 94.0168 |
| Δ | +0.0000 | +0.0187 | +0.0135 |

Interpretation:

```text
The ultra-conservative DINOScore micro-filter improves AJ and OA while preserving AJ_RD on dev7-9.
This is the first downstream evidence that DINOv3 appearance signal can improve the V22Q stability method.
```

---

## 6. Policy B: stronger threshold sanity check

Policy:

```text
feature = last_candidate_cosine
drop if last_candidate_cosine < 0.453
```

Output:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_dev7_9/v24_micro_lastcos0453.pt
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_score_dev7_9/v24_micro_lastcos0453/manifest.json
```

Counts:

```text
considered_rows = 46041
fired_rows = 1255
fired_over_considered = 0.02726

target_visible_rows = 925
flipped_rows = 925
flipped_over_considered = 0.02009
flipped_over_fired = 0.73705
```

Metrics:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| V22Q | 0.3212 | 79.9763 | 94.0033 |
| V24-DINOScore 0.453 | 0.3199 | 80.0209 | 93.9966 |
| Δ | -0.0013 | +0.0446 | -0.0067 |

Interpretation:

```text
The stronger threshold increases AJ more, but it starts to trade off AJ_RD and OA.
This matches the earlier label-level threshold sweep: beyond the ultra-conservative regime, safe-positive loss rises quickly.
```

---

## 7. Current conclusion

The correct V2.4-DINOScore position is:

```text
V24-DINOScore 0.395 is a promising ultra-conservative appearance micro-filter on top of V22Q.
It improves dev7-9 AJ and OA while keeping AJ_RD unchanged.
```

Do not use the stronger threshold as the main policy:

```text
last_candidate_cosine < 0.453 gives higher AJ but reintroduces the same trade-off seen in RiskScore.
```

---

## 8. Recommended next step

Freeze the dev-selected DINOScore policy:

```text
target = V22Q
feature = last_candidate_cosine
threshold = 0.395
```

Next run fresh20-49 with the same fixed policy:

```text
natural
translate_L16
occluder_L16
```

Important caution:

```text
Fresh20-49 has already been used repeatedly during development, so if this becomes a paper claim, an additional untouched holdout such as fresh50-79 should be generated later.
```
