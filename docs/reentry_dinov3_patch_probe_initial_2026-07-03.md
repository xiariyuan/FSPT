# ReEntry DINOv3 Patch / Teacher Probe Initial Results — 2026-07-03

## 1. Correction / clarification

DINOv3 resources are available in this project. There are two distinct routes:

```text
1. Track-On2 + DINOv3 teacher route:
   baselines/track_on/checkpoints_trackon2_dinov3.pt
   third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m

2. Independent DINOv3 patch-embedding route:
   third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
   config model_type = dinov3_vit
   architecture = DINOv3ViTModel
```

This document records the corrected investigation.

---

## 2. Availability inspection

Script:

```text
scripts/inspect_appearance_teacher_availability.py
```

Output:

```text
docs/reentry_appearance_teacher_availability_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_appearance/teacher_availability.json
```

Key findings:

```text
Best next route: dino_patch_features
Torch: 2.2.2+cu121
CUDA: available
GPU: B1.gpu.large
```

Available packages:

```text
torch, torchvision, timm, transformers, PIL, cv2, sklearn, numpy, scipy
```

Important local resources:

```text
baselines/track_on/checkpoints_trackon2_dinov3.pt
baselines/track_on/checkpoints_trackon2_dinov2.pt
baselines/hf/facebook_dinov2_small/model.safetensors
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
checkpoints/tapnext/bootstapnext_ckpt.npz
```

---

## 3. Track-On2 DINOv3 teacher smoke

Script:

```text
scripts/export_trackon2_reentry_stress_cache.py
```

A new smoke run was executed on dev0 translate_L16, 1 video, 16 queries.

### Default uint8, support_grid=0, unconditional memory

```text
cache: outputs/paper_discovery_2026-06-27/reentry_appearance/trackon2_dinov3_smoke/translate_dev0_16q.pt
model load: ~14.3s
inference: ~13.7s
vis_rate: 0.5052
peak_mem_mb: 509.0
```

Comparison on the same 16 queries:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| CoTracker3 offline | 0.5068 | 52.8872 | 78.5392 |
| CoTracker3 online | 0.5232 | 54.6173 | 80.7731 |
| B2-W16-P2 | 0.5373 | 55.4240 | 82.0030 |
| Track-On2 DINOv3 | 0.2643 | 29.0617 | 79.6938 |

Additional variants:

| Variant | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| unit input, unconditional | 0.0268 | 5.5710 | 54.9699 |
| unit input, visibility-selective | 0.0000 | 0.0000 | 39.7590 |
| uint8, support_grid=20, visibility-selective | 0.2324 | 32.6878 | 67.7460 |

Interpretation:

```text
Track-On2 DINOv3 can run in the current environment.
However, under this small ReEntry stress smoke it is much weaker than CoTracker/B2.
Therefore it should not be used as a strong teacher yet.
```

This does not invalidate DINOv3 appearance features; it only says the full Track-On2 teacher route is not ready as a reliable teacher under the current protocol.

---

## 4. Independent DINOv3 model smoke

The DINOv3 directory:

```text
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

has:

```text
config.json: architecture = DINOv3ViTModel, hidden_size=384, image_size=224, patch_size=16
preprocessor_config.json
```

The standard fast image processor is incompatible with the current torch version because it calls:

```text
torch.compiler.is_compiling
```

which is missing in torch 2.2.2. This was bypassed by manual ImageNet-style preprocessing:

```text
resize to 224x224
rescale 0..1
normalize by mean [0.485, 0.456, 0.406]
normalize by std [0.229, 0.224, 0.225]
```

Smoke output:

```text
model_class: DINOv3ViTModel
last_hidden_state_shape: [1, 201, 384]
cls_embedding_shape: [1, 384]
cls_norm: 12.9326
```

So independent DINOv3 patch embedding is usable.

---

## 5. DINOv3 patch feature builder

New script:

```text
scripts/build_local_vit_patch_features.py
```

For each V1-proposed recovery frame sample, it extracts:

```text
query crop
last reliable base-visible crop before trigger_t
candidate crop at base coordinate and frame_t
```

It computes normalized CLS embeddings and outputs 13 similarity features:

```text
query_candidate_cosine
last_candidate_cosine
best_ref_candidate_cosine
worst_ref_candidate_cosine
query_candidate_l2
last_candidate_l2
best_ref_candidate_l2
worst_ref_candidate_l2
query_last_cosine
query_last_l2
last_visible_age_norm
has_last_visible_ref
query_candidate_age_norm
```

---

## 6. DINOv3 patch feature AUC on dev7-9

Datasets:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev7_9_dinov3_uniform1500_crop33.npz
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev7_9_dinov3_uniform3000_crop33.npz
```

### 1500 uniform samples

```text
n_samples = 1500
videos_loaded = 3
```

Top y_safe16 DINOv3 feature:

```text
best_ref_candidate_cosine AUC = 0.7344
```

### 3000 uniform samples

```text
n_samples = 3000
videos_loaded = 3
```

Top DINOv3 AUCs:

| Label | Best DINOv3 feature | AUC |
|---|---|---:|
| y_safe16 | best_ref_candidate_cosine / L2 | 0.7188 |
| y_gt_visible | best_ref_candidate_cosine / L2 | 0.7142 |
| y_safe8 | last_candidate_cosine / L2 | 0.7545 |
| y_utility / y_safe4 | last_candidate_cosine / L2 | 0.8266 |

Comparison to raw RGB patch similarity:

```text
Raw RGB patch top AUC across videos was only ~0.54--0.55.
DINOv3 patch embedding is much stronger.
```

Comparison to top numeric features on the same 3000 samples:

```text
y_safe16 numeric top AUC ≈ 0.7875
y_gt_visible numeric top AUC ≈ 0.7920
y_safe8 numeric top AUC ≈ 0.8071
y_utility/safe4 numeric top AUC ≈ 0.8178
```

Interpretation:

```text
DINOv3 is not always stronger than numeric features, but it is a real appearance signal.
For y_utility/safe4, DINOv3 slightly exceeds the top numeric feature in single-feature AUC.
```

---

## 7. Hard-subset analysis

Script:

```text
scripts/analyze_vit_hard_subsets.py
```

Output:

```text
docs/reentry_dinov3_patch_uniform3000_hard_subsets_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev7_9_dinov3_uniform3000_hard_subsets.json
```

Key subsets:

| Subset | n |
|---|---:|
| all | 3000 |
| numeric_thinks_safe_event_ge_002_dist_lt_025 | 2651 |
| distance_ambiguous_025_050 | 66 |
| low_v1_prob_010_030 | 452 |
| high_event_low_dist_low_v1 | 239 |
| low_segment_max_lt_050 | 894 |

Important findings:

### numeric-thinks-safe subset

```text
condition: event_gate_prob >= 0.02 and base_override_dist_norm < 0.25
```

| Label | Numeric top AUC | DINOv3 top AUC |
|---|---:|---:|
| y_safe16 | 0.7710 | 0.7103 |
| y_safe8 | 0.7596 | 0.7558 |
| y_utility/safe4 | 0.8000 | 0.8237 |

DINOv3 improves the utility/safe4 signal in the region where numeric features think frames are mostly safe.

### low-v1-prob subset

```text
condition: 0.10 <= v1_prob < 0.30
```

| Label | Numeric top AUC | DINOv3 top AUC |
|---|---:|---:|
| y_safe16 | 0.8300 | 0.8317 |
| y_safe8 | 0.7651 | 0.6072 |
| y_utility/safe4 | 0.7797 | 0.7620 |

DINOv3 is competitive for safe16 in low-v1-prob cases, but numeric features remain better for safe8/utility here.

### high-event low-dist low-v1 subset

```text
condition: event_gate_prob >= 0.02, base_override_dist_norm < 0.25, v1_prob < 0.30
```

| Label | Numeric top AUC | DINOv3 top AUC |
|---|---:|---:|
| y_safe16 | 0.7850 | 0.8384 |
| y_safe8 | 0.8182 | 0.6210 |
| y_utility/safe4 | 0.8866 | 0.8478 |

DINOv3 gives a strong safe16 signal in this hard low-v1 subset.

---

## 8. Logistic probe: dev0-6 to dev7-9

Script:

```text
scripts/train_vit_numeric_probe.py
```

Train:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev0_6_dinov3_uniform3000_crop33.npz
```

Validation:

```text
outputs/paper_discovery_2026-06-27/reentry_appearance/dinov3_patch/rgb_dev7_9_dinov3_uniform3000_crop33.npz
```

All validation samples:

| Features | Label | Balanced AUC | Balanced AP |
|---|---|---:|---:|
| numeric | y_safe16 | 0.6623 | 0.8578 |
| DINOv3 | y_safe16 | 0.6905 | 0.8662 |
| numeric + DINOv3 | y_safe16 | 0.6903 | 0.8709 |
| numeric | y_gt_visible | 0.6035 | 0.8671 |
| DINOv3 | y_gt_visible | 0.6800 | 0.8979 |
| numeric + DINOv3 | y_gt_visible | 0.6300 | 0.8804 |
| numeric | y_safe8 | 0.7409 | 0.7906 |
| DINOv3 | y_safe8 | 0.7136 | 0.7989 |
| numeric + DINOv3 | y_safe8 | 0.7553 | 0.8085 |
| numeric | y_utility/safe4 | 0.7904 | 0.6031 |
| DINOv3 | y_utility/safe4 | 0.8153 | 0.6993 |
| numeric + DINOv3 | y_utility/safe4 | 0.8095 | 0.6526 |

Interpretation:

```text
DINOv3-only generalizes better than numeric-only for safe16, gt_visible, and utility/safe4.
Numeric+DINO improves safe8 and AP in several cases, but naive concatenation is not always better than DINO-only.
This suggests distribution shift / feature calibration issues, not lack of DINO signal.
```

Hard subset examples using the balanced classifier:

```text
numeric-thinks-safe subset:
  y_safe16: numeric AUC 0.6071, DINOv3 AUC 0.6837
  y_gt_visible: numeric AUC 0.6050, DINOv3 AUC 0.6917
  y_utility/safe4: numeric AUC 0.7700, DINOv3 AUC 0.8116

low-v1-prob subset:
  y_safe16: numeric AUC 0.5895, DINOv3 AUC 0.6709, combined AUC 0.6738

low-segment-max subset:
  y_safe8: numeric AUC 0.6730, combined AUC 0.6923
  y_utility/safe4: numeric AUC 0.7469, combined AUC 0.7578
```

---

## 9. Current conclusion

DINOv3 patch embeddings are a real improvement over raw RGB patch similarity.

Key conclusion:

```text
Raw RGB patch similarity is too weak across videos.
DINOv3 patch embedding gives meaningful appearance signal.
DINOv3 is especially useful for utility/safe4 and some hard subsets where numeric features are uncertain.
```

However, a naive numeric+DINO logistic probe is not yet a final method:

```text
Combined features do not consistently dominate DINO-only or numeric-only.
This likely reflects train/val distribution shift and calibration mismatch.
```

Track-On2 DINOv3 teacher route can run, but current smoke result is too weak to use as a strong teacher.

---

## 10. Recommended next step

Do not jump directly to a large V2.4 MLP.

Next step should be a DINOv3-assisted low-capacity filter / score probe:

```text
V2.4-DINOScore diagnostic
```

Use a small number of robust DINOv3 signals:

```text
best_ref_candidate_cosine
last_candidate_cosine
best_ref_candidate_l2
last_candidate_l2
```

Combine with numeric confidence in a constrained way rather than generic MLP.

Suggested first diagnostic policy family:

```text
Start from V22Q / V1 proposed frames.
Only apply DINO decision in ambiguous regions:
  event_gate_prob >= 0.02 and base_override_dist_norm < 0.25
  OR v1_prob < 0.30
  OR segment_prob_max < 0.50

Drop if DINOv3 similarity is below threshold.
```

Selection objective on dev7-9:

```text
AJ_RD >= V22Q
OA >= V22Q - 0.005
maximize AJ
```

Before downstream cache evaluation, first run threshold sweeps on DINOv3 feature NPZ to estimate how many frames would be dropped and their label composition.
