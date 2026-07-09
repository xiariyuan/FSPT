# B2-WA Dense Feature Pilot Summary — 2026-06-30

## Goal

Continue the stronger-method path by testing denser pretrained visual features for B2-WA. The goal was to improve harmful override detection beyond trajectory/visibility features.

No RGB fresh20-49 validation data was used.

## Attempts

### 1. DINOv2 via timm

`timm vit_small_patch14_dinov2 pretrained=True` failed because pretrained weights were not available in the local cache and the loader could not locate them from the Hub.

### 2. TrackOn2 internal dense backbone

TrackOn2 has a usable dense backbone interface in code: `Backbone.forward(video)` returns `f4/f8/f16/f32`. However, instantiation currently fails because the environment is missing `mmcv`, which is required by TrackOn2's ViT adapter. We did not try to compile/install mmcv because that is a heavy and risky dependency step.

### 3. CLIP/ViT crop descriptors

CLIP ViT-B/16 OpenAI weights loaded successfully. We extracted query / last-visible / candidate / local-negative crop descriptors for 2000 RGB dev10 windows.

Key AUC:

```text
harmful_w16:
trajectory-only AUC = 0.7386
CLIP-only AUC       = 0.4254
combined AUC        = 0.7200
```

Conclusion: CLIP crop descriptors do not help point-level harmful override detection.

### 4. ResNet50 dense feature sampling

A locally cached pretrained `timm resnet50.a1_in1k` could be loaded as `features_only=True`. We extracted whole-frame dense features and sampled query / last-visible / candidate / local-negative features at point locations.

Key AUC:

```text
window_has_reentry:
trajectory-only AUC = 0.8411
dense-only AUC      = 0.5089
combined AUC        = 0.8428

harmful_w16:
trajectory-only AUC = 0.7386
dense-only AUC      = 0.4729
combined AUC        = 0.7395

helpful_w16:
trajectory-only AUC = 0.6957
dense-only AUC      = 0.4928
combined AUC        = 0.6969

best_reject:
trajectory-only AUC = 0.6594
dense-only AUC      = 0.5383
combined AUC        = 0.6711

best_long:
trajectory-only AUC = 0.6868
dense-only AUC      = 0.4907
combined AUC        = 0.6757
```

Conclusion: ResNet dense features give a small gain for `best_reject`, but not for the core target `harmful_w16`. The gain on harmful override detection is negligible: `0.7386 -> 0.7395`.

## Decision

These dense/appearance pilots do not justify promoting B2-WA as a main-method upgrade.

Current method status:

```text
B2-W16-P2 remains the main method.
B2-WV oracle remains diagnostic headroom.
Runtime-only learned policies are insufficient.
Simple RGB patch features are insufficient.
CLIP crop descriptors are insufficient.
ResNet50 dense features are insufficient for harmful override detection.
```

## If continuing stronger innovation

The only appearance path still worth considering is a true point-correspondence backbone:

```text
1. real DINOv2 dense patch tokens with pretrained DINOv2 weights; or
2. TrackOn2 dense backbone after resolving mmcv / adapter dependency; or
3. a tracker-native local matching feature, not generic classification features.
```

However, these are heavier CCF-A-style extensions. For the current CCF-B / Q2 paper path, the correct decision is to keep B2-W16-P2 as the main method and report learned/appearance verification as exploratory future work.

## Artifacts

```text
scripts/smoke_b2wa_dino_loader.py
scripts/build_b2wa_clip_patch_features.py
scripts/audit_b2wa_clip_signal.py
scripts/smoke_trackon2_backbone_dense.py
scripts/build_b2wa_resnet_dense_features.py
scripts/audit_b2wa_resnet_dense_signal.py
outputs/paper_discovery_2026-06-27/b2wa_clip_pilot/clip_signal_summary.json
outputs/paper_discovery_2026-06-27/b2wa_resnet_dense_pilot/resnet_dense_signal_summary.json
```
