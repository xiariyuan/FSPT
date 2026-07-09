# B2-WA CLIP/ViT Patch Pilot on RGB dev10 — 2026-06-30

## Goal

Test whether a stronger pretrained ViT descriptor helps B2-WA beyond simple RGB patch descriptors and trajectory/visibility features.

No RGB fresh20-49 is used.

## Loader result

`timm` DINOv2-S/14 pretrained was not available in local cache and failed to load. `open_clip ViT-B/16 openai` successfully loaded and downloaded weights.

## What was run

```text
scripts/smoke_b2wa_dino_loader.py
scripts/build_b2wa_clip_patch_features.py
scripts/audit_b2wa_clip_signal.py
```

Feature file:

```text
outputs/paper_discovery_2026-06-27/b2wa_clip_pilot/rgb_dev10_clip_patch_rows.jsonl
```

The pilot used 2000 stratified RGB dev10 windows. For each window, CLIP features were extracted from query, last-visible, candidate, and four local negative patches.

## Main AUC results

Grouped by video CV.

| target | trajectory-only AUC | CLIP-only AUC | combined AUC | decision |
|---|---:|---:|---:|---|
| window_has_reentry | 0.8411 | 0.5559 | 0.8277 | CLIP does not help |
| harmful_w16 | 0.7386 | 0.4254 | 0.7200 | CLIP hurts |
| helpful_w16 | 0.6957 | 0.5281 | 0.6968 | negligible gain |
| best_reject | 0.6594 | 0.5086 | 0.6597 | no meaningful gain |
| best_long | 0.6868 | 0.5136 | 0.6803 | no gain |

## Interpretation

This CLIP/ViT patch pilot does not pass the continuation gate.

The most important target is harmful W16 false-trigger detection:

```text
trajectory-only AUC = 0.7386
CLIP-only AUC       = 0.4254
combined AUC        = 0.7200
```

Thus, CLIP crop embeddings do not improve harmful override detection. They are likely too semantic/global and not suitable for point-level re-identification in this setup.

## Decision

Do not continue with CLIP crop descriptors.

Current method status:

```text
B2-W16-P2 remains the main method.
B2-WV oracle shows diagnostic headroom.
Runtime-only learned policies are insufficient.
Simple RGB patch descriptors are insufficient.
CLIP crop descriptors are insufficient.
```

If pushing appearance verification further, the next viable route is true dense correspondence features, not crop-level CLIP: e.g. DINOv2 dense patch tokens with a proper pretrained backbone or TrackOn2 internal dense feature hooks.
