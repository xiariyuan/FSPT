# B2-WA Appearance Pilot on RGB dev10 — 2026-06-29

## Goal

Test whether lightweight appearance patch features can improve B2-WV window-level verification beyond trajectory/visibility features.

No RGB fresh20-49 is used.

## What was run

### 1. RGB dev10 mini video cache

Created a reusable small video cache from the original RGB-Stacking pkl:

```text
scripts/extract_b2wa_rgb_dev_video_cache.py
outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10/
```

The original pkl is about 2.3GB, so this cache avoids repeated full-pkl loading.

### 2. Patch feature extraction

Extracted patch features for 2000 stratified RGB dev10 candidate windows:

```text
scripts/build_b2wa_patch_features_from_cache.py
outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/rgb_dev10_patch_rows.jsonl
```

Sample labels:

```text
n_rows = 2000
window_has_reentry = 752
w16_harmful_full = 267
w16_helpful_target = 1052
best_action reject = 649
best_action W4 = 660
best_action W8 = 303
best_action W16 = 388
```

Appearance features include multi-radius RGB/histogram/gradient descriptors and query/last-visible/candidate similarity features.

### 3. Appearance signal audit

```text
scripts/audit_b2wa_appearance_signal.py
outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/appearance_signal_summary.json
```

## Main AUC results

Grouped by video CV.

| target | trajectory-only AUC | appearance-only AUC | combined AUC | decision |
|---|---:|---:|---:|---|
| window_has_reentry | 0.8411 | 0.6862 | 0.8263 | appearance does not help |
| harmful_w16 | 0.7386 | 0.4865 | 0.7341 | appearance fails |
| helpful_w16 | 0.6957 | 0.5662 | 0.6997 | tiny gain only |
| best_reject | 0.6594 | 0.5393 | 0.6663 | tiny gain only |
| best_long | 0.6868 | 0.5189 | 0.6916 | tiny gain only |

## Interpretation

This lightweight RGB patch appearance pilot does **not** pass the continuation gate.

The most important target is harmful W16 false-trigger detection. For that target:

```text
trajectory-only AUC = 0.7386
appearance-only AUC = 0.4865
combined AUC = 0.7341
```

So simple appearance descriptors do not identify harmful override windows and do not improve over trajectory/visibility features.

There are tiny combined gains for helpful_w16 / best_reject / best_long, but they are too small to justify full extraction or cache-level B2-WA training.

## Decision

Do not promote B2-WA from simple RGB patch descriptors.

Current status:

```text
B2-W16-P2 remains main method.
B2-WV oracle remains diagnostic headroom.
Runtime-only learned policies remain insufficient.
Simple RGB patch appearance features are insufficient.
```

A stronger appearance verifier would likely need learned descriptors such as DINO/ViT patch features or a dedicated re-identification feature extractor. That is a heavier CCF-A-style extension, not needed for the current CCF-B/Q2 paper path.

## Artifacts

```text
scripts/extract_b2wa_rgb_dev_video_cache.py
scripts/build_b2wa_patch_features_from_cache.py
scripts/audit_b2wa_appearance_signal.py
outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10/manifest.json
outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/patch_feature_summary.json
outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/appearance_signal_summary.json
```
