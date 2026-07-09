# B2-WV Appearance Verifier Plan — 2026-06-29

Current prediction caches do not store video frames, only tracks/visibility/GT. Raw TAPVid DAVIS and RGB-Stacking pkl files are each about 2.3GB, and direct loader smoke tests timed out at 60-120s. Therefore, full appearance-feature extraction should be opt-in and cached, not run repeatedly.

## Implemented script

```text
scripts/build_b2wv_appearance_patch_features.py
```

The script extracts lightweight patch descriptors for candidate windows:

```text
query patch
last-visible base patch
override candidate patch
RGB mean/std + gradient energy
query/override similarity
last-visible/override similarity
```

Default is capped with `--max-rows 2000` to avoid heavy full-dataset loading. If this pilot shows useful signal, then we can cache full RGB dev10 features once.

## Decision

Do not promote B2-WV learned policy yet. Runtime-only policies lose too much AJ_RD. A real upgrade likely requires appearance-level verification, but full extraction should be run only as a deliberate paid-compute step.
