# Next Step — TrackOn2 Repo-Native Logit Instrumentation Plan — 2026-07-06

## Decision

Do not continue hand-written TrackOn2 forward reimplementation.

The next rigorous experiment is to instrument the repo-native evaluation path that originally produced the old TrackOn2 `.npz` caches, and make it save visibility logits/confidence in addition to tracks/visibility.

## Reason

The old TrackOn2 first/input real-base cache is already proven to be repo-native parity:

```text
repo-native DAVIS: AJ=67.04, OA=92.09, delta_avg=79.84
unified bridge:    AJ=67.04, OA=92.09, delta_avg=79.84
abs diff: 0.00
```

Old repo-native cache files:

```text
outputs/trackon2_dinov3_davis_cache/davis/trackon2/000000.npz
...
```

These `.npz` files contain only:

```text
tracks
visibility
```

They do not contain logits.

The hand-written confidence smoke can export logits, but even after schema parity + input256 + M_i=24, first-video metric parity still fails:

```text
old bridge:  AJ_256=58.4117, OA_256=90.0588, AJ_RD_256=0.5257
M24 smoke:   AJ_256=57.3041, OA_256=88.8824, AJ_RD_256=0.5211
```

Therefore, to get logits with paper-grade parity, we must patch the exact repo-native path.

## Relevant source locations

Model setup:

```text
baselines/track_on/evaluation/eval.py
```

Relevant behavior:

```text
if dataset_name == "davis":
    trackon2_args.M_i = 24
model = Trackon_Predictor(trackon2_args, checkpoint_path=..., support_grid_size=20)
```

Prediction and cache save branch:

```text
baselines/track_on/evaluation/evaluator.py
```

Relevant branch:

```text
if cache_predictions and cache_dir and dataset_name:
    cache_path = os.path.join(cache_dir, dataset_name, model_name, f"{j:06d}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        pred_trajectory = torch.from_numpy(cached['tracks']).to(device)
        pred_visibility = torch.from_numpy(cached['visibility']).to(device)
    else:
        with torch.autocast(...):
            pred_trajectory, pred_visibility = model(video.clone(), queries.clone())
        np.savez(cache_path,
                 tracks=pred_trajectory.cpu().numpy(),
                 visibility=pred_visibility.cpu().numpy())
```

## Pre-registered next experiment

### Hypothesis

A minimally instrumented repo-native evaluator can produce a new TrackOn2 cache with:

```text
tracks
visibility
visibility_logit
visibility_conf
```

while preserving exact or near-exact parity with the old repo-native `.npz` tracks/visibility.

### Input

Old repo-native cache for comparison:

```text
outputs/trackon2_dinov3_davis_cache/davis/trackon2/000000.npz
```

Dataset:

```text
datasets/tapvid_davis/tapvid_davis.pkl
```

TrackOn2 checkpoint/config:

```text
baselines/track_on/checkpoints_trackon2_dinov3.pt
baselines/track_on/config/test.yaml
```

Local DINOv3:

```text
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

### Control

Control is the old repo-native `.npz` for the same video.

### Experimental output

New 1-video repo-native logit cache:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke/davis/trackon2/000000.npz
```

Report:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke_report.json
```

### Required keys in new `.npz`

```text
tracks
visibility
visibility_logit
visibility_conf
```

### Pass gate

Strict gate:

```text
old tracks vs new tracks: max_abs_diff <= 1e-4 preferred
old visibility vs new visibility: diff_count = 0 preferred
visibility == visibility_conf >= 0.8
```

If small nondeterminism remains:

```text
tracks mean_abs_diff must be tiny
visibility diff_rate must be near 0 and explained
metric parity must be within the already documented tolerance
```

### Fail gate

If repo-native logit instrumentation still cannot reproduce old `.npz`:

```text
Stop TrackOn2 true-base route for this paper cycle.
Do not use TrackOn2 as main improvement baseline.
Keep TrackOn2 only as context/appendix from existing real-base numbers and diagnostics.
```

## Implementation strategy

Do not change default model behavior globally.

Recommended minimal implementation:

```text
1. Add an optional method or flag to TrackOn Predictor to return logits.
2. Default `Predictor.forward(video, queries)` remains unchanged.
3. Add a separate wrapper/script that calls the optional logit path.
4. Save logits only in a new output directory, never overwrite old repo-native cache.
```

Safer file naming:

```text
scripts/export_trackon2_repo_native_logits_smoke.py
```

This script may either:

```text
A. monkey-patch / subclass Predictor locally and return logits, or
B. copy the small relevant evaluator loop and call a patched Predictor method.
```

But it must follow the repo-native dataloader/evaluator path, not the previous hand-written raw-pkl pipeline.

## What not to do

Do not run 30-video export before the 1-video repo-native logit parity passes.

Do not run ReEntry threshold/local recovery before the 30-video true-base confidence cache exists.

Do not claim TrackOn2 improvement before this success gate is passed:

```text
AJ >= base - 0.10
OA >= base - 0.10
AJ_RD >= base + 0.01
```
