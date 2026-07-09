# TrackOn2 Repo-Native Code Review and Next Step — 2026-07-06

## Review target

Determine the rigorous next step for exporting TrackOn2 visibility logits/confidence with parity to the existing repo-native first/input cache.

## Files reviewed

```text
baselines/track_on/dataset/tapvid.py
baselines/track_on/evaluation/evaluator.py
baselines/track_on/evaluation/eval.py
baselines/track_on/model/trackon_predictor.py
```

## Confirmed repo-native data path

`dataset/tapvid.py` does the following for DAVIS:

```text
frames = resize_video(frames, [256, 256])
target_points *= [256, 256]
query_points format = [t, y, x]
trajectory/tracks format = [T, N, 2] xy in input256 pixel space
video format = [T, 3, 256, 256] in [0,255]
```

`evaluation/evaluator.py::prepare_tapvid_data` then converts:

```text
query_points_i [t,y,x] -> queries [t,x,y]
gt_tracks = trajectory.permute(0,2,1,3)
gt_occluded = not visibility.permute(0,2,1)
```

`evaluation/eval.py` confirms TrackOn2 DAVIS model setup:

```text
if dataset_name == "davis":
    trackon2_args.M_i = 24
model = Predictor(trackon2_args, checkpoint_path=..., support_grid_size=20)
```

`model/trackon_predictor.py` confirms the issue:

```text
forward_frame internally obtains v_logit from self.model.track_frame(...)
then thresholds it with sigmoid(v_logit) >= delta_v
but deletes v_logit and returns only (p, v_t)

forward returns only:
  pred_trajectory
  pred_visibility
```

Therefore logits are available inside the correct model path, but are discarded before cache saving.

## Why the previous hand-written path is no longer acceptable

The old repo-native `.npz` is already verified:

```text
outputs/trackon2_dinov3_davis_cache/davis/trackon2/000000.npz
```

It exactly matches the old unified bridge:

```text
tracks diff = 0
visibility diff = 0
```

The hand-written parity smoke, even after input256 + M_i=24 + schema preservation, still differs:

```text
AJ_256        old 58.4117 vs smoke 57.3041
OA_256        old 90.0588 vs smoke 88.8824
AJ_RD_256     old 0.5257  vs smoke 0.5211
pred_visibility diff_rate = 0.0371
```

Conclusion:

```text
The correct source of truth is the repo-native dataloader/evaluator path, not a manual forward mirror.
```

## Exact next step

Write a small, non-invasive repo-native logit smoke script:

```text
scripts/export_trackon2_repo_native_logits_smoke.py
```

The script must:

```text
1. Import repo-native TAPVid dataset.
2. Use DataLoader with the same settings as evaluation/eval.py for DAVIS.
3. Use Predictor with trackon2_args.M_i = 24 and support_grid_size = 20.
4. Use local DINOv3 via DINOV3_LOCAL_DIR.
5. Run only first DAVIS sample.
6. Follow evaluator.py prepare_tapvid_data for query conversion.
7. Call a logit-preserving TrackOn2 path.
8. Save a new .npz with:
   tracks
   visibility
   visibility_logit
   visibility_conf
9. Compare new tracks/visibility against the old repo-native 000000.npz.
```

## How to implement logit preservation safely

Preferred minimal implementation:

```text
Do not modify default Predictor.forward behavior.
In the smoke script, define a local subclass or local helper that mirrors Predictor.forward but returns logits.
```

However, the helper must consume `video` and `queries` from the repo-native dataloader/evaluator path, not from raw pkl reconstruction.

A permanent patch to `Predictor.forward(return_logits=False)` is allowed only if:

```text
return_logits defaults to False
old call sites continue to return exactly two tensors
new smoke script explicitly calls return_logits=True
```

## Output paths

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke/davis/trackon2/000000.npz
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke_report.json
```

## Pass gate

Compare against:

```text
outputs/trackon2_dinov3_davis_cache/davis/trackon2/000000.npz
```

Required:

```text
new has keys: tracks, visibility, visibility_logit, visibility_conf
old tracks vs new tracks: max_abs_diff <= 1e-4 preferred
old visibility vs new visibility: diff_count = 0 preferred
new visibility == (visibility_conf >= 0.8)
```

If this passes:

```text
Proceed to 30-video repo-native logit export.
```

If this fails:

```text
Stop TrackOn2 true-base route for this paper cycle.
Do not use TrackOn2 as a main ReEntry improvement baseline.
```

## Paper claim status before this passes

Current allowed claim:

```text
TrackOn2 true-base confidence/logit export is not yet parity-validated.
```

Current forbidden claim:

```text
ReEntry improves TrackOn2 true-base.
```
