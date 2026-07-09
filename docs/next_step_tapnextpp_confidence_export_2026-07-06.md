# Next Step — TAPNext++ Confidence/Logit Export — 2026-07-06

## Decision

The next step is **not V26 training** and not another offline-stress table.

The next step is:

```text
Rerun TAPNext++ normal-online DAVIS first/input export while saving raw visibility logits and sigmoid confidence.
Then run a threshold/calibration sweep on fixed strong TAPNext++ coordinates.
```

## Why

The controlled visibility-lag audit already shows the cleanest SOTA-compatible diagnostic so far:

```text
TAPNext++ normal-online coordinates fixed.
Only re-entry visibility is delayed.
Restoring visibility recovers AJ/OA/AJ_RD.
All coordinate thresholds remain unchanged.
```

Key controlled-lag results:

```text
Original TAPNext++ online:
  AJ       65.85
  OA       92.32
  delta_avg 79.09
  AJ_RD   0.5961

Lag 2 restored gain:
  AJ     +1.13
  OA     +1.28
  AJ_RD  +0.0815
  delta_avg +0.00

Lag 4 restored gain:
  AJ     +2.76
  OA     +3.07
  AJ_RD  +0.1512
  delta_avg +0.00

Lag 8 restored gain:
  AJ     +5.32
  OA     +5.85
  AJ_RD  +0.2642
  delta_avg +0.00
```

This proves visibility lag is a real separable failure axis even when TAPNext++ coordinates are strong.

However, the current TAPNext++ cache does **not** contain confidence/logits:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v4.pt
record keys do not include pred_vis_conf or pred_vis_logit.
```

Reason found during review:

```text
scripts/eval_tapnextpp_davis_full_v4.py had pred_vis_conf saving logic.
scripts/eval_tapnextpp_davis_full_v4_gpu.py overwrote the same output cache path without pred_vis_conf.
```

Therefore, realistic confidence-threshold calibration cannot be tested yet.

## Immediate implementation task

Create a new non-overwriting exporter:

```text
scripts/eval_tapnextpp_davis_full_v5_conf.py
```

It should be based on:

```text
scripts/eval_tapnextpp_davis_full_v4_gpu.py
```

But must save these fields per record:

```text
pred_tracks          # [N,T,2] yx normalized, same as before
pred_visibility      # [N,T] bool, vl_logit > 0
pred_vis_logit       # [N,T] float32 raw vl_all
pred_vis_conf        # [N,T] float32 sigmoid(vl_all)
gt_tracks
gt_visibility
query_points
original_size
model_input_size
```

Do **not** overwrite:

```text
tapnextpp_davis_first_input_cache_v4.pt
```

New outputs:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_full_eval_v5_conf.json
```

## Required validation after export

After creating the v5 cache, immediately verify:

```bash
python - <<'PY'
import torch, numpy as np
p='outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt'
pay=torch.load(p,map_location='cpu',weights_only=False)
r=pay['records'][0]
print(sorted(r.keys()))
print('has logit', 'pred_vis_logit' in r)
print('has conf', 'pred_vis_conf' in r)
for k in ['pred_visibility','pred_vis_logit','pred_vis_conf']:
    a=np.asarray(r[k])
    print(k, a.shape, a.dtype, float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanmean(a)))
PY
```

Must pass:

```text
pred_vis_logit exists
pred_vis_conf exists
pred_vis_conf in [0,1]
pred_visibility == (pred_vis_logit > 0)
```

## Then run threshold sweep

Create:

```text
scripts/eval_tapnextpp_visibility_threshold_sweep.py
```

Input:

```text
tapnextpp_davis_first_input_cache_v5_conf.pt
```

Sweep:

```text
tau in {0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95}
```

Visibility rule:

```text
pred_visibility_tau = pred_vis_conf >= tau
```

Metrics:

```text
AJ
OA
delta_avg
J@4px
J@8px
J@16px
AJ_RD
AJ_RD @D4
AJ_RD @D16
pred_visible_rate
false_visible_rate
visibility_lag statistics if easy
```

Expected output:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_visibility_threshold_sweep.json
docs/tapnextpp_visibility_confidence_sweep_2026-07-06.md
```

## Interpretation gates

### If sweep shows a useful threshold trade-off

Example:

```text
Some tau improves AJ_RD/OA with minimal AJ loss, while coordinates remain unchanged.
```

Then next step is a deployable confidence-based visibility calibrator:

```text
Use TAPNext++ strong coordinates.
Use only inference-time confidence/logit + local temporal features.
Recover/keep visibility when expected metric utility is positive.
```

### If sweep shows no useful threshold trade-off

Then controlled lag remains only an oracle diagnostic:

```text
Visibility lag is important in principle, but TAPNext++ native confidence is already well calibrated or not recoverable by thresholding.
```

Do not train V26 unless a real deployable signal exists.

## What not to do next

```text
Do not train V26 yet.
Do not use TAPNext++ offline_w8 as a main method result.
Do not use TrackOn2 offline_batch as a main method result.
Do not overwrite the existing v4 cache.
Do not claim ReEntry improves normal online TAPNext++ until a deployable confidence/logit-based method is shown.
```

## Current safe paper use

Current safe use:

```text
Diagnostic appendix:
TAPNext++ online controlled visibility-lag audit.
```

Unsafe use:

```text
Main result claiming deployable TAPNext++ improvement.
```
