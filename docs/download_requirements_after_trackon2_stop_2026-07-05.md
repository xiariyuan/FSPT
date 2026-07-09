# Download Requirements After TrackOn2 Stop — 2026-07-05

## Decision

Do **not** download more TrackOn2 assets now. TrackOn2 DINOv3/DINOv2 checkpoints already exist locally and the strided/original path has been audited as unsuitable for ReEntry plug-in.

Do **not** re-download BootsTAPNext unless checksum/file corruption is suspected. The server already has:

```text
checkpoints/tapnext/bootstapnext_ckpt.npz  # 741M
```

The missing high-value asset is TAPNext++.

## Must download / acquire

### 1. TAPNext++ official code repository

Required because local search found no TAPNext++ code:

```text
No local TAPNext++ repo/code found.
Only references/papers/tapnextpp_arxiv2604.10582.pdf exists.
```

The code should provide:

```text
1. model definition
2. checkpoint loading
3. inference API for arbitrary query points
4. visibility output format
5. coordinate output format
6. evaluation or demo script
```

Target local path:

```text
external/tapnextpp/
```

### 2. TAPNext++ pretrained checkpoint

Required because old status file says:

```text
outputs/attempt0_2026-06-15_recovery/status/tapnextpp.json
reproduction_status = blocked
known_blockers = ["No checkpoint available in repository", "No local weights found"]
```

Target local path:

```text
checkpoints/tapnextpp/
```

File type is unknown until the official release is inspected. Expected possibilities:

```text
*.pt
*.pth
*.npz
*.safetensors
checkpoint directory with config + weights
```

### 3. TAPNext++ official config / model card / README

Required for protocol parity.

Save under:

```text
external/tapnextpp/README.md
external/tapnextpp/configs/...
checkpoints/tapnextpp/README_OR_SOURCE.txt
```

It must answer:

```text
1. expected input frame scale: uint8 0..255 or float 0..1
2. expected query format: [t,x,y], [t,y,x], normalized or pixel
3. output coordinates: xy or yx; pixel or normalized
4. visibility/occlusion convention
5. supported query modes: first-query, strided/arbitrary query times, online streaming
6. expected model input resolution / resize policy
```

## Already available locally

### TrackOn2

```text
baselines/track_on/checkpoints_trackon2_dinov2.pt
baselines/track_on/checkpoints_trackon2_dinov3.pt
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/
```

No more TrackOn2 downloads are needed for the current route.

### BootsTAPNext / TAPNext wrapper

```text
baselines/track_on/ensemble/tapnext/*.py
checkpoints/tapnext/bootstapnext_ckpt.npz
```

This is **not TAPNext++**. It can be used only as a fallback baseline / sanity route, not the preferred SOTA rescue route.

## Optional fallback downloads

Only if TAPNext++ cannot be acquired:

### A. Vanilla TAPNext checkpoint

Current local file:

```text
checkpoints/tapnext/bootstapnext_ckpt.npz
```

If we want to compare vanilla TAPNext and BootsTAPNext separately, download vanilla TAPNext checkpoint too:

```text
checkpoints/tapnext/tapnext_ckpt.npz
```

Priority: low, because existing TAPNext/BootsTAPNext strided-original cache already showed coordinate/protocol collapse.

### B. LocoTrack checkpoint/code

Use only if it has an easier parity-valid inference path than TAPNext++.

Target paths:

```text
external/locotrack/
checkpoints/locotrack/
```

Priority: medium fallback.

### C. TAPIR / BootsTAPIR checkpoints

Use only as older baseline sanity, not SOTA rescue.

Target paths:

```text
checkpoints/tapir/
```

Priority: low.

## What not to download now

```text
1. More CoTracker3 assets — current route already has enough.
2. More TrackOn2 assets — strided/original failure is not due to missing checkpoint.
3. Random DINO/DINOv2 assets — not the bottleneck for the next route.
4. Training datasets — we are not training V26 or TAPNext++ yet.
5. Kinetics/Kubric full datasets — too expensive before SOTA parity is proven.
```

## After download: required smoke steps

After TAPNext++ code/checkpoint are downloaded:

```text
1. Import smoke:
   Can Python import model and load checkpoint?

2. 1-video inference smoke:
   Run one DAVIS video, max 32 or 64 queries if needed.

3. Cache schema export:
   Convert output into FSPT schema:
     query_points: [N,3] [t,y_norm,x_norm]
     pred_tracks: [N,T,2] yx normalized
     pred_visibility: [N,T] bool
     gt_tracks / gt_visibility / original_size copied

4. Geometry audit:
   query anchor mean/max
   coordinate range
   visible-frame coordinate error

5. Standard metrics:
   AJ / OA / delta_avg

6. AJ_RD / visibility oracle:
   original
   GT visibility
   re-entry-only oracle
   all-visible stress
```

## Go / no-go after download

Proceed to ReEntry plug-in only if TAPNext++ satisfies:

```text
query anchor clean
standard metrics plausible
coordinate quality not collapsed
GT/re-entry visibility oracle shows AJ_RD headroom >= +0.01
```

Stop if:

```text
checkpoint unavailable
inference API cannot support arbitrary queries
coordinate convention cannot be reconciled
standard metrics collapse like TrackOn2/TAPNext strided cache
GT visibility only improves AJ_RD but AJ/delta_avg remain collapsed
```
