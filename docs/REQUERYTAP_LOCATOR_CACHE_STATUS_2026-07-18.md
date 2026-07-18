# ReQueryTAP Locator Cache Status — 2026-07-18

The frozen fit-only locator cache is complete.

```text
train scenes:       12
train valid pairs:  42,615
dev scenes:          5
dev valid pairs:    17,160
points per scene:      256
target frames:       40--55
cache size:          ~415 MiB
```

Every cache uses the official TAPNext++ `lin_proj + image_pos_emb` patch tokens
at 256x256.  Query identity features come only from frame 0.  Target tokens and
coordinates come from frames 40--55.  No model-validation or locked data was
read.

The untrained raw-cosine baseline is weak on all five dev scenes:

```text
animal:                                      median 75.11 px, hit@16 23.77%
scene_recording_20210910_S05_S06_0_ego2:    median 123.78 px, hit@16 8.22%
r0_new_f_:                                   median 68.13 px, hit@16 19.92%
ani13_new_f:                                 median 80.51 px, hit@16 10.57%
r2_new_:                                    median 62.04 px, hit@16 25.55%
```

This establishes that the exact-point locator must learn cross-frame identity;
a raw patch cosine or native-coordinate reset is not a viable method.

The cache index records a cache-contract SHA over only scene split, frames,
point sampling, seed, and feature-extraction fields.  Training-only additions
such as deterministic early-stop selection do not invalidate the already-built
features, but the complete current training-manifest SHA is also recorded.
