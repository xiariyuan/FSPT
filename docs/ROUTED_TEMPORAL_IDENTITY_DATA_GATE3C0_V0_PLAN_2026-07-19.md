# Route-D temporal identity data Gate 3C0 v0 — 2026-07-19

## Status

```text
PREREGISTERED_NOT_RUN
```

## Why the route changes here

Gate 3A v1 and its post-gate controls isolate the failure:

| Signal | Top-8 recall within 12 px | Decision |
|---|---:|---|
| M1 static geometry | 0.3875 | insufficient |
| reverse CoTracker cycle, one-video smoke | 0.3333 | rejected before full run |
| peer relative-motion consensus, best fit-only setting | 0.3750 | insufficient |
| DINOv3 query appearance | 0.3875 | insufficient |
| fixed M1/DINO fusion, best descriptive weight | 0.4375 | insufficient |
| M1 broad top-128 oracle | 0.9125 | support exists deeper in the ranking |

The reverse-cycle control reuses the same tracker after it has already lost
identity. Peer geometry also fails because multiple points drift confidently.
DINOv3 contributes independent appearance signal but cannot solve the task as a
static scalar. No additional threshold or fusion sweep is authorized.

The necessary change is a candidate-conditioned spatiotemporal identity model:
each broad proposal must be evaluated against the observed video history, not
only one static query descriptor or one endpoint correlation map.

## Why the existing 24 training videos are not enough

Gate 2 trained on source indices 8--31, only 24 videos. A top-128 identity model
must learn hard within-video negatives, occlusion duration, appearance change,
and distractor motion. Reusing the 16 Gate 3A videos for model selection would
turn the diagnostic into training feedback and leave no credible internal
evidence.

The server already contains all 1,024 Kubric train TFRecord shards. Gate 3C0
therefore materializes 512 deterministic samples and excludes every raw video
record underlying the existing 64-sample protocol.

## Frozen materialization

```bash
python scripts/preprocess_kubric_tfrecord_no_tf.py \
  --tfds-root /gemini/code/datasets/tapvid_kubric \
  --output-dir /gemini/code/FSPT/outputs/routeD_temporal_identity_kubric512_seed42_20260719 \
  --split train \
  --num-points 64 \
  --shard-size 16 \
  --max-samples 512 \
  --sampling-strategy uniform \
  --hard-fraction 0.5 \
  --seed 42 \
  --hash-source-files
```

The preprocessor SHA256 is
`3f92997ed961f9f3067735a43cece90fbb6794c0ce5dfb2b691632e5977c352b`.

## Raw-record-disjoint split

```text
expanded 0--63:
  overlap verification only; prohibited for gradients, selection, or audit

expanded 64--383:
  gradient train, 320 videos

expanded 384--447:
  checkpoint selection, 64 videos

expanded 448--511:
  fit-only internal audit, 64 videos
```

Indices 0--63 must exactly reproduce the existing manifest's sample tensors
under the same seed and point sampler. Even if point sampling differed, their raw
video records would remain excluded. Video identity, not point identity, defines
the split.

The original model-validation indices 48--63 remain locked; the expanded
manifest's first 64 entries correspond to those same raw records and are never
used.

## Next model scope, not yet training authorization

Passing Gate 3C0 authorizes only a causal feature cache for Gate 3C1:

1. mandatory native plus 128 M1 coarse proposals;
2. frozen CoTracker and DINOv3 observations from frames 0--15;
3. candidate-conditioned local descriptor sequences and correlation evidence;
4. a learned listwise/pruning model trained only on 64--383;
5. checkpoint choice only on 384--447;
6. one frozen top-8 audit on 448--511.

The model may use teacher coordinates only to form training labels or audit
metrics. It may not use future frames, teacher-conditioned proposal generation,
or future rollout features.

Architecture, loss, and the Gate 3C1 pass thresholds will be separately
preregistered after data and feature-shape integrity are measured. This prevents
guessing memory/compute constraints and then silently changing the model.

## Gate 3C0 pass

All conditions are mandatory:

- exactly 512 samples and 32 output shards;
- every source TFRecord used is hashed;
- first 64 sample tensor hashes exactly reproduce the existing protocol;
- indices 64--511 are raw-record-disjoint from the existing 64;
- split membership is exact and video-disjoint;
- no locked or external evaluation data is read.

```text
pass -> AUTHORIZE_GATE3C1_CAUSAL_FEATURE_CACHE_ON_EXPANDED_KUBRIC
fail -> STOP_AND_REPAIR_DATA_IDENTITY_BEFORE_MODELING
```

No new user-supplied data or model weight is needed for this stage.
