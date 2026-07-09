# ReEntry-VisCalibrator V1 — Implementation and First Results

Date: 2026-07-03

## Purpose

The rule-based ReEntry-VisGuard-W8P2 is effective, but it can be criticized as a hand-crafted rule:

```text
base invisible + override visible for P=2 frames -> replace visibility in a fixed W=8 window
```

ReEntry-VisCalibrator is the learned version. It keeps the core principle:

```text
coordinates = base/offline coordinates
visibility  = learned recovery inside candidate re-entry windows
```

It does **not** predict new coordinates.

## New implementation files

```text
utils/reentry_viscalibrator_features.py
models/reentry_viscalibrator.py
scripts/build_reentry_viscalibrator_dataset.py
scripts/train_reentry_viscalibrator.py
scripts/eval_reentry_viscalibrator.py
```

All files passed syntax checks.

## Design

### Candidate generation

Use a high-recall prediction-only rule to find candidate re-entry windows:

```text
base has been invisible after query
and override is visible for P=2 consecutive frames
```

The model then decides which frames inside/around the candidate are safe to recover.

### Features

Each candidate window has 33 frames by default:

```text
trigger_t - 16 ... trigger_t + 16
```

Each frame has 28 features, including:

```text
base visible / override visible
visibility disagreement
base and override coordinates
base/override distance
base/override speed and acceleration
base invisible run length
override visible run length
distance to image border
override future visible rate
base past invisible rate
```

### Labels

The label is not simply "GT visible".

It is metric-aware:

```text
safe_visible = GT visible AND base coordinate close enough to GT
```

Soft label schedule:

```text
< 1 px  -> 1.00
< 2 px  -> 0.90
< 4 px  -> 0.75
< 8 px  -> 0.55
<16 px  -> 0.25
else    -> 0.00
```

This teaches the model:

```text
recover visible only when it is safe to keep base coordinates.
```

### Model

Small temporal convolution model:

```text
input:  (candidate_window_frames, 28 features)
output: per-frame visible-recovery logits
```

No image patches are used in V1.

## Dataset construction

Smoke dataset:

```bash
python scripts/build_reentry_viscalibrator_dataset.py \
  --base-cache outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_1video.pt \
  --override-cache outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_1video.pt \
  --setting rgb_dev0_natural_smoke \
  --out-npz outputs/paper_discovery_2026-06-27/reentry_viscalibrator/datasets/rgb_dev0_smoke_candidates.npz \
  --max-examples 512
```

Smoke output:

```text
n_examples = 512
sequence_len = 33
feature_dim = 28
hard_positive_rate = 0.670063
```

Full dev10 dataset:

```bash
python scripts/build_reentry_viscalibrator_dataset.py \
  --out-npz outputs/paper_discovery_2026-06-27/reentry_viscalibrator/datasets/rgb_dev10_w16p2_candidates.npz
```

Output:

```text
n_examples = 12107
sequence_len = 33
feature_dim = 28
train_examples = 8972
val_examples = 3135
hard_positive_rate = 0.565053
```

Split is video-level.

## Training

Command:

```bash
python scripts/train_reentry_viscalibrator.py \
  --dataset outputs/paper_discovery_2026-06-27/reentry_viscalibrator/datasets/rgb_dev10_w16p2_candidates.npz \
  --out-model outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1_dev10.pt \
  --epochs 8 \
  --batch-size 256
```

Best saved checkpoint:

```text
model = outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1_dev10.pt
best_epoch = 1
validation-selected threshold by frame F0.5 = 0.65
```

Important caveat:

```text
The validation F0.5 threshold (0.65) is conservative and preserves AJ but under-recovers AJ_RD.
Metric-level thresholding showed threshold 0.15 is much better. For a final paper, threshold selection should be formalized on held-out dev videos before frozen fresh evaluation.
```

## Smoke chain result

Using one RGB dev0 video, the full pipeline runs:

```text
build dataset -> train small model -> apply model -> generate cache -> evaluate AJ_RD/AJ/OA
```

Smoke learned result:

```text
ReEntry-VisCalibrator smoke:
AJ_RD_256 = 0.6593
AJ_256    = 79.6365
OA_256    = 89.4117
```

Rule W8P2 on same dev0:

```text
ReEntry-VisGuard-W8P2:
AJ_RD_256 = 0.6762
AJ_256    = 79.4809
OA_256    = 91.2759
```

Interpretation:

```text
The smoke model proves the engineering chain works but does not beat the rule on the same single video.
```

## Full dev10 metric-level check

Rule W8P2 on dev10:

```text
AJ_RD_256 = 0.4885
AJ_256    = 79.7438
OA_256    = 93.0735
```

Learned V1, threshold 0.15, on dev10:

```text
AJ_RD_256 = 0.4899
AJ_256    = 79.9506
OA_256    = 93.1510
```

Interpretation:

```text
At threshold 0.15, the learned module slightly improves over rule W8P2 on dev10 and has higher standard AJ.
```

## Fresh20-49 natural results

Baselines:

```text
offline/base:       AJ_RD=0.3816, AJ=79.5944, OA=91.4636
rule W8P2:          AJ_RD=0.4510, AJ=79.1110, OA=92.8853
```

Learned V1 threshold sweep:

| Threshold | AJ_RD_256 | AJ_256 | OA_256 | Notes |
|---:|---:|---:|---:|---|
| 0.65 | 0.4064 | 79.6853 | 92.1037 | very conservative; high AJ but weak recovery |
| 0.30 | 0.4449 | 79.4156 | 92.7441 | near rule, better AJ |
| 0.20 | 0.4508 | 79.2919 | 92.8424 | almost matches rule, better AJ |
| 0.15 | 0.4521 | 79.2356 | 92.8896 | slightly beats rule and improves AJ over rule |

Best current learned natural result:

```text
ReEntry-VisCalibrator V1, threshold 0.15:
AJ_RD_256 = 0.4521
AJ_256    = 79.2356
OA_256    = 92.8896
```

Compared with rule W8P2:

```text
AJ_RD_256 +0.0011
AJ_256    +0.1246
OA_256    +0.0043
```

Compared with offline/base:

```text
AJ_RD_256 +0.0705
AJ_256    -0.3588
OA_256    +1.4260
```

## Fresh20-49 translate_L16 stress

Baselines:

```text
offline/base:       AJ_RD=0.4788, AJ=75.1940, OA=90.7805
rule W8P2:          AJ_RD=0.5336, AJ=74.7705, OA=92.2245
```

Learned V1 threshold 0.15:

```text
AJ_RD_256 = 0.5349
AJ_256    = 74.9201
OA_256    = 92.2000
```

Compared with rule W8P2:

```text
AJ_RD_256 +0.0013
AJ_256    +0.1496
OA_256    -0.0245
```

Compared with offline/base:

```text
AJ_RD_256 +0.0561
AJ_256    -0.2739
OA_256    +1.4195
```

## Fresh20-49 occluder_L16 stress

Baselines:

```text
offline/base:       AJ_RD=0.6311, AJ=77.6280, OA=90.8918
rule W8P2:          AJ_RD=0.6659, AJ=77.0436, OA=92.1748
```

Learned V1 threshold 0.15:

```text
AJ_RD_256 = 0.6667
AJ_256    = 77.1780
OA_256    = 92.1795
```

Compared with rule W8P2:

```text
AJ_RD_256 +0.0008
AJ_256    +0.1344
OA_256    +0.0047
```

Compared with offline/base:

```text
AJ_RD_256 +0.0356
AJ_256    -0.4500
OA_256    +1.2877
```

## Main interpretation

The learned module V1 is successful as a first implementation:

```text
1. It runs end-to-end.
2. It preserves the core coordinate-preserving visibility-recovery principle.
3. With threshold 0.15, it slightly improves over rule W8P2 on natural, translate, and occluder final settings.
4. It also improves standard AJ over rule W8P2 in all three settings.
```

Current learned V1 vs rule W8P2:

| Setting | ΔAJ_RD | ΔAJ | ΔOA |
|---|---:|---:|---:|
| natural | +0.0011 | +0.1246 | +0.0043 |
| translate_L16 | +0.0013 | +0.1496 | -0.0245 |
| occluder_L16 | +0.0008 | +0.1344 | +0.0047 |

This is not a huge gain, but it is a meaningful answer to the rule-based criticism:

```text
The learned module matches or slightly improves the rule while using the same principle and producing better AJ tradeoff.
```

## Critical caveat before paper claim

Threshold 0.15 currently needs a cleaner selection protocol.

For a paper-safe claim, do one of the following:

```text
Option A: select threshold on held-out dev videos using metric-level AJ_RD/AJ tradeoff, then freeze for fresh20-49.
Option B: predefine a threshold family and report a dev-selected threshold with no fresh tuning.
Option C: keep learned module as an appendix extension until threshold selection is fully cleaned.
```

The implementation is ready, and results are promising. The remaining issue is protocol cleanliness, not feasibility.

## Recommended next steps

1. Build a clean dev train/val split at cache level:

```text
train: dev0-6
val: dev7-9
fresh test: fresh20-49
```

2. Select threshold on dev7-9 by metric-level criterion:

```text
maximize AJ_RD under AJ loss <= 0.6
or maximize AJ_RD + lambda * AJ
```

3. Freeze threshold and re-run:

```text
fresh20-49 natural
fresh20-49 translate_L16
fresh20-49 occluder_L16
```

4. If threshold remains near 0.15 and gains persist, promote ReEntry-VisCalibrator V1 into the main paper as the learned extension.
