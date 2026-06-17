# MMP-Tracker Download Manifest

This file lists what is already available on the server, what is missing, and
what should be uploaded next to support the new matching-first tracker route.

## Server assets already available

### Weights already on server

- `/gemini/code/FSPT/weights/backbone/resnet50_a1_0-14fe96d1.pth`
- `/gemini/code/FSPT/weights/scaled_offline.pth`
- `/gemini/code/FSPT/weights/scaled_online.pth`
- `/gemini/code/FSPT/weights/tapir_checkpoint.npy`
- `/gemini/code/FSPT/weights/bootstapir_checkpoint.npy`
- `/gemini/code/FSPT/weights/clip/ViT-B-16.pt`

### Datasets already on server

- `/gemini/code/datasets/tapvid_kinetics`
- `/gemini/code/datasets/tapvid_davis`
- `/gemini/code/datasets/tapvid_kubric`
- `/gemini/code/datasets/kubric_preprocessed`
- `/gemini/code/datasets/tapvid_rgb_stacking`
- `/gemini/code/datasets/robotap`
- `/gemini/code/datasets/kinetics` (metadata / split files)

These are enough to build and validate the first engineering baseline.

## Priority downloads

## Priority 1: matching pretraining assets

These are the most important missing assets for the new project.

### Option A: datasets

- `MegaDepth`
  - target path: `/gemini/code/datasets/megadepth`
  - purpose: two-view matching warm start for local and global correspondence
- `ScanNet` or `ScanNet++`
  - target path: `/gemini/code/datasets/scannet`
  - purpose: indoor correspondence pretraining and geometric robustness

### Option B: pretrained matching weights

If full datasets are inconvenient, upload a correspondence-oriented checkpoint.

- target path: `/gemini/code/FSPT/weights/matching/`
- recommended naming:
  - `/gemini/code/FSPT/weights/matching/megadepth_pretrain.pth`
  - `/gemini/code/FSPT/weights/matching/scannet_pretrain.pth`

## Priority 2: real-video training assets

The next key unlock is stronger real-video training or pseudo-label data.

### Raw videos

- `Kinetics` raw subset
  - target path: `/gemini/code/datasets/kinetics_raw_subset`
- `YouTube-VOS`
  - target path: `/gemini/code/datasets/youtube_vos`

### Teacher pseudo labels

- Kinetics teacher tracks
  - target path: `/gemini/code/datasets/pseudo_tracks/kinetics_teacher_tracks`
- YouTube-VOS teacher tracks
  - target path: `/gemini/code/datasets/pseudo_tracks/youtubevos_teacher_tracks`

Recommended payload per sample:

- track coordinates
- visibility
- confidence
- query frame index or query point definition
- frame size metadata

## Priority 3: paper-strength hard-case evaluation sets

- `RoboTAP`
  - target path: `/gemini/code/datasets/robotap`
  - purpose: stronger evidence for hard tracking scenarios and cross-domain story

## Recommended minimum upload plan

If you only want to upload the highest-value assets first, do this in order:

1. `MegaDepth`
   - `/gemini/code/datasets/megadepth`
2. `YouTube-VOS` or a real-video subset with enough diversity
   - `/gemini/code/datasets/youtube_vos`

This is the highest-value minimum package for the next phase.

## Upload notes

- Keep folder names stable because future configs will reference them directly.
- If the dataset is extremely large, start with a curated subset and preserve the
  same directory layout that the full dataset would use.
- If you upload weights instead of datasets, include a short text file next to the
  checkpoint describing:
  - source project
  - training data
  - input resolution
  - feature dimension
