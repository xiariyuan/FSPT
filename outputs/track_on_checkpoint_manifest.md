# Track-On Checkpoint Manifest

**Date:** 2026-06-14

## Repository

- **Path:** `/gemini/code/FSPT/baselines/track_on`

## Configs

| Config | Path | Status |
|--------|------|--------|
| test.yaml | `config/test.yaml` | ✅ EXISTS |
| test_dinov2.yaml | `config/test_dinov2.yaml` | ✅ EXISTS |

## Checkpoints

| Checkpoint | Path | Status | Size (MB) | SHA256 prefix |
|------------|------|--------|-----------|---------------|
| Track-On2 DINOv3 | `checkpoints_trackon2_dinov3.pt` | ✅ EXISTS | 89.6 | `c17faf2068026f7b` |
| Track-On2 DINOv2 | `checkpoints_trackon2_dinov2.pt` | ✅ EXISTS | 89.6 | `acf76fc646554886` |
| Track-On-R | `track_on_r.pt` | ❌ NOT FOUND | - | - |
| Verifier | `verifier.pt` | ❌ NOT FOUND | - | - |

## Backbone Weights

### DINOv3 (Local)

- **Path:** `/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m`
- **Size:** 109.5 MB
- **Source:** ModelScope
- **Files:** config.json, model.safetensors, preprocessor_config.json, README.md, LICENSE.md
- **Load method:** `AutoModel.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)` via `DINOV3_LOCAL_DIR` env var

### DINOv2 (Local)

- **Path 1:** `/gemini/code/FSPT/baselines/hf/facebook_dinov2_small/` (HF format)
- **Path 2:** `/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth` (84.2 MB)
- **Load method:** `TRACKON_DINOV2_MODEL_PATH` env var

## Missing Assets

1. **`track_on_r.pt`** — Not found locally. Checkpoint is hosted on HuggingFace: https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt
   - Needs download or internet access
2. **`verifier.pt`** — Not found locally. Checkpoint is hosted on HuggingFace: https://huggingface.co/gorkaydemir/track_on_r/resolve/main/verifier.pt
   - Needs download or internet access

## Current Execution Plan

1. **Track-On2 DINOv3** → proceed immediately with DAVIS + Kinetics eval
2. **Track-On-R** → cannot proceed without checkpoint (will attempt download)
3. **Fallback** → Track-On2 DINOv2 also available if needed
