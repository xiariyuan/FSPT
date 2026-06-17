# Track-On Environment Manifest
# Generated: 2026-06-15

## Current Active Environment

- **Python**: 3.11.8
- **Python path**: `/root/miniconda3/bin/python3`
- **Conda env**: `base` (root)
- **CUDA**: 12.1
- **GPU**: B1.gpu.medium (compute capability 8.6)

## Package Versions

| Package | Version | Status |
|---|---|---|
| torch | 2.4.1+cu121 | ✅ |
| torchvision | 0.19.1+cu121 | ✅ |
| torchaudio | 2.4.1+cu121 | ✅ |
| transformers | 4.57.6 | ✅ |
| mmcv | 2.2.0 | ✅ |
| mediapy | 1.2.6 | ✅ |

## vs Previous State (2026-06-14)

| Package | Before (2026-06-14) | After (2026-06-15) | Change |
|---|---|---|---|
| torch | 2.1.2+cu121 | 2.4.1+cu121 | ✅ upgraded |
| transformers | 4.40.2 | 4.57.6 | ✅ upgraded |
| mmcv | missing | 2.2.0 | ✅ installed |
| mediapy | missing | 1.2.6 | ✅ installed |

## DINOv3 Local Load Status

| Test | Result |
|---|---|
| AutoConfig.from_pretrained (local, trust_remote_code=True) | ✅ config OK: dinov3_vit |
| AutoModel.from_pretrained (local, trust_remote_code=True) | ✅ model OK: 28.7M params |
| No HF network access needed | ✅ local_files_only=True works |

## Track-On Import Status

```bash
cd baselines/track_on && python -c "from model.trackon_predictor import Predictor"
```
→ ✅ Passes (tested in earlier session, mmcv/mmcv-full available)

## Track-On2 DINOv3 Backbone Smoke

- Adapter: `baselines/track_on/model/vit_adapter/dinov3_adapter/dinov3_vit_adapter.py`
- Local backbone: `/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m`
- Patch: `DINOV3_LOCAL_DIR` env var
- Checkpoint: `baselines/track_on/checkpoints_trackon2_dinov3.pt`
- Result: ✅ Adapter smoke passes, 235 expected missing backbone keys

## Environment Satisfies Track-On Requirements

Per `baselines/track_on/README.md` dependencies:
- torch ✅
- torchvision ✅
- transformers ✅
- mmcv ✅
- mediapy ✅

All required packages are present and at compatible versions.