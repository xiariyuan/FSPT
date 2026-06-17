# Track-On DINOv3 Local Backbone Patch Notes

**Date:** 2026-06-14  
**Purpose:** Enable track_on to load DINOv3 from local ModelScope download instead of HuggingFace

## Changes Made

### File: `model/vit_adapter/dinov3_adapter/dinov3_vit_adapter.py`

**Before (line 34-37):**
```python
# NOTE: If you are working in an environment without internet access, 
#       make sure to download the pretrained DINOv3 weights to cache (eg with a call on login node)
#       and set `local_files_only=True` in the line below for your trainings.
self.dinov3 = AutoModel.from_pretrained(pretrained_model_name, device_map=None, local_files_only=False)
```

**After:**
```python
import os
# NOTE: If you are working in an environment without internet access,
#       make sure to download the pretrained DINOv3 weights to cache (eg with a call on login node)
#       and set `local_files_only=True` in the line below for your trainings.
dinov3_local_dir = os.environ.get("DINOV3_LOCAL_DIR", "").strip()
if dinov3_local_dir and os.path.isdir(dinov3_local_dir):
    dinov3_source = dinov3_local_dir
    local_only = True
else:
    dinov3_source = pretrained_model_name
    local_only = False
self.dinov3 = AutoModel.from_pretrained(dinov3_source, device_map=None, local_files_only=local_only)
```

## Behavior

- If `DINOV3_LOCAL_DIR` env var is set and points to a directory, use it with `local_files_only=True`
- Otherwise, fall back to original HuggingFace repo ID behavior
- No changes to training logic, memory handling, or inference code

## Environment Variable

```bash
export DINOV3_LOCAL_DIR=/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

## Dependencies

- transformers >= 4.56 (upgraded from 4.40.2 to 5.12.0)
- PyTorch >= 2.4 (upgrading from 2.1.2)
