# TrackOn2 DINOv2 Backbone Download Instructions — 2026-07-01

## Recommended backbone

Use the DINOv2 route first:

```text
TrackOn2 checkpoint: baselines/track_on/checkpoints_trackon2_dinov2.pt
TrackOn2 config:     baselines/track_on/config/test_dinov2.yaml
HF backbone:         facebook/dinov2-base
```

The code reads the local DINOv2 backbone from:

```bash
export TRACKON_DINOV2_MODEL_PATH=/path/to/facebook/dinov2-base
```

## Recommended local path in this repo

```text
/gemini/code/FSPT/checkpoints/hf/facebook_dinov2_base
```

Set:

```bash
export TRACKON_DINOV2_MODEL_PATH=/gemini/code/FSPT/checkpoints/hf/facebook_dinov2_base
```

## Download method A: Hugging Face CLI

On a machine/server with internet:

```bash
pip install -U huggingface_hub
mkdir -p /gemini/code/FSPT/checkpoints/hf/facebook_dinov2_base
hf download facebook/dinov2-base \
  --local-dir /gemini/code/FSPT/checkpoints/hf/facebook_dinov2_base
```

## Download method B: Python snapshot_download

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/dinov2-base",
    local_dir="/gemini/code/FSPT/checkpoints/hf/facebook_dinov2_base",
    local_dir_use_symlinks=False,
)
PY
```

## Download method C: git-lfs

```bash
mkdir -p /gemini/code/FSPT/checkpoints/hf
cd /gemini/code/FSPT/checkpoints/hf
git lfs install
git clone https://huggingface.co/facebook/dinov2-base facebook_dinov2_base
```

## Verify after download

```bash
cd /gemini/code/FSPT
export TRACKON_DINOV2_MODEL_PATH=/gemini/code/FSPT/checkpoints/hf/facebook_dinov2_base
python scripts/check_trackon_dino_backbone.py
```

Success requires:

```text
config.json exists
model.safetensors or pytorch_model.bin exists
```

## Run TrackOn2 DINOv2 smoke after verification

```bash
cd /gemini/code/FSPT
export TRACKON_DINOV2_MODEL_PATH=/gemini/code/FSPT/checkpoints/hf/facebook_dinov2_base

python scripts/export_trackon2_reentry_stress_cache.py \
  --config baselines/track_on/config/test_dinov2.yaml \
  --checkpoint baselines/track_on/checkpoints_trackon2_dinov2.pt \
  --stress-dataset outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/stress_dataset.pt \
  --out-cache outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov2_translate_L16_dev0_256q/trackon2_dinov2_translate_L16_dev0_256q.pt \
  --out-report outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov2_translate_L16_dev0_256q/trackon2_dinov2_translate_L16_dev0_256q_report.json \
  --max-videos 1 \
  --max-queries 256 \
  --support-grid-size 20
```
