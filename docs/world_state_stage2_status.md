# Stage 2 Status

## Current State

Stage 2 has started with a minimal learnable world-state refiner:

- Dataset: `experiments/world_state_dataset.py`
- Model: `experiments/world_state_model.py`
- Train script: `scripts/train_world_state_stage2.py`

This prototype takes:

- noisy query-frame depth
- query/re-entry 2D coordinates
- occlusion length
- camera motion
- lifted noisy 3D world point

and predicts:

- a refined 3D world state
- a visibility logit

## Losses

The current prototype optimizes:

- `delta_world` regression
- refined world-state regression
- re-entry reprojection loss
- visibility BCE

## Smoke Result

Command:

`python scripts/train_world_state_stage2.py --data-root /gemini/code/FSPT/datasets/pointodyssey --train-splits train --val-splits val --train-max-sequences 1 --val-max-sequences 1 --train-max-samples 8192 --val-max-samples 2048 --epochs 1 --batch-size 256 --num-workers 2 --output-dir /gemini/code/FSPT/outputs/world_state_stage2_smoke`

Result:

- `reproj_median_px = 17.21`
- `reproj_lt4px = 7.57%`
- `world_l2_mean = 0.555`
- `vis_acc = 1.0`

Artifacts:

- `outputs/world_state_stage2_smoke/metrics.json`
- `outputs/world_state_stage2_smoke/last.pt`

## Next Step

Scale this prototype to:

1. `train_max_sequences=8~16`
2. `epochs=5~10`
3. evaluate directly against Stage 1 `Pred3D@0.10 / 0.15`
4. add temporal multi-frame context instead of single query/re-entry pair features
5. replace GT query depth noise proxy with actual predicted depth maps
