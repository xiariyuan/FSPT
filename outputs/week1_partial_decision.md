# Track-On Family — Week 1 Partial Decision

**Date:** 2026-06-15

## Decision

**provisional go Track-On family**

## Evidence

### Track-On2 DINOv3

| Benchmark | Repo-native δ_avg | README δ_avg | Δ |
|-----------|-------------------|--------------|---|
| DAVIS (30 clips) | 79.84 | 79.9 | -0.06 |
| Kinetics (10 clips) | 68.83 | 69.3 | -0.47 |

Both benchmarks show metrics within expected variance of README reference values. No protocol/adapter gap detected.

### Track-On-R

- **Checkpoint NOT available locally** (`track_on_r.pt` not found)
- **Verifier checkpoint NOT available locally** (`verifier.pt` not found)
- Both hosted on HuggingFace: `gorkaydemir/track_on_r`
- Cannot evaluate Track-On-R without downloading checkpoints
- **Recommendation:** Attempt HF download in Phase E, or proceed with Track-On2 only

### Track-On2 DINOv2

- Checkpoint available: `checkpoints_trackon2_dinov2.pt` (89.6 MB)
- DINOv2 backbone: `baselines/hf/facebook_dinov2_small/` and `weights/dinov2/dinov2_vits14_pretrain.pth`
- Config: `config/test_dinov2.yaml` (vit_backbone: dinov2_b, vit_upsample_factor: 1.0)
- Available as fallback if DINOv3 route is dropped

## What was NOT done this substage

1. Full Kinetics eval (only 10-clip smoke)
2. Track-On-R eval (no checkpoint)
3. Attempt 0 unified rescoring (will do in Phase E)
4. Other baselines (CoTracker3, TAPNext++, AllTracker — Phase E)

## Recommendation for Phase E

1. Proceed with Attempt 0 initialization
2. Use `scripts/init_attempt0_run.py`
3. Track-On2 DINOv3 is the primary Track-On baseline
4. Attempt Track-On-R checkpoint download from HF if network available
5. CoTracker3 as the frozen anchor baseline
