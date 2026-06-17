# External Baseline Asset Audit (2026-06-15)

## Summary

| Baseline | Local Code | Local Checkpoint | Status |
|----------|-----------|------------------|--------|
| CoTracker3 Offline | `baselines/cotracker/` ✅ | `checkpoints/scaled_offline.pth` ✅ | **READY** |
| CoTracker3 Online | `baselines/cotracker/` ✅ | `checkpoints/scaled_online.pth` ✅ | **READY** |
| Track-On2 DINOv3 | `baselines/track_on/` ✅ | `checkpoints_trackon2_dinov3.pt` ✅ | **READY** |
| Track-On2 DINOv2 | `baselines/track_on/` ✅ | `checkpoints_trackon2_dinov2.pt` ✅ | **READY** |
| Track-On-R | `baselines/track_on/` ✅ | ❌ `track_on_r.pt` MISSING | **MISSING_CKPT** |
| Verifier (for Track-On-R) | `baselines/track_on/verifier/` ✅ | ❌ `verifier.pt` MISSING | **MISSING_CKPT** |
| TAPIR | `baselines/track_on/ensemble/bootstapir/` ✅ | `weights/tapir_checkpoint.npy` ⚠️ | needs adapter to `bootstapir_predictor` |
| BootsTAPIR | `baselines/track_on/ensemble/bootstapir/` ✅ | `weights/bootstapir_checkpoint.npy` ⚠️ | needs adapter to `bootstapir_predictor` |
| TAPNext++ | `baselines/track_on/ensemble/tapnext/` ✅ | ❌ No tapnext checkpoint found | **MISSING_CKPT** |
| BootsTAPNext | `baselines/track_on/ensemble/tapnext/` ✅ | ❌ No bootstapnext checkpoint found | **MISSING_CKPT** |
| AllTracker | `baselines/track_on/ensemble/alltracker/` ✅ | ❌ No alltracker checkpoint found | **MISSING_CKPT** |
| LocoTrack | `baselines/track_on/ensemble/locotrack/` ✅ | `baselines/locotrack/` ✅ | needs adapter verification |

## Details

### CoTracker3 (local)
- **Repo:** `/gemini/code/FSPT/baselines/cotracker/`
- **Offline CKPT:** `checkpoints/scaled_offline.pth` (512KB)
- **Online CKPT:** `checkpoints/scaled_online.pth` (512KB)
- **Load method:** `CoTrackerPredictor(checkpoint=path)` directly — no torch.hub needed
- **Eval:** ✅ DAVIS completed this round

### TAPIR / BootsTAPIR
- **Wrappers:** `baselines/track_on/ensemble/bootstapir/`
- **Checkpoints:** `weights/tapir_checkpoint.npy`, `weights/bootstapir_checkpoint.npy`
- **Status:** Code and checkpoints both exist, but no adapter verification done
- **Recommended next step:** Test `TAPIRPredictor(path)` initialization
- **Missing checkpoints for TAPNext++:** `tapnext_ckpt.npz`, `bootstapnext_ckpt.npz` — these are separate models from the TAPIR/BootsTAPIR checkpoints

### Track-On-R
- **Missing:** `track_on_r.pt` (hosted at `https://huggingface.co/gorkaydemir/track_on_r`)
- **Missing:** `verifier.pt` (same repo)
- **Network needed for download:** Yes
- **Note:** Only needed if comparing Track-On-R (stronger variant with verifier-guided fine-tuning)

### TAPNext++ / BootsTAPNext / AllTracker
- **Wrappers exist in:** `baselines/track_on/ensemble/`
- **Checkpoints not found** in `weights/` or `baselines/`
- **Need download from HF or official sources**
