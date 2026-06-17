# Fallback Decision (2026-06-15)

## Decision: No Fallback Triggered

The DINOv3 backbone route **passed** successfully. No fallback to Track-On2 DINOv2 is needed.

## Evidence

### DINOv3 Local Load (PASS)
- `AutoConfig.from_pretrained(...)` → `dinov3_vit` config loaded
- `AutoModel.from_pretrained(...)` → 28.7M parameter model loaded
- Local path: `/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m`
- No HF network access required
- `outputs/dinov3_load_smoke.json`

### Track-On2 DINOv3 Adapter (PASS)
- `DINOV3_LOCAL_DIR` env var patch in `dinov3_vit_adapter.py` works
- `Predictor` builds and runs forward with local DINOv3 backbone
- 235 missing checkpoint keys are expected (README: backbone weights not included in checkpoint)
- `outputs/trackon_local_backbone_smoke.txt`

### Track-On2 DINOv3 Repo-Native DAVIS (PASS)
- AJ=67.04, δ_avg=79.84, OA=92.09
- README reference: δ_avg=79.9
- Δ = -0.06 (match within noise)

## What Would Have Triggered Fallback

Per `docs/claude_handoff_recovery_execution_checklist_2026-06-14.md`:

1. `AutoConfig`/`AutoModel` unable to load DINOv3 locally → **not triggered**
2. `track_on` import/build failure → **not triggered**
3. `mmcv` install failure within timebox → **not triggered**
4. `Predictor` forward depends on HF or fails → **not triggered**

## Note on Track-On-R

Track-On-R is blocked by **missing checkpoints** (`track_on_r.pt`, `verifier.pt`), NOT by DINOv3 backbone issues. The DINOv3 backbone is loaded separately and is working. Track-On-R failure is a checkpoint procurement issue, not a backbone route failure.

## Conclusion

DINOv3 route: **PASS**. Proceed with Track-On2 DINOv3 as primary baseline. No need to switch to Track-On2 DINOv2.
