# TrackOn2 DINOv3 ReEntry-TAP Stress Smoke — 2026-07-02

## Environment status

TrackOn2 DINOv3 environment is now runnable in base:

```text
torch = 2.2.2+cu121
CUDA = 12.1 available
mmcv = 2.2.0 with ops
transformers = 4.56.1
huggingface_hub = 0.36.2
DINOv3 local dir = /gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
TrackOn2 Predictor import = OK
```

Important fix during setup:

```text
numpy was downgraded from 2.4.6 to 1.26.4 because torch.from_numpy failed under the previous NumPy/PyTorch combination.
```

## Local DINOv3 backbone

Used local DINOv3 backbone:

```text
/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

Confirmed files:

```text
config.json
model.safetensors
preprocessor_config.json
```

Loaded config:

```text
model_type = dinov3_vit
hidden_size = 384
```

## Export command shape

Both smokes used:

```text
model: TrackOn2 DINOv3
config: baselines/track_on/config/test.yaml
checkpoint: baselines/track_on/checkpoints_trackon2_dinov3.pt
support_grid_size: 20
max_videos: 1
max_queries: 256
```

Note from loader:

```text
Loaded model weights from baselines/track_on/checkpoints_trackon2_dinov3.pt
Info: missing (allowed) weights: 235 keys under {'backbone.vit_encoder.dinov3'}
```

This is consistent with loading the frozen DINOv3 backbone separately via HuggingFace local files while loading TrackOn2-specific weights from the checkpoint.

---

## Translate L16 dev0 256-query smoke

Output cache:

```text
outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_256q/trackon2_dinov3_translate_L16_dev0_256q.pt
```

Report:

```text
records = 1
queries = 256
export_sec = 24.667
vis_rate = 0.5647
peak_mem_mb = 880.2
```

Fair comparison uses the first 256 queries for all methods.

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.7240 | 77.3369 | 90.0728 | 87.5705 |
| CoTracker3 online | 0.7492 | 71.9517 | 88.3001 | 82.4510 |
| B2-W16-P2 | 0.7587 | 78.2747 | 93.8331 | 88.0356 |
| TrackOn2 DINOv3 | 0.5775 | 47.9390 | 70.3470 | 67.0239 |

Interpretation:

```text
TrackOn2 DINOv3 is runnable, but it is not competitive on this translate_L16 ReEntry-TAP smoke subset.
B2-W16-P2 is stronger than both offline and TrackOn2 on AJ_RD and AJ.
```

---

## Occluder L16 dev0 256-query smoke

Output cache:

```text
outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_256q/trackon2_dinov3_occluder_L16_dev0_256q.pt
```

Report:

```text
records = 1
queries = 256
export_sec = 16.132
vis_rate = 0.5363
peak_mem_mb = 880.2
```

Fair comparison uses the first 256 queries for all methods.

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.8061 | 81.1259 | 91.9867 | 89.9515 |
| CoTracker3 online | 0.7629 | 70.3340 | 86.7799 | 84.6946 |
| B2-W16-P2 | 0.8123 | 80.2050 | 93.6386 | 89.8992 |
| TrackOn2 DINOv3 | 0.6101 | 50.0152 | 70.2686 | 68.5814 |

Interpretation:

```text
TrackOn2 DINOv3 is runnable, but it is not competitive on this occluder_L16 ReEntry-TAP smoke subset.
B2-W16-P2 remains the strongest AJ_RD method on this smoke, while offline has slightly higher AJ than B2.
```

---

## Decision

TrackOn2 DINOv3 ReEntry-TAP stress baseline is now technically runnable, but the first dev0 256-query smokes are weak.

Current paper use:

```text
Do not add TrackOn2 DINOv3 ReEntry-TAP smoke as a strong main baseline.
It can be reported in appendix as an external baseline feasibility result if needed.
The parity-valid TrackOn2 first-input bridge remains the safer external plug-in supplement.
```

Recommended next actions:

```text
1. Do not immediately run full fresh20-49 TrackOn2 stress; first understand protocol mismatch.
2. If continuing TrackOn2, run a 1-video natural RGB smoke or parity bridge sanity check under the exact TrackOn2-native protocol.
3. For the current method paper, keep B2/ReEntry-Guard as the main story and TrackOn2 first-input as supplement.
```

---

## Visibility-variant diagnosis

After the full dev0 smokes, we audited whether TrackOn2's weak scores are mainly caused by visibility prediction or coordinate tracking.

### Translate L16 dev0 full

| Variant | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| original | 0.6132 | 29.6552 | 47.5918 | 39.5402 |
| all_visible | 0.5690 | 21.7065 | 81.2393 | 39.5402 |
| GT visibility | 0.7438 | 24.8630 | 100.0000 | 39.5402 |
| query_visible_fill | 0.6132 | 29.6552 | 47.5918 | 39.5402 |

### Occluder L16 dev0 full

| Variant | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| original | 0.6238 | 30.3484 | 49.0573 | 40.6263 |
| all_visible | 0.6241 | 22.1081 | 79.3996 | 40.6263 |
| GT visibility | 0.7634 | 25.7495 | 100.0000 | 40.6263 |
| query_visible_fill | 0.6238 | 30.3484 | 49.0573 | 40.6263 |

### Diagnosis

```text
GT visibility substantially raises AJ_RD, so visibility calibration is part of the issue.
However, delta_avg remains low at about 39.5--40.6 even with GT visibility, and AJ remains low.
Therefore the main protocol mismatch is not only visibility; coordinate tracking quality under this ReEntry-TAP stress adapter is also weak.
```

This explains why TrackOn2 DINOv3 is not ready to be used as a strong main ReEntry-TAP baseline even though the environment and model now run successfully.
