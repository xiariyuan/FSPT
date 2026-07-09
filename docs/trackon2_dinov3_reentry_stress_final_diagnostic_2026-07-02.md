# TrackOn2 DINOv3 ReEntry-TAP Final Diagnostic — 2026-07-02

## Status

TrackOn2 DINOv3 is now technically runnable under the current FSPT environment.

Environment used:

```text
torch = 2.2.2+cu121
CUDA = 12.1 available
mmcv = 2.2.0 with ops
transformers = 4.56.1
huggingface_hub = 0.36.2
numpy = 1.26.4
DINOv3 local dir = /gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
TrackOn2 checkpoint = baselines/track_on/checkpoints_trackon2_dinov3.pt
TrackOn2 config = baselines/track_on/config/test.yaml
```

DINOv3 config was recognized:

```text
model_type = dinov3_vit
hidden_size = 384
```

## Experiments completed

Completed the following TrackOn2 DINOv3 ReEntry-TAP smoke and diagnostic experiments:

```text
1. translate_L16 dev0 256-query smoke
2. occluder_L16 dev0 256-query smoke
3. translate_L16 dev0 full-query smoke
4. occluder_L16 dev0 full-query smoke
5. visibility variants: original / all-visible / GT-visibility / query-visible-fill
6. geometry audit: anchor error, prediction range, coordinate error
7. support-grid and memory-policy variants on 256-query smoke
```

---

## 1. Full-query ReEntry-TAP smoke results

### translate_L16 dev0 full

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | queries |
|---|---:|---:|---:|---:|---:|
| CoTracker3 offline | 0.6667 | 76.0920 | 88.5175 | 87.0434 | 1116 |
| CoTracker3 online | 0.7061 | 46.4554 | 61.2806 | 67.9842 | 1116 |
| B2-W16-P2 | 0.7239 | 77.2732 | 90.6263 | 87.4162 | 1116 |
| TrackOn2 DINOv3 | 0.6132 | 29.6552 | 47.5918 | 39.5402 | 1116 |

### occluder_L16 dev0 full

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | queries |
|---|---:|---:|---:|---:|---:|
| CoTracker3 offline | 0.7348 | 77.4598 | 89.2049 | 88.8365 | 1081 |
| CoTracker3 online | 0.7093 | 45.0745 | 61.5413 | 71.4378 | 1081 |
| B2-W16-P2 | 0.7619 | 77.9665 | 90.7270 | 88.8248 | 1081 |
| TrackOn2 DINOv3 | 0.6238 | 30.3484 | 49.0573 | 40.6263 | 1081 |

### Full-query conclusion

```text
TrackOn2 DINOv3 is runnable but not competitive under the current ReEntry-TAP stress adapter.
B2-W16-P2 remains clearly stronger on both AJ_RD and standard AJ/OA.
```

---

## 2. 256-query sanity smoke

### translate_L16 first 256 queries

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| CoTracker3 offline | 0.7240 | 77.3369 | 90.0728 |
| CoTracker3 online | 0.7492 | 71.9517 | 88.3001 |
| B2-W16-P2 | 0.7587 | 78.2747 | 93.8331 |
| TrackOn2 DINOv3 | 0.5775 | 47.9390 | 70.3470 |

### occluder_L16 first 256 queries

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| CoTracker3 offline | 0.8061 | 81.1259 | 91.9867 |
| CoTracker3 online | 0.7629 | 70.3340 | 86.7799 |
| B2-W16-P2 | 0.8123 | 80.2050 | 93.6386 |
| TrackOn2 DINOv3 | 0.6101 | 50.0152 | 70.2686 |

### 256-query conclusion

```text
The weak full-query result is not caused by a bad long-tail subset; TrackOn2 is already weak on the first 256-query smoke.
```

---

## 3. Visibility variant diagnostic

### translate_L16 dev0 full

| Variant | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| original | 0.6132 | 29.6552 | 47.5918 | 39.5402 |
| all_visible | 0.5690 | 21.7065 | 81.2393 | 39.5402 |
| GT_visibility | 0.7438 | 24.8630 | 100.0000 | 39.5402 |
| query_visible_fill | 0.6132 | 29.6552 | 47.5918 | 39.5402 |

### occluder_L16 dev0 full

| Variant | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| original | 0.6238 | 30.3484 | 49.0573 | 40.6263 |
| all_visible | 0.6241 | 22.1081 | 79.3996 | 40.6263 |
| GT_visibility | 0.7634 | 25.7495 | 100.0000 | 40.6263 |
| query_visible_fill | 0.6238 | 30.3484 | 49.0573 | 40.6263 |

### Visibility conclusion

```text
GT visibility can raise AJ_RD, which means TrackOn2 visibility is a major weakness under this adapter.
However, delta_avg remains around 40, so coordinate quality is also weak.
This is not merely a visibility-threshold problem.
```

---

## 4. Geometry / coordinate audit

### TrackOn2 DINOv3 translate_L16 full

```text
query-frame anchor error median = 0.187 px
query-frame anchor error p95    = 5.820 px
predictions in range ±0.05      = 100%
predicted visibility rate       = 0.303
GT visibility rate              = 0.813
median coordinate error, all frames        = 61.36 px
median coordinate error, GT-visible frames = 44.54 px
```

### TrackOn2 DINOv3 occluder_L16 full

```text
query-frame anchor error median = 0.192 px
query-frame anchor error p95    = 5.804 px
predictions in range ±0.05      = 100%
predicted visibility rate       = 0.297
GT visibility rate              = not shown in summary, but much higher than predicted visibility
median coordinate error, all frames        = 62.05 px
median coordinate error, GT-visible frames = 44.10 px
```

### B2 reference geometry

For comparison, B2 has near-perfect query anchor and much lower coordinate error:

```text
B2 translate_L16:
anchor error median ≈ 0.000004 px
median coordinate error, GT-visible frames ≈ 0.55 px

B2 occluder_L16:
anchor error median ≈ 0.000004 px
median coordinate error, GT-visible frames ≈ 0.49 px
```

### Geometry conclusion

```text
The TrackOn2 adapter is not obviously broken at query initialization: query-frame anchor error is small.
Predictions stay in range, so this is not a simple normalization bug.
The main problem is poor subsequent tracking / coordinate accuracy plus low predicted visibility.
```

---

## 5. support_grid / memory-policy variants, 256-query smoke

### translate_L16 first 256 queries

| Variant | support_grid_size | memory policy | AJ_RD_256 | AJ_256 | OA_256 | visibility rate |
|---|---:|---|---:|---:|---:|---:|
| original baseline | 20 | unconditional | 0.5775 | 47.9390 | 70.3470 | 0.5647 |
| sg0_uncond | 0 | unconditional | 0.5354 | 48.4381 | 71.3087 | 0.5827 |
| sg20_vismem | 20 | visibility_selective | 0.4506 | 43.5466 | 64.5473 | 0.5250 |
| sg0_vismem | 0 | visibility_selective | 0.4265 | 42.2509 | 64.5316 | 0.5269 |

### occluder_L16 first 256 queries

| Variant | support_grid_size | memory policy | AJ_RD_256 | AJ_256 | OA_256 | visibility rate |
|---|---:|---|---:|---:|---:|---:|
| original baseline | 20 | unconditional | 0.6101 | 50.0152 | 70.2686 | 0.5363 |
| sg0_uncond | 0 | unconditional | 0.5997 | 50.2632 | 70.3627 | 0.5479 |
| sg20_vismem | 20 | visibility_selective | 0.5789 | 49.9694 | 67.3224 | 0.5218 |
| sg0_vismem | 0 | visibility_selective | 0.5723 | 49.5919 | 67.1828 | 0.5273 |

### Variant conclusion

```text
Removing support grid does not fix the weakness.
Visibility-selective memory update hurts AJ_RD and AJ in this setup.
The original TrackOn2 configuration is already the best among tested runtime variants, but still far below B2.
```

---

## Final decision

### What we can claim

```text
TrackOn2 DINOv3 has now been made runnable under the ReEntry-TAP stress pipeline.
A full one-video smoke and multiple diagnostic variants were completed.
```

### What we should not claim

```text
Do not claim TrackOn2 DINOv3 is a strong ReEntry-TAP stress baseline here.
Do not put TrackOn2 DINOv3 stress numbers in the main table as a competitive SOTA comparison.
```

### Best paper usage

```text
Use B2-W16-P2 and ReEntry-Guard as the main method results.
Use TrackOn2 first-input bridge as the safer external plug-in supplement.
Optionally mention TrackOn2 DINOv3 ReEntry-TAP stress as appendix feasibility: runnable but weak under this adapter/protocol.
```

### Recommended stop condition

```text
Stop expanding TrackOn2 ReEntry-TAP stress to fresh20-49 for now.
The dev0 full-query smoke and diagnostics are already enough to show that scaling this external baseline would not strengthen the paper.
If continuing TrackOn2 later, first reproduce TrackOn2's native/parity protocol again, then derive an aligned ReEntry-TAP adapter from that protocol.
```
