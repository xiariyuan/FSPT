# TrackOn2 DINOv3 Adapter Sanity and Ablation — 2026-07-02

## Executive decision

TrackOn2 DINOv3 is now technically runnable for ReEntry-TAP stress evaluation, but the current ReEntry-TAP stress results are weak and should **not** be used as a strong main-table external baseline.

The adapter is not obviously broken at the query-frame / coordinate level:

```text
query-frame median error ≈ 0.18--0.19 px
query-frame predicted visibility ≈ 0.94--0.97
out-of-bound rate is near zero
```

However, TrackOn2 has low predicted visibility over full videos and low coordinate accuracy across all visible frames:

```text
translate full pred visibility rate = 0.3032 vs GT visibility 0.8131
occluder full pred visibility rate = 0.2974 vs GT visibility 0.7948
translate full delta_avg_256 = 39.5402
occluder full delta_avg_256 = 40.6263
```

Lowering visibility threshold, using visibility-selective memory, or changing input scale did not rescue the result.

---

## Environment

Final runnable base environment:

```text
torch = 2.2.2+cu121
CUDA = 12.1 available
mmcv = 2.2.0 ops OK
transformers = 4.56.1
huggingface_hub = 0.36.2
numpy = 1.26.4
DINOv3 local dir = /gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
TrackOn2 Predictor import = OK
```

Important environment notes:

```text
1. transformers 4.47.1 did not recognize model_type=dinov3_vit.
2. transformers 4.56.1 fixed DINOv3 config/model loading.
3. numpy 2.4.6 caused torch.from_numpy failure under torch 2.2.2; downgrading to numpy 1.26.4 fixed it.
```

---

## Main dev0 full-query smoke

### Translate L16 dev0 full queries

```text
queries = 1116
TrackOn2 export_sec = 18.746
TrackOn2 vis_rate = 0.3032
TrackOn2 peak_mem_mb = 1432.5
```

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.6667 | 76.0920 | 88.5175 | 87.0434 |
| CoTracker3 online | 0.7061 | 46.4554 | 61.2806 | 67.9842 |
| B2-W16-P2 | 0.7239 | 77.2732 | 90.6263 | 87.4162 |
| TrackOn2 DINOv3 | 0.6132 | 29.6552 | 47.5918 | 39.5402 |

### Occluder L16 dev0 full queries

```text
queries = 1081
```

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.7348 | 77.4598 | 89.2049 | 88.8365 |
| CoTracker3 online | 0.7093 | 45.0745 | 61.5413 | 71.4378 |
| B2-W16-P2 | 0.7619 | 77.9665 | 90.7270 | 88.8248 |
| TrackOn2 DINOv3 | 0.6238 | 30.3484 | 49.0573 | 40.6263 |

---

## 256-query smoke results

### Translate L16 dev0 first 256 queries

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.7240 | 77.3369 | 90.0728 | 87.5705 |
| CoTracker3 online | 0.7492 | 71.9517 | 88.3001 | 82.4510 |
| B2-W16-P2 | 0.7587 | 78.2747 | 93.8331 | 88.0356 |
| TrackOn2 DINOv3 | 0.5775 | 47.9390 | 70.3470 | 67.0239 |

### Occluder L16 dev0 first 256 queries

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.8061 | 81.1259 | 91.9867 | 89.9515 |
| CoTracker3 online | 0.7629 | 70.3340 | 86.7799 | 84.6946 |
| B2-W16-P2 | 0.8123 | 80.2050 | 93.6386 | 89.8992 |
| TrackOn2 DINOv3 | 0.6101 | 50.0152 | 70.2686 | 68.5814 |

---

## Adapter sanity

### Query-frame sanity

| Cache | queries | query-frame median err px | query-frame mean err px | query-frame pred visible | pred vis rate | GT vis rate | OOB rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| translate 256 | 256 | 0.1828 | 0.9135 | 0.9727 | 0.5647 | 0.8232 | 0.010031 |
| occluder 256 | 256 | 0.1884 | 1.0514 | 0.9727 | 0.5363 | 0.8037 | 0.000078 |
| translate full | 1116 | 0.1872 | 1.0141 | 0.9418 | 0.3032 | 0.8131 | 0.002391 |
| occluder full | 1081 | 0.1916 | 0.9958 | 0.9473 | 0.2974 | 0.7948 | 0.000100 |

Interpretation:

```text
The adapter is query-frame aligned. Coordinate convention and resize mapping are probably not the primary cause of low scores.
```

### First re-entry sanity

| Cache | re-entry queries | pred visible at re-entry | median err px | mean err px | <4px |
|---|---:|---:|---:|---:|---:|
| translate 256 | 208 | 0.4663 | 1.3004 | 5.9233 | 0.6587 |
| occluder 256 | 256 | 0.6719 | 1.1840 | 4.4480 | 0.7031 |
| translate full | 457 | 0.3742 | 1.1312 | 5.4762 | 0.7046 |
| occluder full | 534 | 0.5206 | 1.0636 | 3.8691 | 0.7378 |

Interpretation:

```text
At first re-entry, coordinates are often near the target, but predicted visibility is too conservative and long-horizon visible-frame coordinate accuracy is low.
```

---

## Visibility-variant analysis

### Translate L16 dev0 full

| Variant | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| original | 0.6132 | 29.6552 | 47.5918 | 39.5402 |
| GT visibility | 0.7438 | 24.8630 | 100.0000 | 39.5402 |
| all visible | 0.5690 | 21.7065 | 81.2393 | 39.5402 |
| query visible fill | 0.6132 | 29.6552 | 47.5918 | 39.5402 |

### Occluder L16 dev0 full

| Variant | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| original | 0.6238 | 30.3484 | 49.0573 | 40.6263 |
| GT visibility | 0.7634 | 25.7495 | 100.0000 | 40.6263 |
| all visible | 0.6241 | 22.1081 | 79.3996 | 40.6263 |
| query visible fill | 0.6238 | 30.3484 | 49.0573 | 40.6263 |

Interpretation:

```text
Oracle visibility raises AJ_RD, but AJ remains low because delta_avg / coordinate quality is low across visible frames. Thus visibility is not the only issue.
```

---

## Delta-v sweep: translate L16 dev0 256 queries

| delta_v | mean pred vis rate | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.6669 | 0.5814 | 52.0857 | 76.9061 | 67.0239 |
| 0.6 | 0.6338 | 0.5842 | 51.3592 | 75.1977 | 67.0239 |
| 0.7 | 0.6025 | 0.5852 | 49.8951 | 73.2383 | 67.0239 |
| 0.8 | 0.5647 | 0.5775 | 47.9390 | 70.3470 | 67.0239 |

Interpretation:

```text
Lowering visibility threshold improves AJ/OA slightly but does not make TrackOn2 competitive. Coordinate accuracy is unchanged.
```

---

## Memory policy ablation: visibility_selective

### Translate L16 dev0 256 queries

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| B2-W16-P2 | 0.7587 | 78.2747 | 93.8331 | 88.0356 |
| TrackOn2 visibility_selective | 0.4506 | 43.5466 | 64.5473 | 58.7297 |
| TrackOn2 original unconditional | 0.5775 | 47.9390 | 70.3470 | 67.0239 |

### Occluder L16 dev0 256 queries

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|
| B2-W16-P2 | 0.8123 | 80.2050 | 93.6386 | 89.8992 |
| TrackOn2 visibility_selective | 0.5789 | 49.9694 | 67.3224 | 65.2523 |
| TrackOn2 original unconditional | 0.6101 | 50.0152 | 70.2686 | 68.5814 |

Interpretation:

```text
The visibility-selective memory policy does not improve this ReEntry-TAP smoke; it worsens translate and does not meaningfully improve occluder.
```

---

## Input-scale ablation

Translate L16 dev0 256 queries:

| Input scale | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | mean pred vis rate |
|---|---:|---:|---:|---:|---:|
| uint8 / 0..255 | 0.5775 | 47.9390 | 70.3470 | 67.0239 | 0.5647 |
| unit / 0..1 | 0.0246 | 1.8063 | 20.3439 | 12.2418 | 0.0440 |

Interpretation:

```text
The exporter is correct to pass 0..255 frames. Unit-scale input is wrong because TrackOn2 internally divides frames by 255.
```

---

## Final conclusion

TrackOn2 DINOv3 is technically runnable and the adapter is not grossly broken. Query-frame alignment is excellent, and input scale / coordinate scale are validated.

However, on ReEntry-TAP stress dev0, TrackOn2 remains substantially weaker than B2-W16-P2:

```text
translate full:
TrackOn2 AJ_RD=0.6132, AJ=29.6552
B2-W16-P2 AJ_RD=0.7239, AJ=77.2732

occluder full:
TrackOn2 AJ_RD=0.6238, AJ=30.3484
B2-W16-P2 AJ_RD=0.7619, AJ=77.9665
```

Recommended paper use:

```text
Do not use TrackOn2 DINOv3 ReEntry-TAP stress as a strong main-table baseline.
Report it only as appendix feasibility / protocol sanity if needed.
Use the parity-valid TrackOn2 first-input bridge as the safer external supplement.
```

Recommended next action:

```text
Stop TrackOn2 stress expansion for now.
Do not run dev0-9 or fresh20-49 TrackOn2 stress unless a native-protocol parity issue is fixed or a new adapter hypothesis emerges.
Proceed with the B2-W16-P2 + ReEntry-Guard paper story.
```
