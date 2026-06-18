# Phase 3 — CT-Offline Local Refiner Dataset Manifest

**日期**: 2026-06-18  
**协议**: strided+original

---

## 1. Dataset Specification

| 参数 | 值 |
|---|---|
| Baseline | CoTracker3 offline |
| Search center | CT-offline prediction at re-entry frame |
| Search crop size | 224×224 (DINOv2 input) |
| Primary radius | 16px (GT inside crop: 99.4%) |
| Support patches | 4 frames (last visible before occlusion) |
| Support crop size | 112×112 |
| Feature extractor | DINOv2 ViT-S/14 |

## 2. Samples

### Smoke (5 videos)

| 视频 | 样本数 | 划分 |
|---|---|---|
| bike-packing | 59 | Train |
| bmx-trees | 185 | Val |
| breakdance | 99 | Train |
| camel | 4 | Train |
| **Total** | **347** | Train: 162, Val: 185 |

### Data Sanity

| 指标 | Smoke (5v) | 参考 (oracle audit) |
|---|---|---|
| Raw CT-offline median | 3.2px | ~4.0px |
| Raw CT-offline <4px | 58.8% | ~49.2% |
| Oracle r16 <4px | 94.2% | ~91.4% |
| Long-occ raw <4px | 37.1% | ~10% |
| GT inside search crop (16px) | 99.4% | — |

## 3. Output Structure

```
dataset_smoke/
├── build_stats.json          # dataset-level stats
├── index_all.json            # metadata for all samples
├── index_train.json          # training split
├── index_val.json            # validation split
├── bike-packing/             # per-video directories
│   ├── sample_000000.npz     # support_descriptor, search_feature_map, search_crop, support_patches
│   └── ...
├── bmx-trees/
│   └── ...
└── ...
```
