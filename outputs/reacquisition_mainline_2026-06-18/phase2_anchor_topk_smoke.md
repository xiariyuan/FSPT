# Phase 2 — DAVIS Anchor Top-K Recall Smoke

## Protocol

- Dataset: `tapvid_davis`
- Query protocol: `strided+original`
- Anchor source: `query-frame GT patch`
- Retrieval target: `first re-entry frame`
- Feature extractor: `DINOv2 ViT-S/14`
- Top-K: `5`
- Query crop size: `112`
- Sample cap: `max_videos=5`, `max_queries=128`

## Overall

| n | top1 median px | top5 best median px | top1@8 | top5@8 | top1@16 | top5@16 | top1 miss but top5 hit @16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 136.02 | 85.05 | 0.0% | 1.6% | 0.0% | 1.6% | 1.6% |

## Long-Occlusion (occ >= 20)

| n | top1 median px | top5 best median px | top1@8 | top5@8 | top1@16 | top5@16 | top1 miss but top5 hit @16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 108.10 | 85.02 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Buckets

| Occ bucket | n | top1@8 | top5@8 | top1@16 | top5@16 |
|---|---:|---:|---:|---:|---:|
| <20 | 108 | 0.0% | 1.9% | 0.0% | 1.9% |
| 20-49 | 20 | 0.0% | 0.0% | 0.0% | 0.0% |
| 50-99 | 0 | - | - | - | - |
| 100+ | 0 | - | - | - | - |

## Decision Hints

- `Phase 2 pass` requires strong Top-5 recall. The pre-registered ladder threshold was `Top-5 >= 70%`.
- `Phase 3 value` is indicated by `top1 miss but top5 hit`: the anchor is present in the candidate set, but ranking is wrong.
