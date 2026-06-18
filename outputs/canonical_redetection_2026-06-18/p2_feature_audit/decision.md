# P2 Decision

**日期**: 2026-06-18

## P2a — Resource Smoke

| 指标 | DINOv2 |
|---|---:|
| Time/frame | 2.47s |
| Feature grid | (1, 384, 37, 37) |
| GPU peak | 0.06 GB |
| Cacheable | ✅ |
| OOM | ❌ |

**Decision**: ✅ **PROCEED_TO_P2B**

## P2b — Feature Top-K Recall Audit

5 videos / 128 queries, DINOv2 ViT-S/14

| Variant | Top5 median | Top5 <4px | Top5 <16px | Long-occ Top5 <16px |
|---|---:|---:|---:|---:|
| V0: Query-frame anchor | 85.0px | 0.0% | 1.6% | 0.0% |
| V1: Last-visible anchor | 89.0px | 0.0% | 0.0% | 0.0% |
| V2: CT-offline local (32px) | 18.2px | 2.3% | **25.8%** | **20.0%** |

## Go/Stop 判定

| Criteria | 实测 | 判定 |
|---|---:|---|
| overall top5@16px >= 20% (Strong Go) | 25.8% (V2 only) | ✅ |
| long-occ top5@16px >= 10% (Strong Go) | 20.0% (V2 only) | ✅ |
| top1 miss but top5 hit @16px >= 10% | 1.6% | ❌ |

## Decision

**BRANCH_GO → P3A**

V2 (CT-offline centered local search) hits two of three Strong Go criteria.
V0/V1 (whole-frame retrieval) fail completely — consistent with prior findings.

The path forward is P3A: feature-based pseudo labels using CT-offline as coarse prior,
with local DINOv2 matching within a constrained search window.
