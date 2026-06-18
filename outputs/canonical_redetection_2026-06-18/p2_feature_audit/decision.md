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
| V0: Query-frame anchor | 85.1px | 0.0% | 4.7% | 0.0% |
| V1: Last-visible anchor | 89.7px | 0.0% | 0.0% | 0.0% |
| V2: CT-offline local (32px) | 18.3px | 2.3% | **29.7%** | **40.0%** |

## Go/Stop 判定

| Criteria | 实测 | 判定 |
|---|---:|---|
| overall top5@16px >= 20% (Strong Go) | 29.7% (V2 only) | ✅ |
| long-occ top5@16px >= 10% (Strong Go) | 40.0% (V2 only) | ✅ |
| top1 miss but top5 hit @16px >= 10% | 3.9% | ❌ |

## Decision

**BRANCH_GO → P3A (diagnostic only)**

V2 (CT-offline centered local search) satisfies two Strong Go criteria.
V0/V1 (whole-frame retrieval) still fail.

Under the current plan, P3A is oracle-only / stop, so this branch is
diagnostic rather than a training rollout.
