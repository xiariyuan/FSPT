# Phase 2b — Task C: Local-Global Hybrid Candidate Generation Smoke

**日期**: 2026-06-18  
**搜索半径**: 64px  
**Anchor**: Last-visible frame GT patch  
**Feature**: DINOv2 ViT-S/14

---

## Results

| Variant | n | top1@8 | top5@8 | top1@16 | top5@16 | top5@32 |
|---|---:|---:|---:|---:|---:|---:|
| global_whole_frame (baseline) | 128 | 0.0% | 0.0% | 0.0% | 0.0% | 7.0% |
| global_topk_then_local_refine | 128 | 0.0% | 2.3% | 0.0% | 7.0% | 15.6% |
| center_ct_offline | 128 | 0.0% | 2.3% | 0.0% | 7.0% | 28.1% |
| center_dual | 128 | 0.0% | 2.3% | 0.0% | 7.0% | 18.0% |

## Stop Decision

All variants **top5@16px < 10%** → **STOP**

## Key Observation

`center_ct_offline` at top5@32px = 28.1% is the only positive signal — it shows the CoTracker3 offline prior does help when threshold is relaxed. But 16px precision is not achievable with DINOv2 template matching across occlusion boundaries.
