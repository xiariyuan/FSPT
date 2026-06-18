# P2 Feature Audit Summary

**Date**: 2026-06-18  
**Protocol**: `strided+original`
**Scripts**: `scripts/audit_vdit_resource_smoke.py`, `scripts/eval_redetection_feature_audit.py`

## P2a: Resource Smoke

DINOv2 feature extraction is not an engineering dead end.

- `time/frame`: `2.47s`
- `feature grid`: `(1, 384, 37, 37)`
- `GPU peak`: `0.06 GB`
- `cacheable to disk`: `true`
- `OOM`: `false`

Decision: `PROCEED_TO_P2B`

## P2b: Top-K Recall Audit

`5 videos / 128 queries`, DINOv2 ViT-S/14.

| Variant | Top5 median | Top5 <4px | Top5 <16px | Long-occ Top5 <16px |
|---|---:|---:|---:|---:|
| V0: Query-frame anchor | 85.05px | 0.0% | 4.69% | 0.0% |
| V1: Last-visible anchor | 89.72px | 0.0% | 0.0% | 0.0% |
| V2: CT-offline local (32px) | 18.29px | 2.34% | 29.69% | 40.0% |

Go/Stop criteria:

- overall `top5@16px >= 20%`: `29.69%` `OK`
- long-occ `top5@16px >= 10%`: `40.0%` `OK`
- `top1 miss but top5 hit @16px >= 10%`: `3.91%` `NO`

## Decision

`PARTIAL_LOCAL_FEATURE_SMOKE`

This is diagnostic only. The only useful signal is the CT-offline-centered
local search branch; whole-frame retrieval remains weak. P3A pseudo-label
smoke is already stopped and must not be treated as a training path.
