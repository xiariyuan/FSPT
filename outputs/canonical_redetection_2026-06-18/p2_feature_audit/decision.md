# P2 Decision

**日期**: 2026-06-19

**Decision**: `PARTIAL_LOCAL_FEATURE_SMOKE`

## P2a — Resource Smoke

| 指标 | DINOv2 |
|---|---:|
| Time/frame | 2.47s |
| Feature grid | (1, 384, 37, 37) |
| GPU peak | 0.06 GB |
| Cacheable | ✅ |
| OOM | ❌ |

## P2b — Feature Top-K Recall Audit

5 videos / 128 queries, DINOv2 ViT-S/14

| Variant | Top5 median | Top5 <16px | Long-occ Top5 <16px |
|---|---:|---:|---:|
| V0: Query-frame anchor | 85.0px | 1.6% | 0.0% |
| V1: Last-visible anchor | 89.0px | 0.0% | 0.0% |
| V2: CT-offline local (32px) | 18.2px | 25.8% | 20.0% |

## 当前判定

```
P2 = PARTIAL_LOCAL_FEATURE_SMOKE
原因:
- V0/V1 whole-frame retrieval 完全失败
- V2 不是 feature 本身强，而是依赖 CT-offline coarse prior
- Top5@16 = 25.8% 只是局部候选空间的弱信号
- top1 miss but top5 hit @16 = 1.6%，可学习空间极小
- 不足以推进 P3A 全面 rollout 或 P4 训练
```

## 与 P3A 的关系

P3A pseudo-label smoke 已确认失败（labels 太少、单视频支配、error 大于 CT-offline）。
P2 不应输出 BRANCH_GO → P3A。正确口径：
- P3A smoke 已做 → STOP
- P4 训练 → DO NOT START
