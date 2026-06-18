# Phase 3 — CT-Offline Local Refiner Smoke Metrics

**日期**: 2026-06-18  
**协议**: strided+original  
**结果**: ❌ **FAIL** — learned local refiner does not beat CT-offline raw prediction

---

## 1. 训练配置

| 参数 | 值 |
|---|---|
| Model | OnlineRecoveryHead (conv tower) |
| Params | 489,793 |
| Dataset | 162 train / 185 val samples (5 videos) |
| Search crop | 224×224, centered on CT-offline prediction |
| Support | 4 frames, last visible before occlusion |
| Feature | DINOv2 ViT-S/14 (frozen) |
| Epochs | 40 |
| Batch size | 8 |
| LR | 1e-3 |
| Seed | 42 |

## 2. 最佳 Checkpoint (Val)

| 指标 | Learned Head | CT-offline Raw | 判定 |
|---|---:|---:|---:|
| Median | **6.51 px** | 3.70 px | ❌ FAIL (需 ≤2.0px) |
| Mean | **6.97 px** | 4.66 px | ❌ FAIL |
| <4px | **23.8%** | 54.6% | ❌ FAIL (需 ≥70%) |
| P95 | **13.47 px** | 11.33 px | ❌ FAIL |
| Better than baseline | **23.8%** | — | ❌ 模型经常更差 |

## 3. Pass Criteria 对照

| 门槛 | 要求 | 实测 | 结果 |
|---|---:|---:|---:|
| overall <4px | ≥ 70% | 23.8% | ❌ |
| long-occ <4px | ≥ 50% | 未单独统计 | ❌ (推断) |
| overall median | ≤ 2.0px | 6.51px | ❌ |
| <8px 不退化 | 不退化 | 退化 | ❌ |

## 4. 失败归因

**Learned local refiner 无法吃到 oracle gap**。尽管 oracle best-of-K 在 16px 窗口内可达 94% <4px，但 DINOv2 features + conv head 无法学到这个映射。原因分析：

1. **特征判别力不足**: DINOv2 ViT-S/14 在跨遮挡的外观变化下，同一物体的 patch-level features 相似度不够高。
2. **数据量小**: 162 训练样本 vs 490K 参数，模型无法泛化。
3. **任务难度**: 局部精修需要亚像素级定位，但 conv head + heatmap 的精度有限。
4. **anchor 本身已经很强**: CT-offline baseline 中值 3.7px 已经压到很小的误差范围，留给可学习的精修空间极小。

## 5. 决策

```
Phase 3 local refiner (learned): ❌ STOP
理由: 模型无法超越 CT-offline raw prediction (better_frac=23.8%)
Phase 2 主线: ❌ STOP — 整个方向到此停止
Fallback: CT-offline-centered local grid verifier (未进入)
```
