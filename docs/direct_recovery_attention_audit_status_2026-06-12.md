# DiReCT Quick-Screen Status (2026-06-12)

## 三句话

1. DiReCT 假设通过最小定位能力验证：tiny supervised cross-attention 的 GT top-5% hit rate = 0.918，显著优于 Gaussian baseline 的 0.711。
2. 在 hard subset（base>16）上信号更强：0.263 → 0.789，说明 cross-attention 学到了 Gaussian 先验无法覆盖的定位能力。
3. 快筛 PASS (2/3 criteria)，允许进入阶段 2 DiReCT-lite 原型。

## 快筛设置

- 数据：`outputs/recovery_anchor_dataset_v3_full/index_val.json`，n=97 valid samples
- 输入：`support_descriptor` (384,) + `search_feature_map` (384, 37, 37)
- Feature map: 37×37 = 1369 positions, D=384
- Variants: random projections / tiny tuned (300 steps) / Gaussian baseline

## 主结果

| Variant | GT top-1% | GT top-5% | peak-to-GT median |
|---|---|---|---|
| Random projection | 0.000 | 0.010 | 114.9px |
| **Gaussian baseline** | 0.526 | 0.711 | 13.5px |
| **Tiny tuned (300 steps)** | **0.660** | **0.918** | 13.5px |

### Hard subset (base>16, n=38)

| Variant | GT top-5% | peak-to-GT median |
|---|---|---|
| Gaussian baseline | 0.263 | 35.3px |
| **Tiny tuned** | **0.789** | 31.7px |

### Pass/Fail

| Criterion | 结果 |
|---|---|
| tuned top-5% > gaussian | **PASS** (0.918 > 0.711) |
| tuned peak-to-GT < gaussian | **FAIL** (13.5 = 13.5) |
| hard subset signal | **PASS** (0.789 > 0.263) |

**Verdict: PASS (2/3)**

## 解读

1. **Random projection 完全无信号**（top-5% = 0.010），说明 384 维随机投影无法定位 GT。
2. **Gaussian baseline 本身很强**（top-5% = 0.711），因为大多数样本 base error 小，GT 在 base 附近。
3. **Tiny tuned 显著优于 Gaussian**（0.918 vs 0.711），说明 cross-attention 学到了超越"GT 在 base 附近"先验的定位能力。
4. **Hard subset 上改善最大**（0.263 → 0.789），这是关键：当 base error 大时，Gaussian 先验失效，但 cross-attention 仍然有效。
5. Peak-to-GT 距离没有改善（13.5 = 13.5），说明 attention peak 不是精确落在 GT 上，但 GT 在 top-5% 的热区中。

## 进入阶段 2 的条件

快筛通过。但**数据规模严重不足**：

- Train: 149, Val: 97, Total: 246
- **清单要求 >= 1000 有效训练样本，当前只有 149，严重不足**
- **警告：training scale may be insufficient for direct recovery generalization.**

是否进入阶段 2 需要由用户决定。如果进入，必须在文档里标注数据不足的风险。
