# DiReCT Quick-Screen v1.1 Split Validation Status (2026-06-12)

## 三句话

1. **DiReCT 快筛 split 验证 FAIL**：tiny tuned 在 train→val 上 top5% = 0.021，远低于 Gaussian baseline 的 0.711。
2. 同集拟合（v1）top5% = 0.918 → split 后 0.021，泛化完全消失。这与 patch verifier v1 的教训完全一致。
3. **DiReCT 方向正式停止，不再进入 DiReCT-lite 阶段 2。**

## 结果

| Variant | top5% (同集) | top5% (split) |
|---|---|---|
| Gaussian baseline | 0.711 | 0.711 |
| Tiny tuned (all→all) | **0.918** | — |
| Tiny tuned (train→val) | — | **0.021** |
| Gaussian hard (base>16) | 0.263 | 0.263 |
| Tuned hard (train→val) | 0.789 | **0.000** |

## 失败原因

1. Q/K 投影在 train set 上学到了样本特定的映射，无法泛化到 val set 的新视频
2. 384 维 DINO 特征在 train/val 间的分布差异使 Q/K 投影失效
3. 数据规模（train=149, val=97）不足以学习跨视频的通用 cross-attention 模式
