# ReEntry-VisCalibrator Clean Protocol Results — 2026-07-03

## 1. 目的

这份文档记录学习版 `ReEntry-VisCalibrator V1` 的干净协议复核。

核心协议：

```text
训练学习模块：RGB dev0-6
选择阈值：    RGB dev7-9
最终测试：    RGB fresh20-49 natural / translate_L16 / occluder_L16
```

这样可以避免在 fresh20-49 上调阈值。

---

## 2. 数据和模型

切分后的缓存：

```text
train base:     outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_offline_rgb_dev0_6.pt
train override: outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_online_rgb_dev0_6.pt
val base:       outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_offline_rgb_dev7_9.pt
val override:   outputs/paper_discovery_2026-06-27/reentry_viscalibrator/protocol_clean_split/cotracker3_online_rgb_dev7_9.pt
```

训练数据：

```text
dev0-6 records: 7
dev0-6 queries: 8374
candidate windows: 8666
sequence length: 33 frames
feature dimension: 28
```

模型：

```text
outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1_dev0_6_clean.pt
```

注意：训练只使用 dev0-6；dev7-9 不进入训练，只用于阈值选择。

---

## 3. dev7-9 阈值选择

选择规则：

```text
在 dev7-9 上选择 AJ_RD_256 最高的阈值，同时要求 AJ_256 不低于规则版 W8P2 和 positive-only W16P2。
```

dev7-9 基线：

| 方法 | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline/base | 0.2080 | 80.0433 | 91.1636 |
| rule W8P2 | 0.3205 | 79.5826 | 93.8225 |
| positive-only W16P2 | 0.3208 | 79.5746 | 93.8356 |

学习版阈值扫描：

| 阈值 | AJ_RD_256 | AJ_256 | OA_256 | 恢复帧数 |
|---:|---:|---:|---:|---:|
| 0.10 **selected** | 0.3213 | 79.7517 | 94.0018 | 50427 |
| 0.15 | 0.3208 | 79.8247 | 93.9753 | 48985 |
| 0.20 | 0.3205 | 79.9030 | 93.8960 | 47203 |
| 0.30 | 0.3181 | 80.0359 | 93.6797 | 43281 |
| 0.50 | 0.2782 | 80.2929 | 92.9279 | 30322 |
| 0.65 | 0.2480 | 80.3216 | 92.2503 | 19101 |
| 0.75 | 0.2342 | 80.2667 | 91.7821 | 11474 |

结论：dev7-9 选择阈值 `0.10`。它在 dev7-9 上 AJ_RD_256 最高，并且 AJ_256 也高于 rule W8P2 和 positive-only W16P2。

---

## 4. frozen fresh20-49 结果，阈值固定为 0.10

| 设置 | 方法 | AJ_RD_256 | AJ_256 | OA_256 |
|---|---|---:|---:|---:|
| RGB fresh20-49 natural | offline/base | 0.3816 | 79.5944 | 91.4636 |
| RGB fresh20-49 natural | rule W8P2 | 0.4510 | 79.1110 | 92.8853 |
| RGB fresh20-49 natural | positive-only W16P2 | 0.4524 | 79.1098 | 92.8919 |
| RGB fresh20-49 natural | learned V1 clean thr0.10 | 0.4533 | 79.1867 | 92.9234 |
| fresh20-49 translate_L16 | offline/base | 0.4788 | 75.1940 | 90.7805 |
| fresh20-49 translate_L16 | rule W8P2 | 0.5336 | 74.7705 | 92.2245 |
| fresh20-49 translate_L16 | positive-only W16P2 | 0.5348 | 74.7679 | 92.2329 |
| fresh20-49 translate_L16 | learned V1 clean thr0.10 | 0.5354 | 74.8798 | 92.2390 |
| fresh20-49 occluder_L16 | offline/base | 0.6311 | 77.6280 | 90.8918 |
| fresh20-49 occluder_L16 | rule W8P2 | 0.6659 | 77.0436 | 92.1748 |
| fresh20-49 occluder_L16 | positive-only W16P2 | 0.6667 | 77.0443 | 92.1858 |
| fresh20-49 occluder_L16 | learned V1 clean thr0.10 | 0.6671 | 77.1320 | 92.2273 |

---

## 5. 学习版相对基线的增量

| 设置 | ΔAJ_RD vs rule W8 | ΔAJ vs rule W8 | ΔAJ_RD vs positive W16 | ΔAJ vs positive W16 | ΔAJ_RD vs offline | ΔAJ vs offline |
|---|---:|---:|---:|---:|---:|---:|
| RGB fresh20-49 natural | 0.0023 | 0.0757 | 0.0009 | 0.0769 | 0.0717 | -0.4077 |
| fresh20-49 translate_L16 | 0.0018 | 0.1093 | 0.0006 | 0.1119 | 0.0566 | -0.3142 |
| fresh20-49 occluder_L16 | 0.0012 | 0.0884 | 0.0004 | 0.0877 | 0.0360 | -0.4960 |

---

## 6. 复核结论

干净协议后，学习版结果更强：

```text
Natural:      learned AJ_RD=0.4533, rule=0.4510, positive-only W16=0.4524
Translate:    learned AJ_RD=0.5354, rule=0.5336, positive-only W16=0.5348
Occluder:     learned AJ_RD=0.6671, rule=0.6659, positive-only W16=0.6667
```

同时，学习版相对 positive-only W16P2 的普通追踪分数也更高：

```text
Natural:   +0.0769 AJ
Translate: +0.1119 AJ
Occluder:  +0.0877 AJ
```

这说明学习版不只是“想法可行”，而是已经形成了一个更好的综合版本：

```text
它比 rule W8P2 有更高 AJ_RD 和更高 AJ；
它比 positive-only W16P2 有更高 AJ_RD，并且 AJ 也更高；
它相对 offline/base 保留了显著重入提升，同时普通追踪损失仍然较小。
```

---

## 7. 论文表述建议

可以写：

```text
Using a clean train/validation/test protocol, ReEntry-VisCalibrator V1 is trained on dev0-6, selects threshold 0.10 on dev7-9, and improves over both the original rule W8P2 and the positive-only W16P2 rule on all three fresh20-49 settings.
```

中文：

```text
在干净协议下，学习模块只用 dev0-6 训练，在 dev7-9 上选择阈值 0.10，然后固定到 fresh20-49 测试。结果显示，它在三个最终设置上都超过原始规则版和 positive-only 简单规则，同时保持更好的普通追踪分数。
```

仍需注意：

```text
学习模块的提升幅度不巨大，但它现在已经是规则版的更优综合版本。
它仍然主要在 CoTracker-family 上验证，跨 tracker 泛化仍需 LocoTrack 等实验。
```
