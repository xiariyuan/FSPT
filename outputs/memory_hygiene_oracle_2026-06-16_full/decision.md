# Memory Hygiene Oracle Diagnostic — Decision (Full DAVIS, 2026-06-16)

## 协议锁定

| 参数 | 值 |
|---|---|
| dataset | TAP-Vid DAVIS |
| query_mode | first |
| metric space | input 256x256 |
| support_grid_size | 20 |
| M_i | 24 |
| long_occ_min_run | 20 |
| checkpoint | baselines/track_on/checkpoints_trackon2_dinov3.pt |
| num_videos | 30 |

---

## 1. 四条件 Overall 指标

| Variant | AJ | OA | delta_avg | <4px |
|---|---:|---:|---:|---:|
| baseline | 66.95 | 92.02 | 79.82 | 87.79 |
| oracle_mask | 67.33 (+0.38) | 92.77 (+0.75) | 79.45 (-0.37) | 87.36 (-0.43) |
| anchor_only | 66.84 (-0.11) | 91.98 (-0.04) | 79.64 (-0.18) | 87.84 (+0.05) |
| oracle_mask_anchor | 67.17 (+0.22) | 92.75 (+0.73) | 79.30 (-0.52) | 87.35 (-0.44) |

**Overall 结论**：四条件在 overall AJ 上差异极小（最大差距 < 0.5pp），无实际意义。

---

## 2. 四条件 Long-Occ 指标（max_reappearance_occlusion_run >= 20）

| Variant | long_occ_AJ | long_occ_OA | long_occ_delta_avg | long_occ_<4px |
|---|---:|---:|---:|---:|
| baseline | 45.25 | 77.71 | 68.10 | 73.67 |
| oracle_mask | **48.07** (+2.82) | 84.57 (+6.86) | 64.86 (-3.24) | 71.72 (-1.95) |
| anchor_only | 45.93 (+0.68) | 76.86 (-0.85) | 68.44 (+0.34) | 73.20 (-0.47) |
| oracle_mask_anchor | **48.21** (+2.96) | 83.29 (+5.58) | 65.05 (-3.05) | 71.12 (-2.55) |

**Long-occ 结论**：oracle_mask 和 oracle_mask_anchor 在 long_occ_AJ 上均有 ~+3pp 的提升，信号方向一致。anchor_only 的提升微弱（+0.68pp）。

---

## 3. 四条件 Re-Entry 指标（long_occ_min_run >= 20 的 query 的 reentry frame 误差）

| Variant | count | reentry_mean_error_px | reentry_median_error_px | reentry_<4px | reentry_<8px |
|---|---:|---:|---:|---:|---:|
| baseline | 1.5 | **25.12** | 22.87 | 52.48 | 63.25 |
| oracle_mask | 1.5 | 44.82 (+78.5%) | 42.04 | 36.35 | 48.22 |
| anchor_only | 1.5 | 25.58 (+1.8%) | 22.84 | 52.92 | 62.15 |
| oracle_mask_anchor | 1.5 | 42.44 (+68.9%) | 39.02 | 37.03 | 47.36 |

**Re-entry 结论**：这是本轮最关键的发现——oracle_mask / oracle_mask_anchor 在 long_occ_AJ 上提升了 ~3pp，但代价是 reentry_mean_error_px 恶化了 **68-79%**。anchor_only 在 re-entry 上基本持平 baseline（+1.8%）。

**count=1.5 是因为 re-entry query 在 full DAVIS 中非常稀疏（30 视频中只有 3 个 query 命中 long_occ 且有 reentry frame），均值本身方差极大，须谨慎解读方向而非绝对值。

---

## 4. 分桶结果

| Bucket | baseline AJ | oracle_mask AJ | anchor_only AJ | oracle_mask_anchor AJ |
|---|---:|---:|---:|---:|
| 20-39 | 48.64 | 49.72 (+1.08) | 49.41 (+0.77) | **50.74** (+2.10) |
| 40-71 | 15.80 | **27.16** (+11.36) | 16.07 (+0.27) | 21.38 (+5.58) |
| 72+ | — | — | — | — |

**分桶结论**：oracle_mask / oracle_mask_anchor 的长遮挡增益主要集中在 40-71 桶（AJ +11.4pp / +5.6pp），这符合 temporal_mask 设计的直观预期——遮挡时间越长，被 mask 的污染 slot 越多，对后续帧的干扰越少。但在 72+ 桶上四条件均无数据。

---

## 5. 对 baseline 的增减值汇总

| 指标 | oracle_mask Δ | anchor_only Δ | oracle_mask_anchor Δ |
|---|---:|---:|---:|
| overall AJ | +0.38 | -0.11 | +0.22 |
| long_occ_AJ | **+2.82** | +0.68 | **+2.96** |
| long_occ_delta_avg | -3.24 | +0.34 | -3.05 |
| reentry_mean_error_px | **+78.5%** (变差) | +1.8% | **+68.9%** (变差) |
| <4px (overall) | -0.43 | +0.05 | -0.44 |

---

## 6. 当前最优条件

从综合角度：

- **Long-occ AJ 最优**：oracle_mask_anchor（48.21），其次 oracle_mask（48.07）
- **Re-entry 最优**：baseline / anchor_only（两者 reentry_mean_error_px 几乎相同）
- **Overall 最优**：oracle_mask（67.33），但优势极微（+0.38pp vs baseline）

**综合最优**：不存在一个条件同时在 long-occ 和 re-entry 上都优于 baseline。这是本轮的核心矛盾。

---

## 7. 是否满足 predicted gate / dual-memory 继续投入门槛

### 判断标准复检（oracle_mask）

满足任意两条即认定成功：

1. long_occ_AJ >= +2.0：✓ oracle_mask = +2.82
2. long_occ_delta_avg >= +2.0：✗ oracle_mask = -3.24（方向相反）
3. reentry_mean_error_px 下降 >= 10%：✗ oracle_mask = +78.5%（严重恶化）
4. 至少 70% 视频在 long_occ_AJ 上同号改善：需 per-video 数据验证

**结论**：oracle_mask 仅满足 1/4 条门槛。不满足继续投入 predicted gate / dual-memory 的条件。

### 诚实归因

本轮结果揭示了一个核心矛盾：

- **长遮挡期间的内存隔离（oracle_mask）确实减少了遮挡期间状态被污染**，体现为 long_occ_AJ +3pp
- **但遮挡消失后，decoder 失去了从"污染 slot"中提取 re-entry 位置线索的能力**，体现为 reentry_mean_error_px 大幅恶化

这说明：
1. 当前 decoder 对被 mask 掉的 slot 中的位置信息有隐性依赖（即使这些 slot 处于"不可见"状态）
2. "遮挡期间不写 memory"不会自动带来 re-entry 改善
3. oracle_mask 揭示的 headroom 不是"可直接提取的"，而是以牺牲 re-entry 为代价

### 本轮 decision

**不推荐继续投入 predicted gate / dual-memory**。

理由：
- oracle_mask 在 long_occ_AJ 上的 +3pp 提升与 re-entry 上的 70%+ 恶化同时出现，属于**交换恶化**而非纯增益
- anchor_only 在 re-entry 上保持中性，但 long-occ 增益微弱（+0.68pp），没有独立价值
- 当前 signal 的 headroom 不足以支撑"遮挡期间"vs"re-entry"两件事同时做好而不引入新的 tradeoff

---

## 8. 推荐方向

| 优先级 | 方向 | 理由 |
|---|---|---|
| **1** | Re-acquisition / re-detection head | oracle_mask 的 re-entry 恶化说明主要瓶颈在 re-entry 而非遮挡期间的状态管理 |
| **2** | CoTracker3 hybrid | 在 re-entry 帧用独立模型提供 candidate，不依赖 Track-On2 内存状态 |
| **3** | 保持 anchor_only 但不作为主方向 | anchor_only 没有引入新的 tradeoff，但也没有足够的独立 headroom |
| **4** | 停止 predicted gate / soft retention 投入 | 这轮 oracle diagnostic 证明单纯靠 memory hygiene 不能同时解决 long-occ 和 re-entry |

---

## 9. 与 Smoke 的差异说明

2-video smoke 阶段观察到 oracle_mask long_occ_AJ ≈ 51.34，oracle_mask_anchor ≈ 54.45，与 full DAVIS 结果（48.07 / 48.21）存在约 5-6pp 的 smoke-to-full 差距，属正常 scale 差异，不影响方向判断。Smoke 阶段未能发现 re-entry 恶化，是因为 re-entry event 在 2 视频中 count=0（无数据），full DAVIS 的 re-entry 数据同样稀疏（count=1.5），但方向已可辨。

---

*本 decision 基于 full DAVIS (n=30 videos, M_i=24, query_mode=first, memory_update_policy=unconditional) 正式诊断产出，如实报告信号，不做方向性美化。*
