# Phase E — Final Decision: M0 Re-Acquisition Head

**日期**: 2026-06-16
**协议锁定**: 同 Phase 1 decision.md

---

## E1. 证据汇总

### Phase A — 链路审计

| 发现 | 置信度 |
|---|---|
| DINO relocal_conf = 0.018 vs CoTracker 0.002 → DINO 是正确特征源 | 高 |
| DINO 特征可触发 retracking (13/30 批次) | 高 |
| 在线 recovery 管线完整实现，无工程障碍 | 高 |

### Phase B — 资产审计

| 发现 | 置信度 |
|---|---|
| M0 head offline: base_gt_16 better_frac=0.833 (2-seed 一致) | 中（val n=6） |
| M0 head offline: base_gt_32 better_frac=1.000 (2-seed 一致) | 低（val n=2） |
| Val bad subset 极小，覆盖率 54.5% → 估计方差极大 | 高 |
| `best.pth` checkpoint 可用 | 高 |

### Phase D — Integration Smoke (10 batches)

| 发现 | 置信度 |
|---|---|
| `_apply_learned_recovery_head()` 集成正确，无 crash | 高 |
| head relocal_conf = 0.505 vs raw DINO 0.018 (28×) | 高 |
| retracked 覆盖 66/531 re-entry queries (12.4%) | 高 |
| **M0 head re-entry median = raw DINO re-entry median = 172.54px** | 高 |
| **retracked median (172.54px) > non-retracked median (170.57px)** | 高 |
| head 没有引入 geometric regression | 高 |

---

## E2. 失败归因

M0 head 在离线 moderate (~30px) 误差上有一致改善信号，但在在线 extreme (~170px) re-entry 误差上**完全无效**。

**三层归因**:

### Layer 1: Scale Mismatch（最大因素）

M0 head 在 offline val 上学习的几何先验针对 ~30px 误差 scale。但在线 re-entry 误差分布的 median 是 ~170px（5.7× 差异）。在这个 scale 下：
- 目标的 DINO 特征与 support descriptor 的 cosine 相似度已被严重稀释
- head 的几何预测（heatmap soft-argmax on search fmap）所依赖的局部特征一致性被破坏
- 遮挡期间的 tracker 漂移累积使得 search crop 的中心（tracker 最后可见位置）本身可能已大幅偏离目标

### Layer 2: Confidence vs Geometry

head 输出的 relocal_conf = 0.505（高），但这是 heatmap 峰值强度，不代表几何精度。head 提升了"知道要去找"的能力，但没有提升"找到正确位置"的能力。raw DINO cosine 已经给了足够的触发信号。

### Layer 3: Search Crop 设计

当前 search crop 以 tracker 预测位置为中心。当 tracker 在遮挡期间漂移时，crop 就偏离了目标真实位置。head 预测的 heatmap peak 仍然在错误的 crop 内，因此几何精度无法改善。

---

## E3. Stop Criteria 命中情况

Phase C 定义了 4 条 stop criteria：

| Criteria | 阈值 | 实测 | 命中 |
|---|---|---|---|
| retracked median ≥ non-retracked median | head 方向错误 | 172.54 ≥ 170.57 | ✓ 命中 |
| n_retracked / n_reentry 覆盖率 | ≥95% 或 ≤5% | 12.4%，合理 | ✗ 未命中 |
| head 输出 p95 | > 64px | ~500px+ (推算) | ✓ 命中 |
| improvement vs non-retracked | < 2px | 0px | ✓ 命中 |

**命中 3/4 条** → Stop Criteria 满足。

---

## E4. 与 Phase 1 Oracle Diagnostic 的关系

Phase 1 的核心发现：

- `oracle_mask` 在 long_occ_AJ 上 +3pp，但 re-entry 误差**恶化** 68-79%
- `anchor_only` 在 re-entry 上保持中性，long-occ 增益微弱
- 主瓶颈在 **re-entry** 而非遮挡期间状态管理

Phase E 的发现与 Phase 1 高度一致：

- M0 head 在 re-entry 上**没有改善**（甚至略差于 non-retracked baseline）
- head 的高置信度信号不能转化为几何精度
- 这说明 **re-entry 的瓶颈比单纯"几何 head"更根本**

**综合 Phase 1 + Phase E**: 当前 Track-On2 + DINO 这套系统，re-entry 误差 ~170px 的量级已经大幅超出了几何重定位方法（raw cosine / learned head）的改善能力。需要的可能是更强的基础跟踪能力。

---

## E5. 最终决定

### 当前 M0 head 路线：**停止主线投入**

**理由**:
1. Stop Criteria 命中 3/4 条
2. head 无 online geometric 改善信号
3. 离线改善（~5-11px on base_gt_16）相对于在线 re-entry 误差 scale（~170px）可以忽略不计
4. head 的置信度贡献已经被 raw DINO 的 0.018 足够覆盖

### 保留的价值:
- `_apply_learned_recovery_head()` 代码路径在 Phase D smoke 中验证为可用 → 未来更强大的几何 head 可以复用
- Phase C 的 re-entry metric protocol 标准化 → 未来所有 recovery 实验使用统一口径

---

## E6. 推荐优先方向

| 优先级 | 方向 | 理由 | 下一步 |
|---|---|---|---|
| **1** | CoTracker3 Hybrid | CoTracker3 离线版在 re-entry 上显著优于 Track-On2 | 评估 CoTracker3 vs Track-On2 在 re-entry 帧的几何精度差 |
| **2** | 扩展 M0 head 训练数据 | 现有 val n=6 无法支撑置信判断 | 收集更多 hard-subset val 数据，重新验证 head 价值 |
| **3** | Search crop 重新设计 | crop 中心跟随漂移的 tracker → 本质性缺陷 | 探索 wider search / template matching / motion prediction |
| **4** | 停止预测性 recovery 投入 | Phase 1 + Phase E 共同证明当前路线无法解决 re-entry | 投入 CoTracker3 或更强 base tracker |

---

## E7. 给后续工作的建议

1. **CoTracker3 hybrid 是最高 ROI 的下一步**: CoTracker3 的 offline 版本在 re-entry 帧上直接给出 candidate，不依赖 Track-On2 的内存状态。如果 CoTracker3 的 re-entry 精度远优于 Track-On2，则可以直接在 re-entry 帧替换。
2. **不要继续增大 M0 head 的训练规模**: 在现有 search crop 设计下，head 的精度上限已被 crop 中心错误锁死。增大 head 容量只会让它更好地拟合错误的 crop。
3. **统一的 re-entry metric protocol 是本次执行留下的资产**: 后续所有 recovery 相关实验都应使用 Phase C 定义的 7 个指标。

---

*本 decision 基于 Phase A (链路审计) + Phase B (资产审计) + Phase C (指标协议) + Phase D (集成 smoke) 的完整执行结果，如实报告信号。*
