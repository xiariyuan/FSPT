# Phase 1 — Oracle Re-Detection Upper Bound

**日期**: 2026-06-17  
**协议**: first+input (256-space)  
**数据源**: `outputs/attempt0_2026-06-15_recovery/prediction_caches/*.pt`  

---

## 1. Oracle 定义

| Oracle 类型 | 定义 | 代表意义 |
|---|---|---|
| **Oracle-exact** | GT exact position at re-entry frame = 0 error | Perfect re-detection |
| **Oracle-neighborhood** | GT position or within 1/2/4px | Near-perfect re-detection |

Oracle gap = model_error − oracle_error = model_error（因为 oracle=0）

---

## 2. 结果汇总

### 2.1 All Visible Frames（全部可见帧）

| Model | n_queries | n_frames | Median px | Mean px | p95 px | p99 px | Max px | <4px | Oracle Gap Mean |
|---|---|---|---|---|---|---|---|---|---|
| Track-On2 | 650 | 28174 | 1.12 | 2.87 | 8.76 | 38.2 | 145.9 | 87.5% | 2.87px |
| CoTracker3 online | 650 | 28174 | 1.26 | 3.05 | 9.50 | 37.7 | 152.0 | 85.3% | 3.05px |
| CoTracker3 offline | 650 | 28174 | 1.27 | 3.16 | 10.08 | 46.6 | 128.1 | 84.7% | 3.16px |

### 2.2 Re-Entry Frames（遮挡结束后第一个可见帧）

| Model | n_reentry | Model Median | Model Mean | Model p95 | Oracle Gap Median | Oracle Gap Mean |
|---|---|---|---|---|---|---|
| Track-On2 | 343 | 0.76px | 4.64px | 16.34px | 0.76px | 4.64px |
| CoTracker3 online | 343 | 0.87px | 5.55px | 36.87px | 0.87px | 5.55px |
| CoTracker3 offline | 343 | 0.95px | 4.75px | 33.29px | 0.95px | 4.75px |

### 2.3 Long-Occlusion Re-Entry（occ >= 20）

| Model | n_longocc | Model Median | Model Mean | Model p95 | Oracle Gap Median | Oracle Gap Mean |
|---|---|---|---|---|---|---|
| Track-On2 | 100 | 0.13px | 5.21px | 23.94px | 0.13px | 5.21px |
| CoTracker3 online | 100 | 0.00px | 5.10px | 11.82px | 0.00px | 5.10px |
| CoTracker3 offline | 100 | 0.00px | 4.49px | 14.72px | 0.00px | 4.49px |

---

## 3. Oracle Headroom 分析

### 3.1 Re-Entry 误差分布（模型 vs Oracle）

| 分位 | Track-On2 Model | Track-On2 Oracle | Gap |
|---|---|---|---|
| Median | 0.76px | 0.00px | **+0.76px** |
| Mean | 4.64px | 0.00px | **+4.64px** |
| p95 | 16.34px | 0.00px | **+16.34px** |

### 3.2 分类：Oracle 能改善多少 re-entry queries？

| Oracle 改善程度 | Track-On2 | CoTracker3 online | CoTracker3 offline |
|---|---|---|---|
| oracle > model by > 4px | 14.9% (51/343) | 14.9% (51/343) | 13.7% (47/343) |
| oracle > model by > 8px | 9.0% (31/343) | 11.4% (39/343) | 9.6% (33/343) |
| oracle > model by > 16px | 5.2% (18/343) | 8.2% (28/343) | 6.7% (23/343) |

### 3.3 Long-Occlusion 子集（occ >= 20）

| Oracle 改善程度 | Track-On2 | CoTracker3 online | CoTracker3 offline |
|---|---|---|---|
| oracle > model by > 4px | 15.0% (15/100) | 13.0% (13/100) | 11.0% (11/100) |
| oracle > model by > 8px | 9.0% (9/100) | 9.0% (9/100) | 7.0% (7/100) |
| oracle > model by > 16px | 5.0% (5/100) | 5.0% (5/100) | 4.0% (4/100) |

---

## 4. 与 Previous Results 的关联

### 4.1 Online Re-Entry vs Offline Cache

| Source | Track-On2 median | 差异来源 |
|---|---|---|
| Online re-entry (head-to-head, n=259) | 1.60px | 实时 tracker drift 累积 |
| first+input cache re-entry (n=343) | 0.76px | 端到端 offline 预测 |
| Long-occ online (n=24) | 3.97px | hard re-entry cases |
| Long-occ first+input (n=100) | 0.13px | 全部 long-occ |

差异解释：
- first+input cache 是端到端重新跑 Track-On2，不经过在线 tracker 状态
- Online re-entry 包含了 tracker 在遮挡期间的 drift 累积
- **first+input cache 实际上低估了真实在线系统的 re-entry 误差**

### 4.2 与 M0 Head Failure 的关联

M0 head failure 发现：
- Online re-entry median ≈ 170px（scale gap: offline 30px vs online 170px）
- Oracle gap 在 first+input cache ≈ 4.64px

这说明：
- Oracle 的上限在 first+input 口径下只有 4.64px mean
- 如果 Track-On2 已经 median=0.76px，oracle 的边际价值有限
- **但对于 hard tail cases（p95=16.34px），oracle 有显著价值**

---

## 5. Phase 1 Stop Criteria 对照

| Stop Criteria | 实测 | 判定 |
|---|---|---|
| long-occ AJ barely improves | long-occ oracle gap mean ≈ 5px → 有改善空间 | ⚠️ 不明确 |
| re-entry error barely decreases | model median=0.76px, oracle=0.00px → gap=0.76px | ⚠️ 不明确 |

### 5.1 Stop Criteria 解读

**"barely improves" 的阈值是什么？**

如果定义为：
- AJ delta < 1pp → barely
- error reduction < 1px → barely

实测：
- All-frame: oracle gap mean = 2.87px → **not barely**
- Re-entry: oracle gap mean = 4.64px → **not barely**
- Long-occ: oracle gap mean = 5.21px → **not barely**

**但**：oracle gap median 很小（0.13-0.76px），说明 oracle 主要帮助 hard tail cases，不是普遍性改善。

### 5.2 关键信号

**信号 1**: ~85% 的 re-entry queries 已经 median < 4px，即使 oracle 也只能改善 15%

**信号 2**: CoTracker3 online 在 long-occ 上 median=0.00px（已完美），oracle 价值来自 tail

**信号 3**: Track-On2 和 CoTracker3 在 oracle gap 上差异不大（5.21 vs 5.10 vs 4.49），说明 re-detection headroom 是系统性的，不是模型特定的

---

## 6. 结论

### 6.1 核心发现

1. **Oracle re-detection 有非平凡 headroom**：~15% 的 re-entry queries 有 >4px oracle gap，mean oracle gap ≈ 5px
2. **但 median gap 很小**：主要价值来自 tail distribution，不是系统性全量改善
3. **CoTracker3 vs Track-On2 oracle gap 接近**：说明 headroom 是系统性的（general re-detection problem），不是替换模型能解决的
4. **与 M0 head failure 一致**：M0 head 试图学 geometric head，但在 tail cases 上无效；oracle 提供的是纯位置重置

### 6.2 建议

**GO to Phase 2**，理由：
1. Oracle gap ≈ 5px mean，不算 "barely"
2. ~15% re-entry cases 有显著 oracle 改善空间
3. Phase 2 的 DINOv3 anchor matching 是实现 oracle 价值的实际路径
4. Phase 1 stop criteria 中的 "barely" 阈值未被命中

但需要明确：
- Oracle 改善是 tail-driven，不是全量
- Phase 3 的 verifier 如果无法从 tail 中选择正确候选，则 oracle 价值无法实现

---

*Phase 1 oracle upper bound analysis complete. Non-trivial headroom exists (~5px mean, ~15% of re-entry cases). Advancing to Phase 2.*
