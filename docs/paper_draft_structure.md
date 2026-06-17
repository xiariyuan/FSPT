# 论文结构与内容初稿

## 题目

**Support-Memory Driven Causal Re-Entry Selection for Long-Term Point Tracking**

副标题可选：
- *A Benchmark and Lightweight Selector for Occluded Point Re-Localization*
- *Exploiting Pre-Occlusion Memory for Post-Occlusion Point Recovery*

## 投稿目标

TCSVT / Pattern Recognition (CCF B)

---

## 摘要要点

1. 长时点跟踪中，长时间遮挡后的重新定位（causal re-entry）是关键瓶颈
2. 现有方法在遮挡期间仅依赖几何 hold，re-entry 时误差大
3. 我们发现：**pre-occlusion support memory**（遮挡前可见帧的 patch 特征）提供了真实可用的 re-entry 信号
4. 提出 **Support-Memory Hybrid Selector**：一个无需学习的显式评分规则
   - `hybrid_score = 4.0 * support_margin + 0.5 * cand_score`
   - support_margin = cosine(candidate, support_pool) - cosine(baseline, support_pool)
5. 在 PointOdyssey 3-seq val (n=300) 上：
   - Full coverage: median 36.23 → 32.97（-9.0%）
   - Selective 75% coverage: median 31.30（-13.6%）
6. 详细分析证明 support_margin 是主导信号（AUC 0.673），且 learned gate 无法超越显式规则

---

## 贡献点

1. **PRT Benchmark**：提出 Post-Reentry Tracking benchmark，标准化评估长时间遮挡后的点恢复能力
2. **Support-Memory Signal Analysis**：首次量化分析 pre-occlusion memory 对 causal re-entry 的判别力
3. **Hybrid Selector**：提出无需学习的显式 selector，在 full coverage 和 selective 模式下均优于 baseline
4. **Failure Analysis**：深入分析 selector 的失败模式，指出信号质量而非 decision boundary 是当前瓶颈

---

## 章节结构

### 1. Introduction（~1.5 页）

**段落结构**：
- P1: 长时点跟踪的动机和应用场景（video editing, VFX, motion capture, robotics）
- P2: 长时间遮挡后 re-entry 是核心难点。现有方法（cotracker, TAPIR 等）在短期遮挡表现好，但长期遮挡后失效
- P3: 我们观察到一个关键现象：pre-occlusion visible frames 提供了稳定的 point identity descriptor
- P4: 贡献点列表

**关键 Figure**: 一个 teaser figure，展示遮挡前 support frames → re-entry → hybrid selector 优于 baseline

### 2. Related Work（~1 页）

- 2.1 Point Tracking（cotracker, TAPIR, PIPs, BootsTAPIR, LocaTracker）
- 2.2 Long-term Tracking and Re-identification
- 2.3 Template Matching and Memory-based Methods

### 3. Problem Formulation（~0.5 页）

- 3.1 PRT Pipeline: Stage 0 (3D hold) → Stage 1 (reprojection) → Stage 2 (local candidate search) → Stage 3 (selection)
- 3.2 Formal definition of causal re-entry selection

### 4. Support-Memory Signal Analysis（~1.5 页）— **核心分析章节**

**4.1 Support Memory Construction**
- 从最后 M 个可见帧提取 patch，通过 DINOv2 编码
- Mean pooling 得到 persistent point descriptor

**4.2 Per-Feature AUC Analysis**
- Table: support_margin AUC=0.673, cand_score=0.483, cand_ncc=0.502
- 结论: support_margin 是唯一有效信号

**4.3 Oracle Gap Analysis**
- 各 top-k 子集的 oracle median（top-1 到 top-5）
- Oracle gap vs occlusion length / camera motion

**4.4 Pooling Strategy Ablation**
- Table: mean / max / softmax / per-frame-max → 几乎无差异
- 结论: 信号质量不随 pooling 策略变化，瓶颈在 DINOv2 特征本身

### 5. Support-Memory Hybrid Selector（~1 页）— **方法章节**

**5.1 Formulation**
- `hybrid_score = α * support_margin + β * cand_score`
- 系数通过 train set sweep 选定（α=4.0, β=0.5）

**5.2 Two Operating Points**
- Full coverage (threshold=0): always pick best candidate
- Selective: threshold learned on train, apply on val

**5.3 Why Not Learned Gate?**
- LogisticRegression CV AUC = 0.558（接近随机）
- MLP gate 输出集中在 0.5 附近，无法学到有效 boundary
- 结论: 当前信号空间下，显式规则 ≈ learned gate

### 6. Experiments（~2 页）

**6.1 Setup**
- Dataset: PointOdyssey, 3 val sequences, 300 samples
- Train: separate 1 sequence, 200 samples
- Evaluation: Median px, <4px rate, better_frac, coverage

**6.2 Main Results**
- Table 1: Baseline / Oracle / Hybrid Full / Hybrid Selective
  - Baseline: 36.23 median, 0.020 <4px
  - Oracle: 25.90, 0.130
  - Hybrid Full: 32.97, 0.063
  - Hybrid Selective (75%): 31.30, 0.067

**6.3 Coverage-Risk Curve**
- Figure: coverage vs median error
- 从 100% coverage (32.97) 到 75% (31.30) 到 0% (36.23)

**6.4 Statistical Significance**
- Paired permutation test: p=0.0386 (median)
- Bootstrap: mean 32.79 ± 2.91

**6.5 Per-Category Analysis**
- Table: short/long occlusion, low/high camera motion
- Long occlusion (>100 frames): gap 更大，selector 更有效
- Low camera motion: oracle gap 最大（9.39px），改善空间最大

**6.6 Failure Analysis**
- 49% 样本 hybrid 选错且更差（损失 5.11px）
- 但 51% 选对且显著改善（选对子集 median 18.1 vs baseline 30.0）
- 主要失败原因: support_margin 在 gap 小的样本上缺乏判别力

### 7. Discussion and Limitations（~0.5 页）

- 当前利用了 oracle gap 的 31%，上限空间仍大
- top-k=5 限制了候选覆盖度
- DINOv2 特征对某些物体类别判别力不足
- 未来: 更好的 feature backbone, 扩大 top-k, temporal consistency

### 8. Conclusion（~0.3 页）

---

## 关键数字统一

所有数字来源：`outputs/prt_hybrid_selector_v3_3seq/` 和 `outputs/prt_hybrid_selector_paper_table.json`

| 数字 | 值 | 来源 |
|------|-----|------|
| Baseline median | 36.23 | summary.json → baselines.baseline_only |
| Oracle median | 25.90 | summary.json → baselines.oracle |
| Hybrid full median | 32.97 | summary.json → operating_points.full_coverage |
| Hybrid 75% median | 31.30 | threshold_sweep_val.json → coverage=0.75 |
| support_margin AUC | 0.673 | auc.json |
| cand_score AUC | 0.483 | auc.json |
| cand_ncc AUC | 0.502 | auc.json |
| Paired perm p (median) | 0.0386 | significance.json |
| Bootstrap mean | 32.79 ± 2.91 | bootstrap_stability.json |
| n_val | 300 | 3 sequences |
| n_train | 200 | 1 sequence |
| Formula | 4.0 * support_margin + 0.5 * cand_score | summary.json |

## 关键 Figures 需求

1. **Teaser**: support memory → re-entry → hybrid vs baseline 对比可视化
2. **Pipeline diagram**: Stage 0 → 1 → 2 → 3(Hybrid Selector)
3. **Coverage-risk curve**: x=coverage, y=median_px
4. **Per-feature AUC bar chart**
5. **Per-category analysis table/figure** (occ_length × camera_motion)
6. **Failure case visualization**: 选对 vs 选错的典型样例
