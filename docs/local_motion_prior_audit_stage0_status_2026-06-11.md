# Local Motion Prior Recovery Audit — Stage-0 Status (2026-06-12)

## 结论

**Stage-0 FAIL。Motion-prior branch 失败——显式邻域运动先验不足以重新制造 local oracle gap。但 base-centered local DINO search 已重新制造出 oracle gap（base>16: 0.615, base>32: 0.800），与全局 template-bank 的 0.0 本质不同。新的唯一存活路线：base-centered local candidate generation → top1 / selector / verifier。**

运动先验（velocity/translation/affine）作为 crop 中心并不比 base tracker 更好。Base 已经是最优的 crop 中心。但这次审计同时证明了一个更重要的新事实：以 base 为中心的 local DINO search 已经重新制造出了 oracle gap，这和之前全帧 template-bank 的 0.0 是本质不同的信号。

## 关键数据

### Stage-0 通过标准检查

在 base>16px 子集上，neighbor_affine_center 需要满足 3 条中的 2 条：

| # | 标准 | 结果 |
|---|------|------|
| 1 | center_error_median 改善 ≥20% | **FAIL** (-21.7%, 更差) |
| 2 | GT_in_crop@64 提升 ≥+0.15 | **FAIL** (-0.077, 更差) |
| 3 | oracle_better_frac@2px@64 ≥0.25 | **PASS** (0.692) |

**1/3 通过 → Stage-0 FAIL**

base>32px 子集 oracle@2px@64=0.600 ≥ 0.35 → PASS，但 base>16 子集核心标准失败。

### 四种先验中心对比

| | all (n=106) | base>16 (n=13) | base>32 (n=5) |
|---|---|---|---|
| **base center_err_med** | **4.54** | **22.2** | **46.5** |
| velocity center_err_med | 8.84 | 39.8 | 57.8 |
| translation center_err_med | 5.46 | 37.2 | 49.9 |
| affine center_err_med | 5.46 | 27.0 | 49.9 |

| | all | base>16 | base>32 |
|---|---|---|---|
| **base GT@64** | **0.991** | **0.923** | **0.800** |
| velocity GT@64 | 0.774 | 0.692 | 0.600 |
| translation GT@64 | 0.981 | 0.923 | 0.800 |
| affine GT@64 | 0.972 | 0.846 | 0.600 |

| | all | base>16 | base>32 |
|---|---|---|---|
| **base oracle@2px@64** | **0.189** | **0.615** | **0.800** |
| velocity oracle@2px@64 | 0.189 | 0.538 | 0.600 |
| translation oracle@2px@64 | 0.198 | 0.692 | 0.800 |
| affine oracle@2px@64 | 0.198 | 0.692 | 0.600 |

### 关键发现

1. **Base 已经是最优 crop 中心**：base center_error_median 在所有子集上都是最低的（4.54/22.2/46.5 vs 先验的 5-58px）

2. **运动先验让中心更差，不是更好**：affine 在 base>16 上比 base 差 21.7%，velocity 更差（39.8 vs 22.2px）

3. **Local DINO search 确实有效**：oracle@2px@64=0.692 for base>16 (translation/affine) — 说明 crop 内搜索能找到好候选

4. **但这不需要运动先验**：base 自己作为中心就达到 0.615 oracle@2px@64。先验只增加 7.7 个百分点（0.615 → 0.692）

5. **Translation 和 affine 几乎相同**：affine 经常 fallback 到 translation（因为相似变换不稳定）

6. **Escort 点充足**：median=16, frac<4=0.0 — 邻居点不是瓶颈

### Affine 回退分析

affine 经常回退到 translation，说明相似变换拟合不稳定。当邻居点数量有限且空间分布不均匀时，affine 比 translation 没有显著优势。

## 失败原因

1. **Base tracker 太好**：93% 的 triggered+visible 样本 base_error < 16px。运动先验没有改善空间。

2. **运动先验本身不准确**：在 base>16 子集上，affine center_error_median=27px vs base=22px。先验比 base 更差。

3. **根本矛盾**：当 base 好时不需要 recovery；当 base 差时（base>16），运动先验也不够准确。

## 6 个必须回答的问题

1. **neighbor_affine_center 是否比 base_center 更接近 GT？**
   → 否。Base 在所有子集上都是更准确的中心。

2. **它是否显著提高 GT in crop@64？**
   → 否。Affine 在 base>16 上 GT@64=0.846 vs base=0.923，更差。

3. **它是否在 base>16 上重新制造了 local oracle gap？**
   → 部分。translation/affine oracle@2px@64=0.692 vs base=0.615。增加了 7.7 个百分点，但这是先验作为中心 + DINO local search 的综合效果，不是先验的独立贡献。

4. **如果有，gap 大小够不够进入下一阶段？**
   → 不够。Stage-0 标准要求 center improvement ≥20% 或 GT@64 提升 ≥0.15，两者都失败。

5. **失败样本的主要原因是什么？**
   → 运动先验在长遮挡后预测不准确。Base tracker 已经通过全局跟踪给出了更好的位置估计。

6. **下一步应该是 zero-shot local recovery 还是直接停止？**
   → 不停止 online 线。停止 motion-prior center，进入 base-centered local recovery 审计。

## 工件清单

- `outputs/local_motion_prior_audit_stage0/summary.json`
- `outputs/local_motion_prior_audit_stage0/per_sample.jsonl`
- `outputs/local_motion_prior_audit_stage0/per_sequence.json`

## 最终结论

**Motion-prior branch 停止。** 但 online recovery 线未终止——base-centered local candidate generation 是新的唯一存活路线。

被证伪的路线：
- template-bank 全帧搜索（oracle gap = 0）
- motion-prior center（比 base 更差）

存活的路线：
- **base-centered local DINO search**：base>16 oracle@2px@64 = 0.615, base>32 = 0.800

下一步：实现 `scripts/audit_base_centered_local_recovery.py`，验证 base-centered local top1 / selective top1 是否可以直接改善 recovery。
