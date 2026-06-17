# Local Selector Stage-2 Status (2026-06-12)

## 结论

**Stage-2 FAIL (1/3 criteria)。** 小 MLP selector 在 base>16 子集上达到 50% better_2px（通过 criterion 1），但 overall median 恶化严重（24.33 vs base 5.76），且 median improvement 仅 3.6%（< 10%）。

## 数据集

- n=105 samples, 11 sequences, K=5, r=64
- Label distribution: {0: 5, 1: 2, 2: 3, 3: 5, 4: 5, 5(abstain): 85}
- Abstain ratio: 0.810
- Oracle better_2px: 0.190 (overall), 0.667 (base>16)

### 4 个数据问题

1. **oracle_index 类别分布？** → 高度偏斜。81% 是 abstain，非 abstain 样本分散在 5 个 rank（各 2-5 个）。
2. **abstain 占比？** → 81%。训练时 class weight 约 6:1（abstain:each_rank）。
3. **正样本集中在 base>16 还是 easy 样本？** → 主要在 base>16（12/20 non-abstain samples）。
4. **sequence-grouped 切分后样本量？** → train=69 (7 seqs), val=36 (4 seqs)。val 中 base>16 仅 n=4。

## 模型

- MLP: 53 → 64 → 64 → 6 (K+1 classes)
- 输入: 8 event-level + 5×(8+1) candidate-level = 53 dim
- 训练: CE loss with class weights, Adam lr=1e-3, 100 epochs
- Split: sequence-grouped, val_frac=0.4

## 评估结果

| | base | top1 | selector | oracle |
|---|---|---|---|---|
| median (all, n=36) | **5.76** | 17.59 | 24.33 | 10.03 |
| better_2px (all) | — | 0.111 | 0.056 | 0.139 |

| | base>16 (n=4) | base>32 (n=3) |
|---|---|---|
| base median | 41.02 | 46.45 |
| selector median | 39.53 | 39.9 |
| oracle median | 26.3 | 19.68 |
| selector better_2px | **0.500** | **0.667** |

- Abstain rate: 0.194（应远高于此）
- Oracle gap utilization: 0.0

## Stage-2 判定

| # | 标准 | 结果 |
|---|------|------|
| 1 | base>16 selector better_2px ≥ 0.50 | **PASS** (0.500) |
| 2 | base>16 median improvement ≥ 10% | **FAIL** (3.6%) |
| 3 | overall median diff ≤ 1px | **FAIL** (+18.6px) |

**1/3 → FAIL**

## 失败原因

1. **数据太少**: 105 samples, 其中仅 20 个 non-abstain。train 只有 69 samples。
2. **Abstain 类主导**: 81% abstain → 模型学到"总是 abstain"比学到"正确选择"更容易。
3. **Selector 过度接受**: abstain rate 0.194 (应 ≈ 0.8)，导致 easy samples 上接受差候选，median 从 5.76 恶化到 24.33。
4. **Val 中 base>16 太少**: 仅 4 个样本，统计不稳定。

## 分析

Selector 在 base>16 子集上有信号（better_2px=0.500, base>32: 0.667），但 overall 恶化严重。问题不是"候选池不行"（oracle 强），也不是"完全没有判别信号"（base>16 上有改善），而是：

- 数据量不足以学好 abstain 决策边界
- 当前 hand-crafted tabular 特征可能不够
- 需要更多数据或更强的特征

## 5 个结论问题

1. **Local selector 是否优于 raw DINO top1？** → 否。Overall 更差。但 base>16 上略好。
2. **是否能在 base>16 上接近 oracle？** → 部分。better_2px=0.500 vs oracle=0.667 (75% utilization)。
3. **是否保持 overall no-harm？** → 否。Median 恶化 18.6px。
4. **学到的是 abstain 还是选对？** → 两者都没学好。Abstain rate 太低。
5. **下一步？** → 需要更多数据。当前 105 samples 不足以训练 selector。

## 下一步建议

当前证据不支持宣告 online branch 停止（oracle 强，base>16 上 selector 有信号），但也不支持直接做 online integration。需要：

1. 扩大 dataset：用更多 val 视频或 augmented 数据
2. 或改用 simpler 规则：如 "只在 top1_score > 0.8 AND shift > 10px 时接受"
3. 或做 feature engineering：加入 DINO feature space 中的 peak sharpness / entropy 等

## 工件

- `outputs/local_selector_dataset/` — dataset
- `outputs/local_selector_stage2/results.json` — eval results
