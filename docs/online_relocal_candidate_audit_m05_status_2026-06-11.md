# M0.6 Status: Top-k Oracle Audit Recheck (2026-06-11)

## 当前结论

**在当前 `10-batch / n=75 triggered queries` 的审计样本上，online top-k candidate bank 显示出 oracle value，且 CoTracker template-bank 候选明显强于 DINO。当前最合理的下一步是优先验证 CoTracker-based top-k selector。**

这还不是“最终路线结论”，因为目前证据仍来自一个小样本触发子集，而不是完整 triggered 集合。

## 关键数据

### CoTracker vs DINO online top-k oracle gap

审计范围：

- dataset: TAP-Vid DAVIS
- loader size: 30 videos
- current audit budget: `max_batches=10`
- triggered samples observed: `n=75`
- candidate kind: `template_bank`

| 指标 | CoTracker | DINO | PRT Offline |
|------|-----------|------|-------------|
| All: oracle better@2px | **21.3%** | 10.7% | 60.5% |
| BaseBad>16px: oracle better@2px | **64.0%** | 32.0% | 75.7% |
| BaseVeryBad>32px: oracle better@2px | **88.9%** | 38.9% | 76.7% |
| Pairwise dist mean | **211.4px** | 29.0px | 17.0px |
| Unique spatial modes | **3.53** | 1.83 | — |

### 解读

1. **CoTracker top-k 有大量 oracle gap**：64% 的 base_bad 和 89% 的 base_very_bad 中，top-k 里存在比 base 好 >2px 的候选
2. **但 top-1 完全不动（Type A）**：CoTracker 的 raw top-1 不离开 base，所以好候选在 top-k 里但选不到
3. **DINO top-k gap 较小**：32% base_bad，候选更集中（29px pairwise），说明 DINO search 的 peaks 不够分散
4. **PRT offline 仍然最强**：75.7% base_bad，说明 crop-based local search 策略有效
5. **当前 online 结果的候选语义不是 spatial top-k，而是 template-bank candidates**：这里的 `topk_kind=template_bank`

### 当前分叉判定

在当前小样本审计范围内，**分叉 B 暂时成立**：online candidate bank 有 oracle gap，但当前 top-1 / current pick 选不到好候选。

这里必须保留一个限定词：

- 该结论目前只在 `n=75` triggered 样本上成立
- 还没有扩展到完整 triggered 集或 sequence-grouped 稳定性层面

## 下一步路线

### 优先级 1: CoTracker-based top-k selector
- CoTracker 的 top-k 候选多样性最高（211px, 3.5 modes）
- oracle gap 最大（64%/89%）
- 但 top-1 不动 → 需要一个 selector 从 top-k 中选最优

建议执行顺序：

1. 先把 `audit_topk_oracle_gap.py` 在 full triggered 范围重跑一遍
2. 如果方向不变，再实现 CoTracker template-bank selector
3. selector 的首版目标不是全局替换，而是只在 triggered queries 内从 `K=6` 个 template-bank 候选中选一个

### 优先级 2: 如果 selector 效果不够，再考虑 DINO 或混合方案

### 不做
- 不再做 M1 gate（已被阻断）
- 当前不优先改 backbone/search head
- 当前不优先搬 offline crop-based search

但这两条只是当前优先级判断，不是永久否决。若 full-triggered audit 后结果显著缩水，需要重新打开这两个分支。

## 工件清单

- `outputs/topk_oracle_audit_cotracker_v2.json` — CoTracker online top-k audit
- `outputs/topk_oracle_audit_dino_nogate_v2.json` — DINO online top-k audit
- `outputs/prt_val_15seq_1000each_stratified/samples_meta.jsonl` — PRT offline oracle gap 复算
- `scripts/audit_topk_oracle_gap.py` — 当前审计脚本，已修正 subgroup 统计和 `(B,N,T,K)` 对齐
