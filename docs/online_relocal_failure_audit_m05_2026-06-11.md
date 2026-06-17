# M0.5 Final Audit Report (2026-06-11)

## 审计完成。结论：Type A（DINO similarity map peak 不在正确位置）。

## 完整失败链

1. DINO raw cosine search 产生大位移（median 65px）
2. Gate 把位移压回去（conf_threshold=0.15 >> relocal_conf=0.018）
3. 关闭 gate 后，位移确实释放了（anchor median 32.87px）
4. **但这个位移方向是错的**——anchor 比 base 更差（32.87 vs 2.67）

## 两种特征源的对比

| | CoTracker | DINO |
|---|---|---|
| Raw step | ≈ 0px（无位移信号） | ≈ 65px（有大位移） |
| Gate 作用 | N/A（无位移可压） | 把 65px 压到 ~1.6px |
| 关闭 gate 后 | anchor ≈ base | anchor median 32.87px（更差） |
| Failure type | **Type A**: map peak 在 base 处 | **Type A**: map peak 不在正确位置 |

**两者都是 Type A**：similarity map 上的 peak 位置不对应 GT。CoTracker 的 peak 在 base 位置（无信息），DINO 的 peak 在远离 base 但也不在 GT 的位置（有害信息）。

## 恢复训练门槛检查

- raw_top1 non_ambiguous_frac@2px = 100% ✓（DINO 确实产生位移）
- 但 **oracle_topk 是否存在？** → 当前只有 top-1，无法判断 top-k oracle

## 结论

**M0.5 审计结果：当前在线 recovery 的根本问题是 DINO cosine similarity map 的 peak 位置不是 GT 位置。**

这不是 gate/阈值/refinement 的问题，而是搜索策略的问题：
- 当前用 support memory 的 mean-pooled descriptor 做全局 cosine 搜索
- 这个搜索在离线 PRT 实验中有效（因为搜索范围被 crop 限制）
- 但在在线场景中，搜索范围是整帧 feature map，peak 位置不稳定

## 下一步选择

### 选项 A: 继续搜——但不急着训练

现在的关键数据还没做完：**top-k oracle gap 是否存在？** 当前 relocalization 只输出 top-1，如果把 top-k 也导出，能回答：
- 是否存在某个候选比 base 更好
- 如果有，那个候选在什么位置

如果 top-k oracle gap 存在（> 20% base_bad 子集的 oracle 候选比 base 好 >2px），说明 candidate pool 有信息量，只是 top-1 选择策略有问题 → 做 top-k selector

如果 top-k oracle gap 也不存在，说明当前搜索策略本身不够 → 需要更强的 feature 或更聪明的搜索

### 选项 B: 把结论落盘，用 PRT offline 结果投论文

当前最有价值的结论：
1. 15 序列 / 9627 样本上，support_margin 是稳定有效信号（AUC 0.653）
2. 显式 hybrid selector 在 threshold=0 时改善 median 2.31px（39.32→37.01）
3. 在线集成尚未成功——DINO cosine search peak 不在正确位置
4. 需要更强的 feature backbone 或 learned search head

这个结论对于 TCSVT 的"benchmark + diagnosis + baseline"论文是够的。

### 推荐

先落盘结论，再决定是否继续做 top-k oracle audit。
