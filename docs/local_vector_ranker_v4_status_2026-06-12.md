# Local Vector Ranker v4 Status (2026-06-12)

## 结论

`384D candidate vector` 路线已完成最小验证，结果为 **无收益**。

它不是“略有提升但不够显著”，而是：

- 与 Stage-2c 的 `scalar-score + geometry` gated selector **几乎完全相同**
- `positive final median`、`base16 final median`、`overall no-harm` 全部没有变化
- per-sample 预测与 Stage-2c 几乎一致，只改动了 `1/97` 个样本，且该改动没有带来改善

因此当前可以明确说：

- 问题不在于“只用了 scalar score，没用完整向量”
- 问题在于当前 `support_descriptor + local candidate DINO feature` 这套表示，在 top-k 内 **仍然缺少区分 oracle 与次优候选的判别信息**

## 工件

- 训练脚本：
  - `scripts/train_local_vector_ranker_v4.py`
- 结果：
  - `outputs/local_vector_ranker_v4_seed0/results.json`
  - `outputs/local_vector_ranker_v4_seed0/per_sample_predictions.jsonl`

对照基线：

- `outputs/local_selector_stage2b_from_anchor_sweep/results.json`
- `outputs/local_selector_stage2b_from_anchor_sweep/per_sample_gate_0.50.jsonl`

## 实验设置

数据：

- selector dataset: `outputs/local_selector_dataset_v3_from_anchor_smoke/`
- anchor root: `outputs/recovery_anchor_dataset_v3_full/`

模型：

1. Gate:
   - 仍沿用 Stage-2c 同类 event gate
2. Ranker:
   - positive-only
   - 输入不再是 candidate scalar score
   - 输入改为：
     - `support_descriptor` 384D
     - candidate location 对应的 384D DINO vector
     - 少量 geometry / rank metadata
   - 组合形式：
     - `cand * support`
     - `cand - support`
     - geometry metadata

## 结果

| Metric | Stage-2c baseline | Vector ranker v4 |
|---|---:|---:|
| overall final median | 11.14 | 11.14 |
| overall diff vs base | -0.09 | -0.09 |
| positive final median | 20.33 | 20.33 |
| base16 final median | 22.87 | 22.87 |
| base16 better\_2px | 0.526 | 0.526 |
| base32 final median | 62.69 | 62.69 |
| gate positive recall | 1.000 | 1.000 |
| ranker top1 acc on positive | 0.250 | 0.250 |

## 关键观察

### 1. 完整向量没有带来额外可分性

本轮最该验证的问题是：

- 如果不用 `cand_score` 这个单标量，而直接用每个候选位置的 `384D DINO vector`
- 是否能把 `positive final median` 从 `20.33` 压向 `oracle 10.67`

答案是否定的。

结果完全卡在原地：

- positive: `20.33 -> 20.33`
- base16: `22.87 -> 22.87`

### 2. per-sample 行为几乎没有变

与 Stage-2c 的样本级对比：

- `97` 个样本里，chosen candidate 只改了 `1` 个
- 该变化出现在 `breakdance`，且是负向变化

这说明：

- 即使把完整 384 维向量喂给 ranker
- 它学到的决策边界仍几乎和旧 ranker 等价

### 3. `dance-twirl` / `parkour` 仍然复现同样的错误模式

典型样本：

- `dance-twirl`
  - oracle 常在 rank `3`
  - 误差 `3.8 ~ 6.8px`
  - ranker 仍偏向 rank `1`
  - 误差 `13 ~ 19px`

- `parkour`
  - oracle 与次优都能改善
  - ranker 仍系统性偏向与 Stage-2c 相同的次优 rank

这说明：

- 当前失败不是 loss 或 MLP 容量问题
- 也不是“只看 scalar 不看 vector”问题
- 而是当前 candidate 表示本身没有把 oracle 和次优候选分开

## 结论更新

到目前为止，下面这些路线都已基本证伪：

1. scalar-score tabular ranker
2. pairwise/listwise loss on scalar/geometry features
3. full 384D candidate vector ranker

所以如果还要继续 online local selector 线，下一步就不应再做：

- 调 loss
- 调 threshold
- 调 hidden size
- 换同一批 DINO 向量的浅层 ranker

## 还剩下的合理方向

只剩两类值得继续：

1. `image patch verification`
   - 直接比较 `query_crop / support_patch / candidate_patch / baseline_patch`
   - 不再假设单点 feature vector 足够表达局部判别信息
2. `integration smoke`
   - 接受当前 selector 性能
   - 先验证在线接入后是否仍保持 overall no-harm

## 当前建议

如果目标是“继续研究可恢复性”，优先顺序应改成：

1. 做 `candidate patch verifier`
2. 如果不想再投入新模块研发，就停止 selector 研究，转做一次 `integration smoke`

不建议继续在当前 `DINO feature vector ranker` 这条线上追加时间。

