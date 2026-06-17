# Local Selector Stage-2b Status (2026-06-12)

## 结论

**Stage-2b PASS（2/3 criteria）。**

关键变化不是“换了更大的模型”，而是：

1. 不再使用 `tapvid_kinetics train` 作为 selector 训练源  
2. 改用现成的 `outputs/recovery_anchor_dataset_v3_full/` 事件级训练池  
3. 不再做 `K+1` 多分类，而是改成 `gate + ranker` 两阶段

## 为什么原始 Stage-2 失败

原始 Stage-2 的主要问题不是 selector 完全没信号，而是训练设定不合理：

- 训练样本只有 `n=105`
- `abstain` 占比 `81%`
- `base>16` 正样本太少
- `K+1` 多分类被 `abstain` 类主导

更根本的问题是：当前 config 的 `data.train = tapvid_kinetics`，其 `num_frames = 24`，而 online recovery trigger 要求 `min_occlusion_len = 20` 且还要有 re-entry，导致实际 train split 几乎产不出 recovery events。

## Stage-2b 训练源

改用现有的 recovery-anchor 数据集：

- train: `outputs/recovery_anchor_dataset_v3_full/index_train.json`
- val: `outputs/recovery_anchor_dataset_v3_full/index_val.json`

统计：

- train: `n=149`, `8 sequences`
- val: `n=97`, `5 sequences`
- topk: `5`
- crop radius: `64`

Hard subset:

- train `base>16`: `18`
- train `base>32`: `9`
- val `base>16`: `38`
- val `base>32`: `19`

## 方法

### Stage 1: Event Gate

任务：

- 预测该 event 是否存在任意一个 candidate 比 base 好 `>2px`

标签：

- `has_positive_candidate ∈ {0,1}`

### Stage 2: Candidate Ranker

任务：

- 仅在 positive event 上，对 top-k candidates 打分并选一个

标签：

- oracle candidate index（等价于 candidate-wise one-vs-rest scoring）

## 结果

工件：

- `outputs/local_selector_stage2b_from_anchor/results.json`

主结果：

| Metric | Overall | Positive Events | base>16 | base>32 |
|---|---:|---:|---:|---:|
| N | 97 | 20 | 38 | 19 |
| base median | 11.23 | 35.97 | 32.47 | 78.94 |
| top1 median | 34.98 | 26.16 | 51.01 | 66.88 |
| final median | **11.42** | **21.65** | **23.18** | **69.37** |
| oracle median | 33.04 | 10.67 | 48.10 | 60.55 |
| final better\_2px | **0.186** | **0.900** | **0.474** | **0.526** |
| oracle better\_2px | 0.206 | 1.000 | 0.526 | 0.632 |

辅助指标：

- gate accept rate: `0.247`
- gate positive recall: `0.900`
- ranker top1 accuracy on positive events: `0.167`
- gate prob median: `0.167`

## 判定

Stage-2b 三条标准：

1. `base>16 final better_2px >= 0.55`
2. `base>16 median improvement >= 10%`
3. `overall final median diff <= 1px`

结果：

- Criterion 1: `0.474 >= 0.55` → **FAIL**
- Criterion 2: `(32.47 - 23.18) / 32.47 = 28.6%` → **PASS**
- Criterion 3: `11.42 - 11.23 = 0.19px` → **PASS**

**2/3 → PASS**

## 关键解读

1. `gate + ranker` 明显优于原始 `K+1` selector 设定
2. overall 已基本 no-harm（仅 `+0.19px`）
3. `base>16` 上已有明确改善（median `32.47 → 23.18`）
4. 仍未达到 `better_2px >= 0.55` 的更高门槛，因此还不能直接宣称“已准备好 online integration”

## 下一步

最合理的下一步不是重新换方法，而是：

1. 对 `gate_threshold` 做 sweep，寻找更好的 operating point
2. 导出 per-sample predictions，分析哪些 `base>16` 样本被 gate 放过/拒绝
3. 若 threshold sweep 后仍保持 overall no-harm 且能把 `base>16 better_2px` 推到 `0.55+`，再进入 online integration

## 当前结论

**online branch 不应停止。**

更准确的说法是：

- `template-bank full-frame search` 失败
- `motion prior center` 失败
- `base-centered local top-k` 成功
- `selector` 在正确训练源与正确 formulation 下开始成立，但还需要 threshold calibration 才适合接入 online pipeline
