# PRT Sequence Verifier Status

## Why This Exists

`lightweight temporal/statistics acceptor` 已经在无泄漏条件下失败。

因此下一步不能再继续：

- 调 threshold
- 调 hidden dim
- 调 lr
- 再堆更多手工统计特征

必须直接升级为：

- `sequence-level verifier over candidate tracklets`

也就是让模型直接看短窗口序列，而不是只看事后汇总的统计量。

---

## Added Script

已新增：

- `scripts/train_prt_sequence_verifier.py`

这个脚本直接消费：

- `scripts/build_prt_candidate_dataset.py` 生成的 `dataset_cache.npz`

不重新跑视频推理。

---

## Causal Inputs Only

脚本会自动写出：

- `feature_audit.json`

其中明确记录：

### Allowed

1. sequence token features
   - baseline relative XY
   - candidate relative XY
   - baseline step XY
   - candidate step XY
   - candidate minus baseline XY
   - baseline track score
   - candidate track score

2. global features
   - candidate DINO score
   - candidate NCC
   - candidate distance-from-baseline
   - candidate initial gap
   - occlusion length
   - camera motion
   - reentry type onehot

3. static patch inputs
   - query patch
   - baseline patch
   - candidate patch

### Explicitly Excluded

- `baseline_err`
- `cand_err`
- `baseline_track_err`
- `cand_track_err`
- `gt_reentry_xy`
- `gt_window_xy`
- `gt_window_vis`

这些只能用于：

- supervision
- evaluation

不能作为输入特征。

---

## Model Shape

当前是最小可行的 sequence model：

1. `PatchEncoder`
   - 编码 `query / baseline / candidate` 三个静态 patch

2. `GRU over candidate tracklet tokens`
   - 看 baseline 与 candidate 在短窗口中的相对轨迹

3. `Fusion head`
   - 融合：
     - sequence embedding
     - global metadata
     - patch embeddings
     - patch cosine similarities

输出：

- 该 candidate 是否应被接受

---

## Training Target

当前仍然是 candidate-level binary accept/reject：

- label = 1，当且仅当该 candidate 是 `label_index == candidate_id + 1`
- 否则为 0

这不是最终形式，但足够回答核心问题：

> 直接看 sequence，能否比手工 temporal statistics 更好地利用 oracle gap？

---

## Recommended Smoke Run

```bash
python scripts/train_prt_sequence_verifier.py \
  --train-cache /gemini/code/FSPT/outputs/prt_candidate_train/dataset_cache.npz \
  --val-cache /gemini/code/FSPT/outputs/prt_candidate_val/dataset_cache.npz \
  --output-dir /gemini/code/FSPT/outputs/prt_sequence_verifier_smoke \
  --epochs 20 \
  --batch-size 128 \
  --lr 3e-4
```

---

## Success Criterion

如果它想证明自己比轻量 verifier 更强，至少要做到：

1. `pred_median_px < baseline_median_px`
2. `better_frac > 0.5`
3. 不只是偶然提升 `<4px`
4. 明显优于：
   - `single-frame acceptor`
   - `lightweight temporal verifier`

---

## Smoke Outcome

目录：

- `outputs/prt_sequence_verifier_smoke`

确认：

- 已生成 `feature_audit.json`
- 输入不含任何 GT error
- GT 只用于 supervision 和 evaluation

最优结果：

- `best epoch = 5`
- `pred_median_px = 39.66`
- `baseline_median_px = 37.97`
- `oracle_median_px = 31.33`
- `pred_lt4px = 2%`
- `baseline_lt4px = 2%`
- `better_frac = 11%`
- `accept_rate = 29%`

对比 `lightweight temporal verifier v2_clean`：

- `pred_median_px`: `40.71 -> 39.66`
- `better_frac`: `1% -> 11%`
- `accept_rate`: `11% -> 29%`

解释：

- sequence-level verifier 的确比轻量统计特征版更强。
- 但它仍然没有跨过最关键的门槛：
  - `pred_median_px` 仍高于 baseline
  - `<4px` 没有提升
- 因此，它还不能被视为“稳定利用 oracle gap 的成功方法”。

---

## Failure Meaning

如果 sequence-level verifier 仍失败，则说明：

- candidate 存在 oracle gap
- 但当前 causal candidate generation + short-window verification 组合仍不足以稳定利用它

那时更合理的收敛方向将是：

- `PRT benchmark`
- `world-state oracle/noisy-depth evidence`
- `candidate oracle gap`
- `failure analysis`

而不是继续做弱信号方法调参。

当前状态已经基本落在这个分支上。
