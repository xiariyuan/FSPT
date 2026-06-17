# PRT Candidate Dataset Status

## Added

### 1. Candidate dataset builder

已新增：

- `scripts/build_prt_candidate_dataset.py`

目标：

- 从 `PRT` queries 构建 temporal verifier 所需的中间缓存
- 不训练模型，只做数据层 materialization

核心输出：

- `dataset_cache.npz`
- `samples_meta.jsonl`
- `stats.json`
- `preview.json`

---

## Cache Contents

当前 `dataset_cache.npz` 已包含：

1. 单帧输入
   - `query_patch`
   - `baseline_patch`
   - `cand_patches`
   - `query_xy`
   - `gt_reentry_xy`
   - `baseline_xy`
   - `cand_xy`

2. 单帧误差与分数
   - `baseline_err`
   - `cand_err`
   - `cand_score`
   - `cand_ncc`
   - `cand_dist_norm`

3. 标签
   - `label_index`
   - `oracle_index`
   - `reentry_type_id`

4. query 元数据
   - `occ_length`
   - `camera_motion`

5. temporal window
   - `gt_window_xy`
   - `gt_window_vis`
   - `window_frames`

6. temporal tracklets
   - `baseline_track_xy`
   - `baseline_track_score`
   - `baseline_track_err`
   - `cand_track_xy`
   - `cand_track_score`
   - `cand_track_err`

---

## Label Semantics

`label_index` 的定义：

- `-1`: `abstain`
- `0`: `baseline`
- `1..K`: `best candidate index + 1`

`oracle_index` 的定义：

- `0`: baseline 在单帧误差上最好
- `1..K`: 某个 candidate 在单帧误差上最好

当前标签逻辑：

1. 如果 best candidate 比 baseline 好至少 `positive_margin_px`，则标记为 `best_candidate`
2. 否则如果 baseline 已经足够好 (`baseline_err <= abstain_threshold_px`)，则标记为 `baseline`
3. 否则标记为 `abstain`

这保证 temporal verifier 第一步学的是：

- 什么时候该接受候选
- 什么时候该回退 baseline
- 什么时候该 abstain

---

## Smoke Results

### v1

目录：

- `outputs/prt_candidate_smoke`

摘要：

- `count = 16`
- `candidate_beats_baseline_frac = 0.625`
- `label_name_counts = {best_candidate: 8, abstain: 6, baseline: 2}`

### v2 with temporal tracklets

目录：

- `outputs/prt_candidate_smoke_v2`

摘要：

- `count = 8`
- `candidate_beats_baseline_frac = 0.75`
- `label_name_counts = {best_candidate: 5, abstain: 3}`
- `baseline_track_window_median_px = 78.20`
- `best_candidate_track_window_median_px = 78.97`

解释：

- 单帧上 candidate 经常优于 baseline
- 但短窗口 tracklet 后，这个优势未自动保留
- 这正是 temporal verifier 要解决的问题

---

## Added Minimal Verifier

已新增：

- `scripts/train_prt_temporal_verifier.py`

它读取 `dataset_cache.npz`，先用轻量 temporal/statistical features 做二分类 acceptor：

- 输入：
  - 单帧误差差
  - DINO score
  - NCC
  - 距离
  - baseline/candidate 短窗口 median/mean/max error
  - baseline/candidate 短窗口 score
  - `occ_length`
  - `camera_motion`
  - `reentry_type`

- 输出：
  - 是否接受某个 candidate

这是最小可行版本，不是最终模型。

---

## Temporal Verifier Audit

### v1 (invalid)

旧版 `temporal verifier v1` 结果不能使用。

原因：

- 输入特征错误地包含了 GT 派生误差：
  - `baseline_err`
  - `cand_err`
  - `baseline_track_err`
  - `cand_track_err`

这属于 feature leakage。

因此：

- `pred_median = oracle_median`
- `better_frac = 62%`

这一组数字无效，不能进入论文主结果。

### v2 clean (valid but negative)

目录：

- `outputs/prt_temporal_verifier_v2_clean`

关键结果：

- `baseline_median_px = 37.97`
- `pred_median_px = 40.71`
- `oracle_median_px = 31.33`
- `baseline_lt4px = 2%`
- `pred_lt4px = 3%`
- `oracle_lt4px = 6%`
- `better_frac = 1%`
- `accept_rate = 11%`

解释：

- 去掉泄漏后，轻量 temporal/statistical verifier 不再有效。
- 当前 causal 特征组合：
  - `cand_score`
  - `cand_ncc`
  - `cand_dist_norm`
  - baseline/candidate track geometry
  - baseline/candidate track score stats
  - `occ_length`
  - `camera_motion`
  - `reentry_type`
  仍不足以可靠区分好/坏候选。

结论：

- `temporal verifier` 这个大方向没有被证伪。
- 但“轻量统计特征 + 小 MLP acceptor”这条实现路线已经失败。

---

## Next Recommended Commands

### 1. Build train cache

```bash
python scripts/build_prt_candidate_dataset.py \
  --data-root /gemini/code/FSPT/datasets/pointodyssey \
  --splits train \
  --reentry-types in_frame_occlusion,offscreen_return \
  --min-occ-length 20 \
  --min-camera-motion 0.30 \
  --max-sequences-per-split 2 \
  --max-samples 256 \
  --topk 5 \
  --post-window 4 \
  --output-dir /gemini/code/FSPT/outputs/prt_candidate_train_smoke
```

### 2. Build val cache

```bash
python scripts/build_prt_candidate_dataset.py \
  --data-root /gemini/code/FSPT/datasets/pointodyssey \
  --splits val \
  --reentry-types in_frame_occlusion,offscreen_return \
  --min-occ-length 20 \
  --min-camera-motion 0.30 \
  --max-sequences-per-split 1 \
  --max-samples 128 \
  --topk 5 \
  --post-window 4 \
  --output-dir /gemini/code/FSPT/outputs/prt_candidate_val_smoke
```

### 3. Train minimal temporal verifier

```bash
python scripts/train_prt_temporal_verifier.py \
  --train-cache /gemini/code/FSPT/outputs/prt_candidate_train_smoke/dataset_cache.npz \
  --val-cache /gemini/code/FSPT/outputs/prt_candidate_val_smoke/dataset_cache.npz \
  --output-dir /gemini/code/FSPT/outputs/prt_temporal_verifier_smoke \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3
```

---

## Decision Rule

当前已经满足下面的失败条件：

- 降低 `pred_median_px`
- 提高 `<4px`
- 让 `better_frac > 0.5`

因此下一步就不是继续调这个轻量 acceptor，而是：

- 上更强的 temporal sequence model
- 或把论文收敛为 benchmark + oracle/noisy-depth + failure analysis
