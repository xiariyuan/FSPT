# PRT Build Status

## Completed

### 1. Benchmark definition

已新增：

- `docs/prt_benchmark_definition.md`

内容包括：

- `PRT` 问题定义
- `PRT-Synth` / `PRT-Real`
- query 定义
- `off-screen` vs `in-frame occlusion`
- 必要分层
- 核心指标
- baseline ladder

### 2. Split builder

已新增：

- `scripts/build_prt_splits.py`

当前功能：

- 从 `PointOdyssey` 提取 `re-entry` queries
- 计算 `occ_length`
- 计算 `camera_motion`
- 区分：
  - `in_frame_occlusion`
  - `offscreen_return`
  - `mixed`
- 输出：
  - `stats_output`
  - `preview_output`

### 3. Smoke run completed

命令：

```bash
python scripts/build_prt_splits.py \
  --data-root /gemini/code/FSPT/datasets/pointodyssey \
  --splits val \
  --min-occ-length 10 \
  --min-camera-motion 0.0 \
  --max-sequences-per-split 2 \
  --stats-output /gemini/code/FSPT/outputs/prt_split_stats_smoke.json \
  --preview-output /gemini/code/FSPT/outputs/prt_split_preview_smoke.json \
  --preview-per-type 3
```

结果文件：

- `outputs/prt_split_stats_smoke.json`
- `outputs/prt_split_preview_smoke.json`

Smoke 结果摘要：

- `2` 个 `val` 序列
- `262,054` 条 `PRT` queries
- `in_frame_occlusion = 188,574`
- `offscreen_return = 73,480`
- `occ_gte_20 = 174,160`
- `cam_gte_0.30 = 37,607`

说明：

- `PRT` query 定义是可执行的
- `off-screen return` 在 PointOdyssey 中样本量充足
- 足以支持后续 hardest split 的 benchmark 构建

---

## Next Recommended Step

下一步不要继续做模型，而是先把完整统计跑出来。

建议命令：

```bash
python scripts/build_prt_splits.py \
  --data-root /gemini/code/FSPT/datasets/pointodyssey \
  --splits train,val,test \
  --min-occ-length 10 \
  --min-camera-motion 0.0 \
  --stats-output /gemini/code/FSPT/outputs/prt_split_stats.json \
  --preview-output /gemini/code/FSPT/outputs/prt_split_preview.json \
  --preview-per-type 20
```

跑完之后应立即整理：

1. overall query count
2. `offscreen_return` 占比
3. `occ20+cam0.30` hardest split 样本量
4. `offscreen_return + occ20+cam0.30` 样本量
5. train/val/test 各 split 的分布差异

---

## After Full Split Stats

只有在完整 split 统计出来之后，才建议进入：

- `scripts/build_prt_candidate_dataset.py`

它应构建：

- baseline re-entry point
- top-k causal candidates
- re-entry 后短窗口
- GT choice:
  - `baseline`
  - `best_candidate`
  - `abstain`

这将作为 `temporal verifier` 的直接输入。
