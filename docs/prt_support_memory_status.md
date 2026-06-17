# PRT Support-Memory Status

## Current Direction

主线已从：

- single-frame acceptor
- lightweight temporal verifier
- sequence-level verifier

切换为：

**Persistent Support Memory for Selective Causal Re-entry Tracking**

核心假设：

- 单帧 query patch 不足以恢复长遮挡后的 point identity
- 遮挡前最后若干可见 support views 能提供更稳定的点级记忆
- 正确的决策形式应是 `baseline / candidate / abstain`，而不是单纯 accept/reject

---

## Code Added / Updated

### 1. Multi-support evaluation

文件：

- `scripts/eval_multi_support_dino.py`

新增：

- 结构化 `summary`
- `direct_selection`
- `oracle_fallback`
- `margin_abstention`
- `risk_coverage`

目的：

- 明确区分真实 direct selection 与 oracle fallback
- 为 selective prediction 主表提供可复用统计

### 2. Candidate cache with support patches

文件：

- `scripts/build_prt_candidate_dataset.py`

新增缓存字段：

- `support_patches`
- `support_frames`
- `support_count`

目的：

- 把 pre-occlusion support memory 直接固化进 `dataset_cache.npz`
- 后续训练 support-memory ranker 不需要重新裁剪视频

### 3. Minimal support-memory ranker

文件：

- `scripts/train_prt_support_memory_ranker.py`

当前最小设计：

- frozen DINOv2 patch encoder
- attention-style support pooling
- multiclass prediction over `baseline + top-k candidates`

当前不包含：

- explicit abstain class
- future-window temporal verification

原因：

- 第一阶段只验证 support memory 是否能把 direct selection 再往前推
- abstain/calibration 下一步基于 score margin 做

---

## Immediate Goal

需要先拿到两个 smoke 结果：

1. `eval_multi_support_dino.py`
   - 验证结构化 summary 输出正确
   - 确认 direct selection / margin abstention / risk-coverage 可复现

2. `build_prt_candidate_dataset.py` with `support_patches`
   - 验证新的 cache 字段能正常 materialize
   - 作为 support-memory ranker 的输入

只有这两步通过，才继续汇报 ranker smoke。
