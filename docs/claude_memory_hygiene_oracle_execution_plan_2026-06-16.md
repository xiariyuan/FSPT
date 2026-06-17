# Claude 执行计划：Track-On2 Memory Hygiene Oracle 实验（2026-06-16）

这份文档是直接可执行的 checklist。目标不是继续 brainstorm，而是把 `Memory Hygiene` 主线变成一套可复现的诊断实验。

---

## 0. 总目标

验证下面这个核心假设是否成立：

> `Track-On2` 的 long-occ failure，关键瓶颈之一来自遮挡期间的 state contamination，而不是缺少一个后处理 reranker。

如果成立，下一阶段才值得继续做：

- predicted visibility gate
- soft retention
- persistent anchor / dual-memory
- verifier-guided gate

如果不成立，立刻停止在这条线上追加复杂度，转向：

- re-acquisition / re-detection
- Track-On-R
- TAPNext++
- 更强 base tracker

---

## 1. 本轮绝对不能做错的事

### 1.1 不要把“冻结 / 不写 memory”当作正式 oracle 实验

原因：

- train / inference 分布偏移会污染结论
- 如果没提升，你无法证明是 memory 假设错了，还是实现方式不对

### 1.2 正式 oracle 诊断必须优先走 `temporal_mask` 通路

实现原则：

- 仍然维持 rolling append
- GT visible 的新 slot：`mask=False`
- GT invisible 的新 slot：`mask=True`

换句话说：

- 不是“不写”
- 而是“写入但不让后续 attention 读到它”

### 1.3 不允许混协议

本轮所有实验都固定在：

- dataset: `TAP-Vid DAVIS`
- query protocol: `first`
- metric space: `input 256x256`
- checkpoint: `baselines/track_on/checkpoints_trackon2_dinov3.pt`
- config: `baselines/track_on/config/test.yaml`
- support grid: `20`
- `M_i = 24`

不要把这一轮结果和 `M_i=72`、`strided+original`、Kinetics、Attempt 0 主协议混写。

---

## 2. Phase A：环境与 baseline 口径锁定

### A1. 恢复 Track-On2 默认 baseline 行为

目标：

- 保证 repo 中默认 `Predictor` / `eval.py` 仍然对应原始 baseline
- exploratory selective-write 不能成为默认路径

要求：

- 给 memory update 加一个显式 policy 开关
- 默认值必须是 `unconditional`
- exploratory selective path 只能在显式指定时启用

验收：

- `baselines/track_on/evaluation/eval.py` 在默认 config 下仍可作为 baseline 入口
- `config/test.yaml` 中明确写出 `memory_update_policy: unconditional`

### A2. 锁定 DAVIS 评估口径

确认并记录：

- DAVIS eval 下 `M_i` 实际值
- `TAPVid` 数据读取路径
- `support_grid_size`
- query mode

必须写入 manifest，不能只靠记忆。

产物：

- `outputs/memory_hygiene_oracle_2026-06-16_smoke/manifest.json`

---

## 3. Phase B：实现独立诊断脚本

新增脚本：

- `scripts/trackon2_memory_hygiene_diagnostic.py`

这份脚本必须绕开当前 `Predictor.forward()` 的默认 memory 更新路径，原因是：

- 我们需要在同一 checkpoint / 同一 frame features / 同一 query set 下精确控制 memory policy
- 不能把工程修复分支和正式诊断混在一起

### B1. 脚本职责

脚本需要手动维护：

- `q_init`
- `point_memory`
- `temporal_mask`
- `tracking_to_original`
- support grid 注入

并支持以下四个条件：

1. `baseline`
2. `oracle_mask`
3. `anchor_only`
4. `oracle_mask_anchor`

### B2. 四个条件的精确定义

#### `baseline`

- 所有 active queries 每帧都 roll + append
- 新 slot 一律 `mask=False`

#### `oracle_mask`

- 仍然 roll + append
- 对原始 query：
  - GT visible -> `mask=False`
  - GT invisible -> `mask=True`
- support grid query 始终按 baseline 处理

#### `anchor_only`

- `slot 0` 作为 persistent anchor
- query 初始化时把 `q_init` 写入 `slot 0`
- 之后只对 `slots[1:]` 做 roll + append
- 新 slot 一律 `mask=False`

#### `oracle_mask_anchor`

- anchor_only
- 再叠加 oracle_mask

### B3. 输出格式

脚本至少输出：

- `metrics_summary.json`
- `per_video_metrics.json`
- `summary.md`
- `manifest.json`

不要只 print。

---

## 4. Phase C：指标实现

### C1. overall 指标

每个条件必须输出：

- `AJ`
- `OA`
- `delta_avg`
- `<4px`

### C2. long-occ 子指标

定义：

- `max_reappearance_occlusion_run >= 20`

输出：

- `long_occ_AJ`
- `long_occ_delta_avg`
- `long_occ_<4px`
- `num_long_occ_queries`

### C3. re-entry 指标

对每个 query：

1. 找到 query frame 之后最长的连续 occlusion run
2. 该 run 必须后接一个 visible frame
3. 该 visible frame 定义为 `reentry_frame`

输出：

- `reentry_mean_error_px`
- `reentry_median_error_px`
- `reentry_<4px`
- `reentry_<8px`

### C4. 分桶

至少输出以下 bucket：

- `20-39`
- `40-71`
- `72+`

如果 `M_i=24` 已固定，也可以额外补：

- `20-23`
- `24-39`
- `40+`

---

## 5. Phase D：首轮 smoke

### D1. smoke 范围

先跑：

- `max_videos = 2`
- 四个条件全部跑完

目标不是拿最终数值，而是验证：

- 脚本正确
- 四条件都能出结果
- 指标计算正常
- support grid / query 映射 / anchor slot 没有 shape bug

### D2. smoke 验收标准

必须满足：

1. 四个条件全部成功运行
2. `metrics_summary.json` 可解析
3. `per_video_metrics.json` 中每个视频都有四个条件
4. `anchor_only` 和 `oracle_mask_anchor` 不出现 query 数错位
5. `oracle_mask` 不改变 overall query 集合

如果 smoke 失败，先修脚本，不要急着跑 full DAVIS。

---

## 6. Phase E：full DAVIS

只有 smoke 通过后才允许继续。

full DAVIS 运行要求：

- 同一命令模板
- 只把 `max_videos` 从 `2` 提到 `0` 或 `30`
- 不改任何协议项

full 输出目录建议：

- `outputs/memory_hygiene_oracle_2026-06-16_full/`

产物至少包括：

1. `manifest.json`
2. `metrics_summary.json`
3. `per_video_metrics.json`
4. `summary.md`
5. `decision.md`

---

## 7. 成功 / 失败判断标准

### 7.1 认定 `oracle_mask` 成功

满足以下任意两个：

1. `long_occ_AJ >= +2.0`
2. `long_occ_delta_avg >= +2.0`
3. `reentry_mean_error_px` 下降至少 `10%`
4. 至少 `70%` 的视频在 `long_occ_AJ` 上同号改善

### 7.2 认定这条线暂时不值得继续

如果：

- `long_occ_AJ < +0.5`
- `reentry_mean_error_px` 基本不变
- `anchor_only`、`oracle_mask`、`oracle_mask_anchor` 都没有稳定增益

则本轮结论应写成：

> Memory contamination 不是当前 Track-On2 long-occ 的主要可用 headroom。

不能硬解释。

---

## 8. 本轮之后才允许做的事

只有在 `oracle_mask` 成功之后，才继续：

1. `predicted_visibility_gate`
2. `soft retention`
3. `persistent anchor + learned gate`
4. `verifier-guided gate`

### 8.1 predicted gate 第一版要求

不要直接做 hard threshold gate。

优先方向：

```text
memory_new = alpha * q_new + (1 - alpha) * memory_old
alpha = f(v_logit, u_logit)
```

因为 hard gate 容易死锁。

---

## 9. 强制输出清单

本轮必须交付：

1. `docs/memory_hygiene_experiment_review_2026-06-16.md`
2. `docs/claude_memory_hygiene_oracle_execution_plan_2026-06-16.md`
3. `scripts/trackon2_memory_hygiene_diagnostic.py`
4. `outputs/memory_hygiene_oracle_2026-06-16_smoke/manifest.json`
5. `outputs/memory_hygiene_oracle_2026-06-16_smoke/metrics_summary.json`
6. `outputs/memory_hygiene_oracle_2026-06-16_smoke/summary.md`

如 smoke 通过，再继续 full DAVIS。

---

## 10. 给 Claude 的一句话任务定义

先在不改变 DAVIS baseline 协议的前提下，实现一个独立的 Track-On2 Memory Hygiene 诊断脚本，严格对照 `baseline / oracle_mask / anchor_only / oracle_mask_anchor` 四个条件，优先用 `temporal_mask` 做 oracle diagnostic，先跑 2 个视频 smoke，产出 manifest、metrics_summary、per_video_metrics、summary.md；只有 smoke 通过后再跑 full DAVIS，并按 long-occ / re-entry 指标决定是否继续投入 predicted gate / dual-memory。
