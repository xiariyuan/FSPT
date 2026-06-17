# Memory Hygiene 实验设计评审（2026-06-16）

## 1. 结论先行

当前 `Track-On2 + DINOv3` 的 Memory Hygiene 主线值得继续，但必须从“战略叙述”升级成“可执行诊断协议”。

最关键的修正有五条：

1. `oracle gate` 不能再用“冻结 / 不写 memory”作为主实现。
2. 正式诊断应优先走 `temporal_mask` 通路，避免 train/inference 分布偏移污染结论。
3. 评估必须做严格单变量对照，不允许混入 query protocol、分辨率、checkpoint、support grid 差异。
4. 指标不能只看 overall DAVIS；必须以 `long-occ` 和 `re-entry` 为主。
5. 不能只做线性 ablation；必须至少跑 `2x2` 因子矩阵：`baseline / oracle_mask / anchor_only / oracle_mask+anchor`。

---

## 2. 当前共识

### 2.1 已收敛的主线

当前最稳的表述是：

> 长遮挡 point tracking 的关键瓶颈之一，不是缺少一个后处理 reranker，而是遮挡期间 tracker 持续把低置信状态写入 memory，导致 state contamination；因此应优先验证 visibility-aware memory retention，并将其作为 long-occ / re-entry 改进的切入点。

### 2.2 已高置信关闭的方向

以下方向不再作为主线继续投入：

- CoTracker3 后处理 recovery / rerank
- local selector / geometry rerank
- direct recovery attention / DiReCT
- DINO patch rerank
- neighbor-deviation pseudo-label filtering
- RouteA integrated relocal / verifier 作为主方法

这些路线的问题一致：作用在输出空间，无法修复 tracker 内部状态已经被污染这一根因。

---

## 3. 代码层面的根因判断

当前审查聚焦于：

- [baselines/track_on/model/trackon.py](/gemini/code/FSPT/baselines/track_on/model/trackon.py)
- [baselines/track_on/model/trackon_predictor.py](/gemini/code/FSPT/baselines/track_on/model/trackon_predictor.py)

旧逻辑的问题是：

1. query 一旦激活，后续每帧都会生成新的 `q_t`
2. `point_memory` 采用 FIFO 滚动写入
3. 遮挡帧也照写
4. 长遮挡下错误特征会持续覆盖历史 memory
5. re-entry 时 decoder 依赖的是已经污染过的状态

这是一类 `state contamination` 问题，不是普通的后处理候选选择问题。

---

## 4. 为什么不能直接用“冻结 / 不写”

这是本轮评审里最重要的修正。

如果只在推理时把 memory 写入规则改成“不可见就不写”，那么一旦 `oracle gate` 没提升，结论会不可信，因为你无法区分：

- `memory contamination` 不是主瓶颈
- 还是模型根本没在训练中见过这种 memory dynamics，提升被分布偏移吃掉了

所以：

> “冻结 / 不写”可以作为工程探索，但不应该作为正式 oracle 诊断的主实现。

更合理的做法是：

- 仍然维持 rolling append 的时间结构
- 但对 GT 不可见帧新写入的 slot 直接打 `temporal_mask=True`
- 让 `memory_attention` 在后续帧中忽略这类污染写入

这样更接近训练时已经见过的 `p_mask` 通路，分布偏移更小，诊断结论更可信。

---

## 5. 正式诊断协议

### 5.1 严格对照要求

以下条件必须固定：

- 同一 checkpoint：`baselines/track_on/checkpoints_trackon2_dinov3.pt`
- 同一 config：`baselines/track_on/config/test.yaml`
- 同一 dataset：`TAP-Vid DAVIS`
- 同一 query protocol：`first`
- 同一评估分辨率：`input 256x256`
- 同一 support grid：`20x20`
- 同一 `M_i`：先固定为 `24`

备注：

- 当前 [baselines/track_on/evaluation/eval.py](/gemini/code/FSPT/baselines/track_on/evaluation/eval.py) 在 `dataset_name == "davis"` 时会把 `M_i` 强制设为 `24`。
- 因此所有这轮 DAVIS 诊断都必须以 `M_i=24` 为口径，不允许继续把这一轮结果写成 `M_i=72` 的 story。

### 5.2 第一阶段必须跑的 2x2 因子矩阵

1. `baseline`
   - 原始 rolling FIFO
   - 所有 active query 每帧都写入 memory

2. `oracle_mask`
   - 保持 rolling append
   - GT visible 的新 slot 设为 `unmasked`
   - GT invisible 的新 slot 设为 `masked`

3. `anchor_only`
   - 保留 baseline rolling 行为
   - 但固定一个 persistent anchor slot，不参与 FIFO 滚出

4. `oracle_mask_anchor`
   - 同时启用 `oracle_mask` 和 `persistent anchor slot`

### 5.3 support grid 口径

由于 support grid 不是 GT 标注 query：

- support grid query 在四个条件下保持相同更新规则
- 不参与 long-occ / re-entry 指标统计

这样能避免因为 support grid 缺 GT visibility 而人为引入不一致。

---

## 6. 指标要求

### 6.1 必报指标

每个条件都必须输出：

- overall `AJ`
- overall `OA`
- overall `delta_avg`
- overall `<4px`

### 6.2 主判断指标

不能只看 overall。第一优先级必须是：

- `long_occ_AJ`
- `long_occ_delta_avg`
- `long_occ_<4px`
- `reentry_mean_error_px`
- `reentry_median_error_px`
- `reentry_<4px`
- `reentry_<8px`

### 6.3 分桶指标

至少按 occlusion length 分桶：

- `20-39`
- `40-71`
- `72+`

如果 `M_i=24` 这轮先固定，则建议同时输出：

- `20-23`
- `24-39`
- `40+`

因为这能直接观察“滚动记忆长度附近”是否出现结构性断崖。

### 6.4 显著性门槛

为避免 DAVIS 只有 30 个视频导致的主观解释，建议预注册如下成功标准：

满足以下任意两个，才认定 `oracle_mask` 成功：

1. `long_occ_AJ >= +2.0`
2. `long_occ_delta_avg >= +2.0`
3. `reentry_mean_error_px` 下降至少 `10%`
4. 至少 `70%` 的视频在 `long_occ_AJ` 上同号改善

如果：

- `long_occ_AJ < +0.5`
- 且 `reentry_mean_error_px` 基本不变

则这条线应视为没有足够 headroom。

---

## 7. 工程实现建议

### 7.1 不要直接复用当前 Predictor 的 selective-write patch 作为正式诊断

当前主干里已经存在 exploratory selective-write 改动，但它不适合作为正式 oracle 诊断实现，原因有二：

- 它走的是“是否写入”而不是“写入后是否被 mask”
- 它会把工程修复和因果诊断混在一起

因此正式诊断建议单独写脚本，手动维护：

- `q_init`
- `point_memory`
- `temporal_mask`
- `tracking_to_original` 映射

### 7.2 anchor slot 的最小实现

第一轮不必改 architecture，只需做一个低风险 proxy：

- 固定 `slot 0` 作为 persistent anchor
- query 初始化时把 `q_init` 写入 `slot 0`
- 后续 FIFO 只滚动 `slots[1:]`

这不是最终论文方法，但足以回答：

> 长遮挡问题里，“anchor 被 roll 出”到底是不是独立因素。

---

## 8. 本轮建议产物

至少应产出：

1. `docs/claude_memory_hygiene_oracle_execution_plan_2026-06-16.md`
2. `outputs/memory_hygiene_oracle_2026-06-16_smoke/manifest.json`
3. `outputs/memory_hygiene_oracle_2026-06-16_smoke/metrics_summary.json`
4. `outputs/memory_hygiene_oracle_2026-06-16_smoke/summary.md`
5. 如 smoke 通过，再产出 full DAVIS 版本的同名目录

不允许只留下终端日志。

---

## 9. 结论

当前最值得继续做的不是再造一个 reranker，而是先把 Memory Hygiene 这条线变成一个可证伪、可量化、可归因的诊断协议。

最重要的判断标准只有一个：

> `oracle_mask` 在严格单变量条件下，是否真的给 `long-occ / re-entry` 带来稳定 headroom。

如果答案是“是”，再继续做 predicted gate / soft retention / dual-memory。

如果答案是“否”，就不要在这条线上继续堆复杂度，而应转向 re-acquisition / stronger base / Track-On-R / TAPNext++。
