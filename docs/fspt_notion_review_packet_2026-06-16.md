# FSPT 项目结构与当前方向审阅（2026-06-16）

## 0. 使用说明

这份文档是给 Notion 审阅准备的 Markdown 主文档，目标是把当前项目最值得审阅的内容浓缩成一份可读、可追溯、不过度膨胀的结构化材料。

范围控制如下：

- 保留当前主线：`Track-On2 + DINOv3 local backbone + Attempt 0 unified evaluation`
- 保留几条已经高置信关闭的失败路线，以及对应代码/结果入口
- 保留当前最重要的工件与 blocker
- 不展开全仓历史分支，不包含数据集、权重和大体量训练缓存

事实边界：

- `Attempt 0` 当前仍是 **partial**，不是 full final ranking
- `first + input` unified bridge 已闭环
- `strided + original` 当前只完成 **single-video smoke**
- CoTracker3 bridge 的最终根因修复位于 `datasets/metrics.py`，不是 exporter bug

---

## 1. 项目目标

FSPT 当前聚焦的问题是：

1. 长遮挡后的 point re-entry 恢复
2. first-frame anchored point tracking 的稳定性
3. 遮挡、重现、长时间 drift 这些困难事件上的结构性改进

过去这一轮工作的核心教训已经很明确：

> 不要继续在 CoTracker3 的后处理空间里反复加 patch，而应该把变化层级抬到更强 base tracker、训练范式、或者多 teacher / verifier 机制。

---

## 2. 当前最重要结论

### 2.1 当前 runnable baseline 排名

目前可运行并完成 repo-native 与 unified bridge 对齐的 baseline 共有 3 条：

| Baseline | DAVIS AJ | DAVIS delta_avg | DAVIS OA | 当前结论 |
|---|---:|---:|---:|---|
| `trackon2_dinov3` | 67.04 | 79.84 | 92.09 | 当前最强 runnable baseline |
| `cotracker3_baseline` | 64.89 | 77.36 | 91.80 | repo-native 与 unified 完全对齐 |
| `cotracker3_offline` | 62.66 | 77.22 | 88.15 | repo-native 与 unified 完全对齐 |

### 2.2 当前 Attempt 0 的真实状态

- `Attempt 0` 当前只能写成 **Partial Repo-Native Decision**
- 不能写成 full unified final ranking
- `Track-On-R`、`TAPNext++`、`AllTracker` 仍被 checkpoint / 资产阻塞
- `first + input` unified bridge 已经对 3 条 runnable baseline 实现 `0.00 diff`
- `strided + original` 目前只在单视频 smoke 上跑通完整链路

### 2.3 当前最重要的工程结论

1. **DINOv3 国内源路线已跑通到本地加载与 Track-On2 注入**
2. **Track-On2 当前是最值得保留的 recovery 主 teacher candidate**
3. **统一指标桥接的核心 bug 已修复**
4. **当前最合理的新主线不是继续修 CoTracker3 后处理，而是换基座/换训练路线**

---

## 3. 已排除 / 高置信失败路线

下面这些方向已经不值得继续追加资源，除非问题定义本身改变。

### 3.1 CoTracker3 后处理 recovery / rerank

已排除方向包括：

- local selector
- local geometry rerank
- DINO patch rerank
- DiReCT attention
- current-data direct recovery head
- base-centered local recovery

统一结论：

- oracle ceiling 可以看到
- 但真实可学习 ranker 无法泛化到真正有效的正样本选择
- 在 CoTracker3 已学好的特征空间上做后处理，信号不足以稳定超过现有 `pred_visibility` 或原始 tracker 行为

代表性文档：

- `docs/local_geometry_rerank_status_2026-06-12.md`
- `docs/direct_recovery_attention_audit_v1_1_status_2026-06-12.md`
- `docs/online_recovery_status_report_2026-06-10.md`
- `docs/online_recovery_gate_m1_status_2026-06-11.md`
- `docs/online_recovery_head_m0_status_2026-06-10.md`

代表性代码：

- `scripts/audit_local_geometry_rerank.py`
- `scripts/audit_direct_recovery_attention.py`
- `scripts/debug_online_recovery_smoke.py`
- `scripts/train_online_recovery_head.py`
- `scripts/train_online_recovery_gate.py`
- `models/online_recovery_head.py`
- `models/recovery_features.py`
- `models/cotracker_refiner.py`

代表性结果：

- `outputs/local_geometry_rerank_audit/results.json`
- `outputs/direct_recovery_attention_audit_v1_1/results.json`
- `outputs/online_recovery_real_eval_audit.json`
- `outputs/online_recovery_learned_head_smoke_audit.json`
- `outputs/base_centered_local_recovery_audit/summary.json`

### 3.2 Neighbor-deviation pseudo-label filtering

这条线的关键结论已经非常稳定：

- 在 **GT tracks 分析** 上，neighbor deviation 是强 signal
- 但在 **predicted tracks / pseudo-label filtering** 上，它输给更直接的 `pred_visibility`
- 所以它可以保留为分析贡献，但不能继续作为训练过滤主线

代表性文档：

- `docs/neighbor_deviation_deep_analysis_v1_1_status_2026-06-13.md`
- `docs/predicted_neighbor_deviation_signal_v1_status_2026-06-13.md`

代表性代码：

- `scripts/analyze_neighbor_deviation_deep.py`
- `utils/neighbor_deviation.py`

代表性结果：

- `outputs/neighbor_deviation_deep_analysis_v1_1/summary.json`
- `outputs/neighbor_deviation_deep_analysis_v1_1/pseudo_label_filtering_stats.json`
- `outputs/predicted_neighbor_deviation_signal_v1/summary.json`

### 3.3 Reliability / calibration 作为主线

这条线不是完全没价值，而是价值形态已经确定：

- 有分析价值
- 有论文辅助叙事价值
- 但不是 AJ / OA / delta_avg 的主指标提升方法

也就是说，它更像“诊断层”和“分析层”，不是下一阶段的主要工程下注点。

### 3.4 RouteA integrated relocal / verifier

结论：

- AJ 增益接近 0
- 与其继续追加 patch，不如转向更强 base model 或不同训练范式

---

## 4. 当前核心 idea

### 4.1 短期工程主线

保留 `Track-On2 + DINOv3 local backbone`，原因是：

- 本地已完成 backbone 注入与 smoke
- repo-native DAVIS / Kinetics 都有可信结果
- 当前 runnable baselines 中性能最强
- 相比继续在 CoTracker3 feature space 上做修补，更像一条已经有闭环证据的主线

### 4.2 中期研究主线

最值得下注的是：

- `Track-On2 -> Track-On-R` 的多 teacher / verifier meta-selection 路线

不是因为当前仓库已经完整复现了它，而是因为：

- 当前失败的方向主要都局限在单模型后处理空间
- 而 Track-On-R 本质上是在 **base model + pseudo-label 机制 + teacher 选择机制** 三层同时换范式

### 4.3 并行备选

仍值得继续跟踪，但当前没有进入主执行面的方向：

- `TAPNext++`：长序列 + re-detection 训练范式
- `AllTracker`：dense flow 范式

它们的共同点是：

- 都不是继续在 CoTracker3 上做后处理修补
- 都更接近“问题表示变化”或“训练机制变化”

---

## 5. 当前最重要的模型结构

### 5.1 Track-On2 主体结构

核心代码：

- `baselines/track_on/model/trackon.py`
- `baselines/track_on/model/trackon_predictor.py`

结构摘要：

```text
Video
-> Backbone
-> SimpleFPN
-> Query feature initialization
-> Feature attention
-> Query attention
-> Memory attention
-> Reranking head
-> Prediction head
-> Tracks + visibility
```

#### 关键代码片段 1：主模块装配

```python
class Track_On2(nn.Module):
    def __init__(self, args, nhead=4):
        super().__init__()
        self.backbone = Backbone(args)
        self.fpn = SimpleFPN(args)
        self.reranking_head = Rerank_Module(args, nhead)
        self.prediction_head = Prediction_Head(args, nhead)

        self.feature_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)
        self.query_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)
        self.memory_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)
```

含义：

- backbone 负责视觉特征
- FPN 做 multi-scale 特征融合
- query / feature / memory 三路 attention 共同决定在线追踪状态
- reranking head 和 prediction head 是 Track-On2 的关键区别点之一

#### 关键代码片段 2：Predictor 的职责

```python
class Predictor(torch.nn.Module):
    def __init__(self, model_args=None, checkpoint_path=None, support_grid_size=20):
        ...
        self.model = Track_On2(model_args)
        ...
        self.model.memory_extension(ime_size)

    def forward_frame(self, frame, new_queries=None):
        ...
        p, v_logit, q_new = self.model.track_frame(
            self.q_init[:self.N],
            self.temporal_mask[:self.N],
            self.point_memory[:self.N],
            frame_features,
            H,
            W
        )
```

含义：

- 外层 predictor 负责 query 生命周期管理
- query 会按时间激活，不是一次性全部灌入
- memory 会在逐帧推理中持续更新
- 可选 support grid 也是在 predictor 层加入

### 5.2 verifier 相关结构

代码入口：

- `baselines/track_on/verifier/verifier.py`
- `baselines/track_on/verifier/verifier_predictor.py`

这部分当前不是本仓库最成熟的结果来源，但它对应了后续可能扩展到 Track-On-R 风格路线的结构锚点，因此值得保留给审阅者。

---

## 6. Attempt 0 评估 / bridge / rescoring 结构

### 6.1 当前统一链路

```text
repo-native predictor
-> per-video .npz cache
-> exporter -> unified .pt cache
-> validator
-> official parity audit
-> unified rescore
-> status JSON / final_decision
```

核心代码入口：

- `scripts/attempt0_export_tapvid_repo_cache.py`
- `scripts/attempt0_export_strided_original_cache.py`
- `scripts/attempt0_validate_cache.py`
- `scripts/attempt0_metric_parity.py`
- `scripts/attempt0_rescore_cache.py`
- `utils/attempt0_schema.py`
- `datasets/metrics.py`

### 6.2 当前真实状态

- `first + input` bridge：完成，3 条 baseline `0.00 diff`
- `strided + original`：单视频 smoke 完成，全量 DAVIS 未完成
- blocked baselines：`trackonr`、`tapnextpp`、`alltracker`

### 6.3 当前最关键的修复点：`datasets/metrics.py`

早先 CoTracker3 unified bridge 的 AJ gap 最终根因不在 exporter，而在统一 metric wrapper 的判断逻辑。

修复后的核心代码片段：

```python
scale_w = max(width - 1.0, 1.0)
scale_h = max(height - 1.0, 1.0)
xy_scale = torch.tensor([scale_w, scale_h], device=device, dtype=pred_tracks.dtype)

pred_tracks_xy = pred_tracks[..., [1, 0]].to(dtype=pred_tracks.dtype)
gt_tracks_xy = gt_tracks[..., [1, 0]].to(dtype=gt_tracks.dtype)

pred_tracks_px = pred_tracks_xy * xy_scale
gt_tracks_px = gt_tracks_xy * xy_scale
```

这段修复的本质是：

- unified cache 明确定义为 normalized `[y, x]`
- 不能再根据值域去猜“它是不是像素坐标”
- 因为 CoTracker3 合法轨迹可以越界，normalized 值也可能小于 0 或大于 1

### 6.4 `strided + original` exporter 的关键约束

```python
query_points_norm = np.zeros_like(query_points_px, dtype=np.float32)
query_points_norm[:, 0] = query_points_px[:, 0]
query_points_norm[:, 1] = query_points_px[:, 1] / max(orig_h - 1, 1)
query_points_norm[:, 2] = query_points_px[:, 2] / max(orig_w - 1, 1)
```

关键点：

- `query_points[:, 0]` 是 frame index，**不归一化**
- 只对空间坐标做按分辨率归一化

### 6.5 `strided + original` eval 脚本最近修复

代码入口：

- `scripts/attempt0_eval_strided_original.py`
- `scripts/attempt0_export_strided_original_cache.py`

这一轮已明确修过的问题包括：

1. `PROJECT_ROOT` 路径错误
2. Track-On2 调用签名错误
3. CoTracker3 predictor / checkpoint 组合错误
4. 视频 tensor 维度应为 `(B, T, C, H, W)`，不能保留 `(B, T, H, W, C)`
5. cache 导出与命名修复
6. unified cache 中 query frame index 保留原义

---

## 7. 当前最重要工件

### 7.1 环境与 DINOv3 接入

- `outputs/track_on_env_manifest.md`
- `outputs/dinov3_load_smoke.json`
- `outputs/fallback_decision.md`
- `outputs/trackon_local_backbone_smoke.txt`
- `docs/dinov3_modelscope_no_wait_execution_checklist_2026-06-14.md`

### 7.2 Attempt 0 恢复工件

- `outputs/attempt0_2026-06-15_recovery/final_decision.md`
- `outputs/attempt0_2026-06-15_recovery/manifests/attempt0_manifest.json`
- `outputs/attempt0_2026-06-15_recovery/status/trackon2.json`
- `outputs/attempt0_2026-06-15_recovery/status/cotracker3_baseline.json`
- `outputs/attempt0_2026-06-15_recovery/status/cotracker3_offline.json`
- `outputs/attempt0_2026-06-15_recovery/status/trackonr.json`
- `outputs/attempt0_2026-06-15_recovery/status/tapnextpp.json`
- `outputs/attempt0_2026-06-15_recovery/status/alltracker.json`

### 7.3 unified bridge 代表性工件

- `outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt`
- `outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt`
- `outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt`
- `outputs/attempt0_2026-06-15_recovery/unified_rescoring/trackon2_dinov3_davis_first_input_rescore.json`
- `outputs/attempt0_2026-06-15_recovery/unified_rescoring/cotracker3_baseline_davis_first_input_rescore.json`
- `outputs/attempt0_2026-06-15_recovery/unified_rescoring/cotracker3_offline_davis_first_input_rescore.json`
- `outputs/attempt0_2026-06-15_recovery/reports/unified_protocol_bridge_round4.md`
- `outputs/attempt0_2026-06-15_recovery/reports/round4_strided_original_feasibility.md`

---

## 8. 当前 blocker

1. `Track-On-R` 缺 checkpoint / 网络资产
2. `TAPNext++` 当前本地无 checkpoint
3. `AllTracker` 当前本地无 checkpoint
4. full `strided + original` DAVIS 尚未完成
5. 所以当前不能产出真正完整的 Attempt 0 主表排名

---

## 9. 审阅建议

对于 Notion 中的详细审阅，推荐顺序如下：

1. `docs/dinov3_modelscope_no_wait_execution_checklist_2026-06-14.md`
2. `docs/claude_attempt0_detailed_task_list_2026-06-14.md`
3. `outputs/attempt0_2026-06-15_recovery/final_decision.md`
4. `baselines/track_on/model/trackon.py`
5. `baselines/track_on/model/trackon_predictor.py`
6. `datasets/metrics.py`
7. `scripts/attempt0_*`
8. 失败路线相关状态文档与结果
9. `docs/new_direction_ranking_2026-06-13.md`

---

## 10. 最终结论

当前最准确的结论应该表述为：

- `Track-On2 + DINOv3 local backbone` 是当前最强 runnable baseline
- `first + input` unified bridge 已对 runnable baselines 全部闭环
- `strided + original` 只完成了 single-video smoke，还未完成 full DAVIS
- CoTracker3 后处理空间上的大多数修补路线已经高置信关闭
- 下一阶段最值得下注的是更强 base model / 多 teacher / 新训练范式，而不是继续在 CoTracker3 特征空间上追加 patch

