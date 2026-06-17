# Claude Round 4 Unified Rescoring Fix Checklist (2026-06-15)

> Update (2026-06-16): This checklist is preserved as the historical Round 4 execution plan, not the final root-cause record.
> The final verified root cause for the CoTracker3 unified-bridge mismatch is in `datasets/metrics.py`, where a `max_abs <= 1.5` heuristic misclassified normalized-but-out-of-bounds tracks as raster pixels and skipped the required `*(size - 1)` rescaling.
> Do not use this checklist as evidence that the exporter was the final bug. The authoritative post-fix status is documented in `outputs/attempt0_2026-06-15_recovery/final_decision.md`.

## 0. 目的

这不是新一轮 baseline 扩张，也不是 checkpoint 获取轮。

这一轮只做一件核心事情：

**把当前已经可运行的 3 条 baseline (`trackon2_dinov3`, `cotracker3_baseline`, `cotracker3_offline`) 从“repo-native numbers 存在，但 unified rescoring 不可信”推进到“统一导出 / 统一校验 / 统一重算至少有一条可信协议桥”的状态。**

本轮优先级固定为：

1. 先修 `Track-On2` unified cache / unified rescoring
2. 再把同样的桥接逻辑应用到 `CoTracker3 baseline / offline`
3. 最后才决定能否进一步升级到真正的 Attempt 0 主协议：`strided + original`

本轮**不做**：

1. `Track-On-R` checkpoint 获取
2. `TAPNext++` checkpoint 获取
3. `AllTracker` checkpoint 获取
4. 新模型、新训练、新后处理分支
5. 把 partial decision 冒充成 full Attempt 0 final ranking

---

## 1. 开始前必须接受的已完成事实

下面这些内容已经完成，不要重复耗时：

1. `DINOv3` 本地 backbone 路线已通过
2. `track_on` 的本地 DINOv3 注入已通过
3. `Track-On2 DINOv3` repo-native DAVIS / Kinetics smoke 已通过
4. `CoTracker3 online / offline` 本地 repo-native DAVIS 已通过
5. 当前 partial decision 已存在：
   - `outputs/attempt0_2026-06-15_recovery/final_decision.md`

当前 repo-native 结果可视为已确认事实：

| Baseline | DAVIS AJ | DAVIS delta_avg | DAVIS OA |
|---|---:|---:|---:|
| `trackon2_dinov3` | 67.04 | 79.84 | 92.09 |
| `cotracker3_baseline` | 64.89 | 77.36 | 91.80 |
| `cotracker3_offline` | 62.66 | 77.22 | 88.15 |

---

## 2. 这一轮必须接受的新诊断事实

在开始写代码前，先接受下面这些为当前工作区里的**已验证事实**。

### 2.1 当前 `trackon2` unified cache 不是可信重算输入

当前已有：

- `outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis.pt`
- `outputs/attempt0_2026-06-15_recovery/unified_rescoring/trackon2_dinov3_davis_validation.json`
- `outputs/attempt0_2026-06-15_recovery/unified_rescoring/trackon2_dinov3_davis_rescore.json`

但其中的 validation 虽然 `valid=true`，却出现了 30/30 records 的巨大 query-anchor warning。

例如：

- `bike-packing` 的 `anchor_mean_error_px ~= 210.56`
- `anchor_max_error_px ~= 410.67`

这说明：

**当前 cache 的字段语义与 Attempt 0 schema 预期不一致。**

不要再把这个 cache 的 rescore 结果当作模型真实表现。

### 2.2 当前错误更像“导出协议错位”，不是“模型性能接近 0”

已做过的最小诊断显示：

- 按 schema 当前假设直接比较：
  - `mean err ~= 210.56 px`
  - `max err ~= 410.67 px`
- 只把 `pred_tracks` 从 `[x, y]` 交换成 `[y, x]` 后再看 query anchor：
  - `mean err ~= 2.07 px`
  - `max err ~= 9.68 px`

这强烈说明：

1. 当前导出的 `pred_tracks` 很可能按 `[x, y]` 写进了 unified cache
2. 但 Attempt 0 schema / rescore 脚手架期望的是 normalized `[y, x]`

也就是说：

**现在的问题首先是 adapter/export 协议不对，不是 `Track-On2` 真跑崩。**

### 2.3 `track_on` 原生 TAP-Vid 路线与 Attempt 0 schema 的约定不同

请直接以代码为准：

1. `baselines/track_on/dataset/tapvid.py`
2. `baselines/track_on/evaluation/evaluator.py`
3. `baselines/track_on/utils/eval_utils.py`
4. `utils/attempt0_schema.py`
5. `scripts/attempt0_rescore_cache.py`

当前要明确的语义差异是：

#### `track_on` / `cotracker3` repo-native TAP-Vid 路线

1. `tracks` 原生缓存是 `(B, T, N, 2)`
2. 坐标是 raster / pixel space
3. track 坐标顺序是 `[x, y]`
4. query points 传给 metric 的格式是 `[t, y, x]`
5. `track_on` 当前 TAP-Vid dataset 默认：
   - `resize_to_256 = True`
   - `queried_first = True`

#### Attempt 0 unified schema / rescoring 路线

1. `pred_tracks` / `gt_tracks` 期望 `(N, T, 2)`
2. 坐标期望 normalized
3. 坐标顺序期望 `[y, x]`
4. `query_points` 期望 `(N, 3)` 且是 `[t, y, x]`
5. `scripts/attempt0_rescore_cache.py` 还会根据 `metric_resolution_mode` 决定用：
   - `original_size`
   - 或 `model_input_size`

### 2.4 当前 repo-native cache 默认只够做 “first + input” 协议桥

这是这轮最容易被误写错的地方。

`baselines/track_on/dataset/tapvid.py` 当前默认：

1. `queried_first = True`
2. `resize_to_256 = True`

所以当前已经落地的 `trackon2` / `cotracker3` repo-native `.npz` 缓存，本质上对应的是：

- **first-query**
- **256 input-space**

而不是 Attempt 0 主表要求的：

- **strided**
- **original resolution**

因此：

**本轮最小可交付目标是先打通 `first + input` 的可信协议桥。**

只有在这条桥完全可信后，才能决定是否继续冲真正的 `strided + original` Attempt 0 主协议。

### 2.5 当前 `outputs/track_on_unified_rescoring.json` 不是 authoritative unified result

这个文件当前更像 repo-native 记录快照，不是完整统一重算产物。

本轮不要继续沿用它作为“已完成 unified rescoring”的证据。

---

## 3. 本轮强制交付物

Round 4 结束时，至少必须交付：

1. 一份 **protocol bridge 说明文档**
2. 一套修正后的 `trackon2` unified bridge cache + validation + rescore + diff report
3. 一套修正后的 `cotracker3_baseline` unified bridge cache + validation + rescore + diff report
4. 一套修正后的 `cotracker3_offline` unified bridge cache + validation + rescore + diff report
5. 更新后的 `status/*.json`
6. 更新后的 `final_decision.md`

如果只修了代码但没有产出这些工件，算未完成。

---

## 4. 本轮执行顺序

顺序固定，不要打乱。

### Phase R4-A. 先写清楚协议桥，不要先盲改脚本

开始前先阅读：

1. `scripts/attempt0_validate_cache.py`
2. `scripts/attempt0_rescore_cache.py`
3. `scripts/attempt0_metric_parity.py`
4. `utils/attempt0_schema.py`
5. `utils/attempt0_predictions.py`
6. `baselines/track_on/dataset/tapvid.py`
7. `baselines/track_on/evaluation/evaluator.py`
8. `baselines/track_on/utils/eval_utils.py`
9. `datasets/metrics.py`

然后先写：

- `outputs/attempt0_2026-06-15_recovery/reports/unified_protocol_bridge_round4.md`

这份文档必须包含一个明确表格，至少列出：

1. 字段名
2. repo-native / cache 中的真实语义
3. Attempt 0 schema 期望语义
4. 是否需要 transpose
5. 是否需要 `xy -> yx`
6. 是否需要 pixel -> normalized
7. 归一化到底是按 `input=256` 还是按 `original_size`

这一步不做，后面大概率会继续把字段写错。

---

### Phase R4-B. 先做 `Track-On2` 的 first/input 协议桥

#### Step B1. 新建一个可复用的 exporter，不要手工拼 cache

建议新增脚本，推荐名字：

- `scripts/attempt0_export_tapvid_repo_cache.py`

功能目标：

把 repo-native `.npz` prediction cache + 对应 GT/query，导出为 Attempt 0 unified cache。

最低参数至少支持：

1. `--baseline-name`
2. `--cache-dir`
3. `--tapvid-pkl`
4. `--dataset-type`
5. `--query-protocol`
6. `--space`
7. `--out-cache`
8. `--out-report`

推荐参数语义：

1. `--query-protocol {first,strided}`
2. `--space {input,original}`

#### Step B2. 第一版只先支持当前可验证的桥：`first + input`

也就是：

1. query protocol 使用 `first`
2. resolution semantics 使用 `input`
3. input size 明确写成 `[256, 256]` 或从真实模型输入元数据读取

不要第一步就直接跳去 `strided + original`。

#### Step B3. Track-On2 exporter 的转换要求

以 `outputs/trackon2_dinov3_davis_cache/davis/trackon2/` 为输入，必须完成：

1. 读入 `.npz` 中的：
   - `tracks` `(1, T, N, 2)`
   - `visibility` `(1, T, N)`
2. 去掉 batch 维
3. 转成 Attempt 0 的 `(N, T, 2)` / `(N, T)`
4. 明确处理坐标顺序：
   - repo-native 是 `[x, y]`
   - unified schema 要写成 `[y, x]`
5. 明确处理归一化：
   - 第一轮桥接按 `input=256` 来归一化
6. `query_points` 必须写成 normalized `[t, y, x]`
7. `gt_tracks` / `gt_visibility` 必须与同一 query set 严格对齐
8. `adapter_version` / `raw_coordinate_note` / `extra_meta` 必须清楚写出：
   - `source_query_protocol=first`
   - `source_space=input256`
   - `source_track_format=xy`
   - `export_track_format=yx_normalized`

不要覆写当前坏掉的旧 cache。

必须新产出明确命名的 bridge cache，例如：

- `outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt`

#### Step B4. Track-On2 bridge cache 先跑 validator

运行：

```bash
python scripts/attempt0_validate_cache.py \
  --cache outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt \
  --require-gt \
  --out outputs/attempt0_2026-06-15_recovery/unified_rescoring/trackon2_dinov3_davis_first_input_validation.json
```

如果你发现默认的 query-anchor tolerance `2px` 对这类 predictor 太严格，可以**小范围**补一个 CLI 参数，例如：

- `--query-anchor-tol-px`

但要求是：

1. 你必须把这个修改写进脚本
2. 你必须在报告里记录实际使用的 tolerance
3. 你不能靠“忽略 warning”来糊过去

本轮对 validator 的最低验收要求：

1. `valid = true`
2. 不允许再出现 `100px+` 级别的 anchor error
3. 若仍有 warning，也必须在报告里能解释其量级合理

#### Step B5. Track-On2 bridge cache 跑 official parity

运行：

```bash
python scripts/attempt0_metric_parity.py \
  --cache outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt \
  --out outputs/attempt0_2026-06-15_recovery/unified_rescoring/trackon2_dinov3_davis_first_input_parity.json \
  --query-mode first
```

通过标准仍然是：

1. `max_abs_diff <= 1e-3`
2. `pass_threshold_1e-3 = true`

注意：

这个 parity 只证明 `datasets/metrics.py` wrapper 和 official core 一致，**不代表**已经和 repo-native number 对齐。

#### Step B6. Track-On2 bridge cache 跑 unified rescoring

运行时必须使用：

1. `--query-mode first`
2. `--metric-resolution-mode input`

示例：

```bash
python scripts/attempt0_rescore_cache.py \
  --cache outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt \
  --out outputs/attempt0_2026-06-15_recovery/unified_rescoring/trackon2_dinov3_davis_first_input_rescore.json \
  --query-mode first \
  --metric-resolution-mode input \
  --long-occ-min-run 20
```

#### Step B7. 必须写 Track-On2 的 repo-native vs unified-bridge diff report

新增：

- `outputs/attempt0_2026-06-15_recovery/reports/trackon2_davis_first_input_repo_vs_unified.json`

至少记录：

1. repo-native AJ / OA / delta_avg
2. unified bridge AJ / OA / `<avg`
3. 二者差值
4. 采用的 query protocol
5. 采用的 metric resolution mode
6. 是否可接受

Track-On2 本轮桥接的验收标准：

1. `AJ`, `OA`, `<avg` 与 repo-native 的绝对差值优先控制在 `<= 0.5`
2. 最大允许差值 `<= 1.0`

如果差值仍然远大于这个范围：

**Round 4 historical instruction**: 当时的排障顺序是先停在 Track-On2 bridge，不继续扩散到 CoTracker3。
**Update (2026-06-16)**: 这个顺序已经完成并过期。当前最终根因已经确认在 `datasets/metrics.py`，不要把这里理解成“现在还需要继续修 exporter”。 

---

### Phase R4-C. 把同样的桥接逻辑应用到 `CoTracker3`

只在 `Track-On2` first/input bridge 已经可信后再做。

输入目录分别是：

1. `outputs/cotracker3_online_davis_cache/davis/cotracker3_online/`
2. `outputs/cotracker3_offline_davis_cache/davis/cotracker3_offline/`

Attempt 0 行名映射必须保持：

1. `cotracker3_baseline = online variant`
2. `cotracker3_offline = offline variant`

#### Step C1. 导出两个 bridge cache

必须新增：

1. `outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt`
2. `outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt`

#### Step C2. 分别跑 validator / parity / rescore

必须分别产出：

1. `.../cotracker3_baseline_davis_first_input_validation.json`
2. `.../cotracker3_baseline_davis_first_input_parity.json`
3. `.../cotracker3_baseline_davis_first_input_rescore.json`
4. `.../cotracker3_offline_davis_first_input_validation.json`
5. `.../cotracker3_offline_davis_first_input_parity.json`
6. `.../cotracker3_offline_davis_first_input_rescore.json`

#### Step C3. 分别写 repo-native vs unified-bridge diff report

必须新增：

1. `outputs/attempt0_2026-06-15_recovery/reports/cotracker3_baseline_davis_first_input_repo_vs_unified.json`
2. `outputs/attempt0_2026-06-15_recovery/reports/cotracker3_offline_davis_first_input_repo_vs_unified.json`

如果某一个变体失败，不要把两个都写成 blocked。

失败必须分类记录为：

1. bridge/export path hypothesis (historical Round 4 label; not the final verified CoTracker3 root cause)
2. cache/video order mismatch
3. visibility shape mismatch
4. query set mismatch
5. resolution semantics mismatch

---

### Phase R4-D. 只有在上面 3 条桥都可信后，才允许评估 `strided + original`

这一步是“能做则做”，不是“默认已完成”。

#### Step D1. 先书面回答两个问题

在真正动手前，先在：

- `outputs/attempt0_2026-06-15_recovery/reports/round4_strided_original_feasibility.md`

里明确回答：

1. 当前已有 repo-native `.npz` cache 是否足以直接支持 true `strided` unified export？
2. 如果不够，是否必须重新跑 predictor？

目前从代码事实看，大概率答案应接近：

1. 当前 `.npz` cache 对应的是 `first-query`
2. 若要 true `strided`，大概率需要重新构造 strided queries，并重新推理

#### Step D2. 如果你决定实现 true `strided + original`

那么必须满足：

1. 不是只改文档
2. 不是只改 query_mode 参数
3. 必须真的拿到与 `strided` query set 对应的预测
4. 必须真的把坐标语义对齐到 `original resolution`

只有在这 4 条都做到后，才能把结果写进 Attempt 0 主统一表。

#### Step D3. 如果这一步做不到，正确结论是 “bridge partial”

也就是：

1. `first + input` 协议桥已完成
2. `strided + original` Attempt 0 主协议仍未完成

**不能**因为 first/input bridge 打通，就把 Attempt 0 写成 full complete。

---

## 5. 需要修改和更新的工件

### 5.1 `status/*.json`

至少更新：

1. `outputs/attempt0_2026-06-15_recovery/status/trackon2.json`
2. `outputs/attempt0_2026-06-15_recovery/status/cotracker3_baseline.json`
3. `outputs/attempt0_2026-06-15_recovery/status/cotracker3_offline.json`

更新要求：

1. 如果只完成 `first + input` bridge：
   - `rescoring_status` 不要写成 `complete`
   - 推荐写成：`partial_first_input_bridge_complete`
2. `rescoring_note` 必须明确：
   - 已完成什么
   - 未完成什么
   - true Attempt 0 `strided + original` 是否仍 pending
3. `query_mode` / `metric_resolution_mode` 必须反映本轮真实完成的协议
4. `notes` 里必须写清：
   - repo-native number
   - unified bridge number
   - 二者 diff 是否可接受

### 5.2 `final_decision.md`

更新：

- `outputs/attempt0_2026-06-15_recovery/final_decision.md`

本轮更新规则：

1. 只要 true `strided + original` 还没完成，就继续保持 **partial decision** 标题
2. 新增一节：
   - `Round 4 Unified-Bridge Status`
3. 把下面两类内容分开写：
   - repo-native confirmed numbers
   - unified bridge status
4. 不要把 first/input bridge 数字直接冒充 Attempt 0 主统一表最终数字

### 5.3 顺手清理残留误导项

当前 `final_decision.md` 的 artifact 表里仍有少量“needs update / partial”式残留描述。

本轮在确认事实后顺手清掉，但前提是：

1. 你真的核实了对应文件
2. 不要把还没完成的东西写成完成

---

## 6. 建议的最小调试命令

如果你需要快速复核当前 cache 的错位症状，可先跑类似下面的最小诊断。

### 6.1 query anchor 交换轴诊断

```bash
python - <<'PY'
import torch, numpy as np
p='outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis.pt'
payload=torch.load(p,map_location='cpu',weights_only=False)
rec=payload['records'][0]
q=np.asarray(rec['query_points'],dtype=np.float32)
pred=np.asarray(rec['pred_tracks'],dtype=np.float32)
orig=np.asarray(rec['original_size']).astype(np.int64)
qt=np.round(q[:,0]).astype(int)
anchor_pred=pred[np.arange(len(qt)), qt]
err=np.linalg.norm((anchor_pred-q[:,1:3])*np.array([orig[0]-1,orig[1]-1],dtype=np.float32),axis=1)
err_swap=np.linalg.norm((anchor_pred-q[:,[2,1]])*np.array([orig[0]-1,orig[1]-1],dtype=np.float32),axis=1)
print('mean/max as-is:', float(err.mean()), float(err.max()))
print('mean/max after swap-query-view:', float(err_swap.mean()), float(err_swap.max()))
PY
```

如果结果仍呈现：

1. as-is 几百像素
2. swap 后只有几像素

那就继续按当时的 exporter / bridge 假设做局部排查，但这只是 Round 4 的历史排障分支。
**Update (2026-06-16)**: 对 CoTracker3 这条线，最终已证实不是 exporter 根因，而是 `datasets/metrics.py` 的 normalized-coordinate rescaling heuristic。

---

## 7. 本轮停止条件

出现下面任一情况时，本轮应停止继续扩工作面，先写清阻塞：

1. `Track-On2` first/input bridge 仍然接近 0 分
2. exporter 无法证明 cache 与 video / query 顺序一一对应
3. 你发现当前 `.npz` cache 不能恢复出与 GT 对齐的 query set
4. true `strided + original` 需要完整重跑，但本轮时间不够

一旦触发停止条件，必须新增：

- `outputs/attempt0_2026-06-15_recovery/reports/round4_blocker_note.md`

写清：

1. 已完成部分
2. 未完成部分
3. 具体阻塞点
4. 下一轮应该从哪个文件继续

---

## 8. 本轮正确的成功定义

本轮“成功”有两个层级，不要混淆。

### Success Level 1: 协议桥成功

满足：

1. `trackon2`
2. `cotracker3_baseline`
3. `cotracker3_offline`

三条 baseline 至少在 DAVIS 上完成了：

1. corrected unified export
2. validation
3. official parity
4. repo-native vs unified bridge diff

并且你能清楚说出：

- 这是 **first + input** bridge，不是 full Attempt 0 主协议

### Success Level 2: true Attempt 0 main-table 成功

只有当你真的完成了：

1. `strided`
2. `original`
3. 统一导出
4. 统一重算
5. 多 baseline 可横向比较

才允许把 Attempt 0 写成真正完整主表。

在没有达到 Level 2 之前，所有文档都必须继续使用：

- `partial`
- `bridge`
- `pending`

之类的准确措辞。

---

## 9. 本轮结束时你必须回报什么

结束时不要只给终端日志。

必须明确回报：

1. 新增或修改了哪些脚本
2. 产出了哪些新工件
3. `trackon2` unified bridge 是否可信
4. `cotracker3_baseline` unified bridge 是否可信
5. `cotracker3_offline` unified bridge 是否可信
6. true `strided + original` 是否已完成
7. 当前 Attempt 0 状态到底是：
   - `repo-native partial`
   - `first/input bridge partial`
   - 还是 `full unified main table complete`

如果最后不是第 3 种，就不要写成第 3 种。
