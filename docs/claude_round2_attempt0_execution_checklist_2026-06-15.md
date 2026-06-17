# Claude Round 2 Attempt 0 Execution Checklist (2026-06-15)

## 0. 目的

这份文档是 **第二轮 Claude 执行清单**。

它建立在下面这个事实之上：

1. `DINOv3` 本地下载已完成
2. `track_on` 本地 DINOv3 backbone 已打通
3. `Track-On2 DINOv3` repo-native 复现已通过
4. 但 **Attempt 0 还没有完成**

第二轮的目标不是重新修环境，也不是重复验证 `Track-On2 DINOv3`。

第二轮只做一件事：

**把 Attempt 0 从“单条 Track-On2 复现 + hold”推进到“真正的多 baseline 统一横向表”，优先补齐 `CoTracker3` 锚点与剩余 baseline 的真实阻塞。**

---

## 1. 当前已完成状态

Claude 在开始前，必须接受下面这些为**已完成事实**，不要重复投入时间：

### 1.1 Track-On 环境已修复

已存在并可视为当前真相：

- `outputs/track_on_env_manifest.md`

当前环境实测：

1. `torch == 2.4.1+cu121`
2. `torchvision == 0.19.1+cu121`
3. `torchaudio == 2.4.1+cu121`
4. `transformers == 4.57.6`
5. `mmcv == 2.2.0`
6. `mediapy == 1.2.6`
7. `timm == 1.0.27`

不要再把时间花在 Phase A 环境修复上，除非新命令实际报依赖错误。

### 1.2 DINOv3 本地 load 已通过

已存在：

- `outputs/dinov3_load_smoke.json`

结论：

1. `AutoConfig.from_pretrained(...)` 通过
2. `AutoModel.from_pretrained(...)` 通过
3. 本地 DINOv3 目录可用

### 1.3 `track_on` 本地 backbone smoke 已通过

已存在：

- `outputs/trackon_local_backbone_smoke.txt`

结论：

1. `Predictor` import 通过
2. 模型构建通过
3. 最小 forward 通过
4. `DINOV3_LOCAL_DIR` 注入有效
5. 无 HF 网络访问

### 1.4 Track-On2 DINOv3 repo-native 已复现

已存在：

1. `outputs/track_on2_repo_native_metrics.json`
2. `outputs/track_on2_kinetics_metrics.json`
3. `outputs/trackon_adapter_sanity_report.json`
4. `outputs/trackon_metric_parity_report.json`
5. `outputs/week1_partial_decision.md`

确认结论：

1. DAVIS `delta_avg = 79.84`
2. README 参考 `79.9`
3. Kinetics 10-clip smoke `delta_avg = 68.83`
4. README 参考 `69.3`

这一步已经够了。不要再重复做 Track-On2 的 Week 1 环节。

---

## 2. 当前 Attempt 0 的真实缺口

第二轮必须明确：**Attempt 0 目前还没有闭环。**

### 2.1 Track-On2 只有 repo-native，不是 unified rescoring 完成

当前存在：

- `outputs/track_on_unified_rescoring.json`

但这个文件只是“部分记录”，里面已经明确写了：

- `attempt0_rescore_cache.py` 还没真正运行

另外：

- `outputs/attempt0_2026-06-15_recovery/status/trackon2.json`

里 `rescoring_status` 仍然是：

- `pending`

所以：

**不要把当前 Track-On2 状态误写成“已完成 unified rescoring”。**

### 2.2 `CoTracker3` 的阻塞被写得过于绝对

当前 `final_decision.md` 把 `CoTracker3` 写成：

- `torch.hub requires GitHub access`

但本地实际已经有：

1. `baselines/cotracker/`
2. `baselines/cotracker/checkpoints/scaled_offline.pth`
3. `baselines/cotracker/checkpoints/scaled_online.pth`
4. 历史 DAVIS 评测快照：
   - `outputs/cleanup_logs/baseline_refs/cotracker3_davis_result_eval_20260302.json`

这说明：

**CoTracker3 不应被直接归类为“必须联网才能做”。**

正确判断应是：

1. `track_on` 自带的 `ensemble/cotracker.py` 依赖 `torch.hub`
2. 但仓库本地已有 `baselines/cotracker` 正式代码与 checkpoint
3. 因此应优先尝试 **绕开 `track_on` 的 torch.hub wrapper**，直接用本地 `baselines/cotracker` 跑 repo-native / 导出 / 适配

### 2.3 剩余 baseline 的真实情况

当前更像下面这个状态：

1. `Track-On2 DINOv3`
   - repo-native 通过
   - unified cache / unified rescoring 未闭环
2. `Track-On-R`
   - 缺 `track_on_r.pt`
   - 缺 `verifier.pt`
3. `CoTracker3 baseline`
   - 本地代码和权重存在
   - 需要重新接进 Attempt 0
4. `CoTracker3-offline`
   - 本地代码和权重存在
   - 需要重新接进 Attempt 0
5. `TAPNext++`
   - 本地没有看到 checkpoint
6. `AllTracker`
   - 本地没有看到 checkpoint

---

## 3. 第二轮的总优先级

第二轮执行顺序固定为：

1. **先接通 `CoTracker3 baseline` 与 `CoTracker3-offline`**
2. **再把 Track-On2 补成 unified cache / validator / rescoring 真闭环**
3. **再处理 `Track-On-R` checkpoint 缺失**
4. **再处理 `TAPNext++` / `AllTracker` checkpoint 缺失**
5. **最后重写 Attempt 0 决策**

理由：

1. `CoTracker3` 是 Attempt 0 的锚点，优先级高于继续折腾 Track-On2
2. `Track-On2` 已证明可跑，不是当前最不确定的一项
3. `Track-On-R` / `TAPNext++` / `AllTracker` 当前主要是 checkpoint 获取问题

---

## 4. Phase R2-A: 接通 CoTracker3 锚点

### Step A1. 先读这两个事实源

开始前先看：

1. `docs/attempt_0_cotracker3_offline_anchor_2026-06-13.md`
2. `outputs/cleanup_logs/baseline_refs/cotracker3_davis_eval_20260302_summary.txt`

目的是明确：

1. CoTracker3-Offline 在 Attempt 0 里必须作为强公开 2D 锚点
2. 本地历史上已经跑通过至少一版 DAVIS eval 快照

### Step A2. 不要通过 `track_on/ensemble/cotracker.py` 跑 CoTracker3

这个 wrapper 当前是：

- `torch.hub.load("facebookresearch/co-tracker", ...)`

第二轮**禁止**先走这条路。

应该优先基于本地：

1. `baselines/cotracker/`
2. `baselines/cotracker/checkpoints/scaled_offline.pth`
3. `baselines/cotracker/checkpoints/scaled_online.pth`

重新接 Attempt 0。

### Step A3. 做 CoTracker3 本地 repo-native 复现

至少完成：

1. `cotracker3_baseline`
2. `cotracker3_offline`

要求：

1. 能明确说清 `baseline` 与 `offline` 各自对应的 checkpoint / config / query protocol
2. 若只能先跑通其中一个，优先 `cotracker3_offline`
3. 至少把 DAVIS 跑通

必须产出：

1. `outputs/cotracker3_baseline_repo_native_metrics.json`
2. `outputs/cotracker3_offline_repo_native_metrics.json`

如果使用历史结果作为临时参考，必须明确写：

1. 哪些是历史快照
2. 哪些是本轮重新运行

### Step A4. 做 CoTracker3 checkpoint manifest

写：

- `outputs/cotracker3_checkpoint_manifest.md`

至少记录：

1. repo path
2. checkpoint path
3. checkpoint sha256
4. 使用的 eval config
5. dataset path
6. 是否本轮重新运行
7. 是否使用历史快照辅助对照

### Step A5. 如果 CoTracker3 仍失败，分层记录阻塞

不要用一句“GitHub 不通”结束。

必须把失败分成下面几类之一：

1. 本地 repo import 失败
2. 本地 checkpoint 路径不匹配
3. eval 配置与当前数据路径不匹配
4. 运行时 CUDA / shape / protocol 错误
5. 只有 `track_on` wrapper 依赖 `torch.hub`，但本地 repo 路线未充分尝试

只在第 5 类被排除后，才允许把 CoTracker3 写成真正 blocked。

---

## 5. Phase R2-B: 把 Track-On2 补成真正的 unified 闭环

### Step B1. 明确认定当前不是 unified rescoring 完成

先在工作记录里写清楚：

1. 当前 `outputs/track_on_unified_rescoring.json` 只是部分汇总
2. 当前还没有经过 `attempt0_validate_cache.py`
3. 当前还没有经过 `attempt0_rescore_cache.py`
4. 当前 `trackon2` status 的 `rescoring_status = pending`

### Step B2. 为 Track-On2 导出 unified cache

必须导出符合 `utils/attempt0_schema.py` 的统一缓存。

至少完成：

1. DAVIS
2. 若可承受，再补 Kinetics quick subset

输出位置应进入：

- `outputs/attempt0_2026-06-15_recovery/prediction_caches/`

### Step B3. 跑 validator / rescore / parity

必须真正运行：

1. `scripts/attempt0_validate_cache.py`
2. `scripts/attempt0_rescore_cache.py`
3. `scripts/attempt0_metric_parity.py`

要求更新：

1. `outputs/attempt0_2026-06-15_recovery/status/trackon2.json`
2. `rescoring_status`
3. `AJ / OA / <avg / <4px`
4. long-occ 子表信息

### Step B4. 若 Track-On2 unified cache 暂时困难

必须把原因写得精确：

1. 是 repo-native evaluator 只给 summary，不给 per-query predictions
2. 还是 adapter 没写
3. 还是数据字段对不上 schema

不要笼统写“Phase E not done”。

---

## 6. Phase R2-C: Track-On-R 的真实阻塞确认

### Step C1. 不要再花时间验证 DINOv3 路线

这条线已经通过。

Track-On-R 当前真正阻塞是：

1. `track_on_r.pt` 缺失
2. `verifier.pt` 缺失

### Step C2. 先查本地和上级目录是否已有 checkpoint

在 `FSPT` 之外的常见下载目录也查一次，例如：

1. `/gemini/code`
2. 数据目录
3. 任何 `downloads/`、`weights/`、`checkpoints/` 子目录

只要发现，就不要再把它标成 HF blocked。

### Step C3. 如果真的没有，就把它正式记为 checkpoint-blocked

更新：

- `outputs/attempt0_2026-06-15_recovery/status/trackonr.json`

至少写清：

1. 缺 `track_on_r.pt`
2. 缺 `verifier.pt`
3. 是否仅缺 Track-On-R 评测 checkpoint，还是训练链也缺
4. 一旦拿到文件，应直接执行哪些命令

---

## 7. Phase R2-D: TAPNext++ 与 AllTracker 的本地资产审计

### Step D1. 先做资产审计，不要先写 blocked

当前已看到：

1. `weights/tapir_checkpoint.npy`
2. `weights/bootstapir_checkpoint.npy`
3. 本地有 `tapnext` / `alltracker` wrapper 代码
4. 但没有明确看到：
   - `tapnext_ckpt.npz`
   - `bootstapnext_ckpt.npz`
   - `alltracker.pth`

第二轮先写一个统一审计文件：

- `outputs/external_baseline_asset_audit_2026-06-15.md`

内容至少包括：

1. baseline 名称
2. 本地 wrapper 是否存在
3. 本地 checkpoint 是否存在
4. 需要的官方 checkpoint 名称
5. 推荐下载 URL
6. 当前状态：`ready / missing_checkpoint / missing_repo / unknown`

### Step D2. 若 checkpoint 缺失，再正式写 blocked

更新：

1. `outputs/attempt0_2026-06-15_recovery/status/tapnextpp.json`
2. `outputs/attempt0_2026-06-15_recovery/status/alltracker.json`

不要只留空 skeleton。

至少填写：

1. `reproduction_status = blocked`
2. `known_blockers`
3. `next_action`
4. 官方 checkpoint 名称与 URL

### Step D3. 若发现本地其实已有 checkpoint

就立刻切换策略：

1. 先做 repo-native DAVIS
2. 再尝试 unified cache

不要因为上一轮的先验而放弃。

---

## 8. Phase R2-E: 重写 Attempt 0 状态与决策

### Step E1. 所有 6 个 status JSON 都必须脱离 skeleton 状态

当前问题是：

1. 很多 status 仍是默认 skeleton
2. 没写真实 blocker
3. 没写下一步

第二轮结束时，6 个 status JSON 都至少要满足：

1. `reproduction_status` 不是空泛的 `pending`
2. 有明确 `known_blockers` 或真实结果
3. 有 `next_action`
4. 如果 blocked，写出阻塞资产而不是泛化网络问题

### Step E2. 重写 `final_decision.md`

当前这份：

- `outputs/attempt0_2026-06-15_recovery/final_decision.md`

需要重写或至少大幅修订。

修订要求：

1. 不再把 CoTracker3 简化成 “torch.hub requires GitHub access”
2. 明确区分：
   - 已完成 repo-native reproduction
   - 已完成 unified cache / rescoring
   - 仅有 reference-only row
   - 真正 blocked
3. 明确说明 Attempt 0 到底卡在：
   - checkpoint 获取
   - adapter 导出
   - unified rescoring
   - 还是 repo-native eval

### Step E3. 第二轮结束允许的结论

只允许下面两类：

1. `hold — Track-On2 repo-native confirmed; CoTracker3 anchor partially recovered; remaining baselines blocked by missing checkpoints or missing export adapters`
2. `provisional compare-ready for Track-On2 vs CoTracker3; full Attempt 0 still blocked on Track-On-R / TAPNext++ / AllTracker checkpoints`

不要声称 Attempt 0 已完成，除非 unified rescoring 真正补齐。

---

## 9. 第二轮的最低交付物

第二轮至少必须新增下面这些工件中的大部分：

1. `outputs/cotracker3_checkpoint_manifest.md`
2. `outputs/cotracker3_baseline_repo_native_metrics.json`
3. `outputs/cotracker3_offline_repo_native_metrics.json`
4. `outputs/external_baseline_asset_audit_2026-06-15.md`
5. 更新后的 `outputs/attempt0_2026-06-15_recovery/status/*.json`
6. 更新后的 `outputs/attempt0_2026-06-15_recovery/final_decision.md`
7. 若 Track-On2 unified cache 路线补齐：
   - validator report
   - rescore report
   - parity report

---

## 10. Claude 的执行纪律

1. 不要重复做 DINOv3 / Track-On2 backbone smoke
2. 优先把 `CoTracker3` 这个锚点真正接回 Attempt 0
3. 不要把 “track_on wrapper 依赖 torch.hub” 误写成 “CoTracker3 完全不可离线”
4. 对每个 blocked baseline，必须写出缺的具体资产
5. 没有真正 unified rescoring，就不要写成 unified 已完成

---

## 11. 一句话交接给 Claude

直接把下面这段发给 Claude：

先执行 `/gemini/code/FSPT/docs/claude_round2_attempt0_execution_checklist_2026-06-15.md`。

不要重复修 DINOv3 或重复跑 Track-On2 Week 1。当前第二轮重点是：

1. 把 `CoTracker3 baseline / CoTracker3-offline` 作为 Attempt 0 锚点真正接回来，优先走本地 `baselines/cotracker` + 本地 checkpoint，不要先走 `track_on` 里的 `torch.hub` wrapper
2. 把 `Track-On2 DINOv3` 补成真正的 unified cache / validator / rescore 闭环；当前 `track_on_unified_rescoring.json` 只是部分汇总，不算 unified rescoring 完成
3. 重新审计 `Track-On-R / TAPNext++ / AllTracker` 的本地 checkpoint 资产，再把 status JSON 和 final decision 改成真实阻塞，不要写成笼统网络问题

全过程必须持续更新 status JSON 和决策文档，不允许只留终端日志。
