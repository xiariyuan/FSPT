# Claude Handoff Recovery Execution Checklist (2026-06-14)

## 0. 目的

这份文档不是新的研究计划。

它只做一件事：

**把当前服务器重启后已经恢复出的真实状态、已完成工件、当前阻塞、以及 Claude 下一步应执行的详细顺序固定下来。**

目标是避免 Claude 重复做已经做过的事，也避免它误判“DINOv3 路线已经完全跑通”。

---

## 1. 当前恢复结论

截至 `2026-06-14` 当前工作区的真实状态如下。

### 1.1 已确认完成

1. `ModelScope` 上的目标 DINOv3 模型已经下载到本地目录：
   - `third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/`
2. 以下工件已经存在：
   - `outputs/dinov3_modelscope_manifest.json`
   - `outputs/dinov3_local_layout_report.json`
   - `outputs/dinov3_load_smoke.json`
   - `outputs/dinov3_env_hint.txt`
   - `outputs/trackon_local_backbone_patch_notes.md`
3. `baselines/track_on/model/vit_adapter/dinov3_adapter/dinov3_vit_adapter.py` 已经被改为优先读取：
   - `DINOV3_LOCAL_DIR`
4. Attempt 0 的脚手架 smoke 已做过，存在：
   - `outputs/attempt0_smoke/`
   - 其中包括 `manifests/`、`status/`、synthetic cache / validation / rescore / parity 工件
5. `baselines/track_on` 本地仓库已存在。
6. `Track-On2` 本地 checkpoint 已存在两份：
   - `baselines/track_on/checkpoints_trackon2_dinov3.pt`
   - `baselines/track_on/checkpoints_trackon2_dinov2.pt`

### 1.2 当前没有闭环完成

下面这些事情**还不能声称已经完成**：

1. `track_on` 本地 DINOv3 backbone smoke
2. `Track-On2 / Track-On-R` repo-native DAVIS eval
3. `Track-On2 / Track-On-R` repo-native Kinetics eval
4. `track_on_env_manifest.md`
5. `track_on_checkpoint_manifest.md`
6. `trackon_adapter_sanity_report.json`
7. `trackon_metric_parity_report.json`
8. `track_on2_repo_native_metrics.json`
9. `track_onr_repo_native_metrics.json`
10. `track_on_unified_rescoring.json`
11. `week1_partial_decision.md`
12. Attempt 0 全表、各 baseline status 更新、`final_decision.md`

### 1.3 当前真实阻塞

当前最关键的阻塞不是“ModelScope 不可用”，而是下面这些：

1. **当前默认 Python 环境并不满足 DINOv3 / Track-On 所需依赖**
   - 目前实测：
     - `torch == 2.1.2+cu121`
     - `transformers == 4.40.2`
     - `mmcv` 缺失
     - `mediapy` 缺失
2. 在当前环境下，重新执行 DINOv3 load 会失败：
   - `transformers 4.40.2` 不认识 `dinov3_vit`
3. 在当前环境下，`track_on` import smoke 会失败：
   - `ModuleNotFoundError: No module named 'mmcv'`
4. 本地没有发现 `Track-On-R` checkpoint
5. 本地没有发现 `verifier` checkpoint
6. 还没有确认 TAP-Vid DAVIS / Kinetics 的实际数据路径是否准备好到可直接评测

这意味着：

**DINOv3 下载落盘成功 != 当前环境里 Track-On-R 路线已经可直接评测。**

---

## 2. 关键文件与现状

Claude 开始执行前，必须先阅读这些文件：

1. `docs/dinov3_modelscope_no_wait_execution_checklist_2026-06-14.md`
2. `docs/claude_attempt0_detailed_task_list_2026-06-14.md`
3. `docs/trackonr_execution_checklist_2026-06-13.md`
4. `docs/new_direction_ranking_2026-06-13.md`
5. `docs/attempt_0_unified_reproduction_checklist_2026-06-13.md`
6. `docs/attempt_0_cotracker3_offline_anchor_2026-06-13.md`
7. `docs/attempt_0_status_template_2026-06-13.md`
8. `docs/claude_attempt0_execution_brief_2026-06-13.md`
9. `ai回复第一轮.txt`

然后核对下面这些已有工件：

1. `outputs/dinov3_modelscope_manifest.json`
2. `outputs/dinov3_local_layout_report.json`
3. `outputs/dinov3_load_smoke.json`
4. `outputs/dinov3_env_hint.txt`
5. `outputs/trackon_local_backbone_patch_notes.md`
6. `outputs/attempt0_smoke/manifests/attempt0_manifest.json`
7. `outputs/attempt0_smoke/status/*.json`

---

## 3. 必须先知道的事实

### 3.1 `track_on` DINOv3 patch 已落地

当前 `baselines/track_on/model/vit_adapter/dinov3_adapter/dinov3_vit_adapter.py` 已包含本地路径注入逻辑。

也就是说，这一步**不要重复设计**，应先验证现有 patch 是否在正确环境里可用。

### 3.2 不要再用旧环境假装 smoke 已通过

之前输出文件里记录过一版：

- `torch 2.5.1`
- `transformers 5.12.0`

但当前实际默认环境实测不是这个版本。

所以 Claude 不能仅凭已有 JSON 判断“当前环境没问题”，必须重新做：

1. 版本核对
2. import smoke
3. 本地 DINOv3 load smoke

### 3.3 Attempt 0 初始化应优先用 `scripts/init_attempt0_run.py`

仓库里有两个相近脚本：

1. `scripts/init_attempt0_run.py`
2. `scripts/attempt0_init_run.py`

本轮应以文档指定、且已经 smoke 过的：

- `scripts/init_attempt0_run.py`

为准。

不要混用两个输出目录规范。

---

## 4. Claude 执行总顺序

执行顺序固定如下：

1. **先恢复并验证 Track-On 环境**
2. **再做 DINOv3 本地 backbone smoke**
3. **若 DINOv3 路线失败，立刻写 fallback 决策并切到 Track-On2 DINOv2**
4. **若 DINOv3 路线通过，再继续 Track-On family Week 1**
5. **Track-On family 子阶段完成后，再进入完整 Attempt 0**

不要跳步。

---

## 5. Phase A: 环境恢复与再验证

### Step A1. 记录当前环境实际版本

至少记录：

1. Python version
2. `torch`
3. `torchvision`
4. `torchaudio`
5. `transformers`
6. `mmcv`
7. `mediapy`
8. CUDA version
9. GPU 型号

输出到：

- `outputs/track_on_env_manifest.md`

必须明确区分：

1. 当前默认环境是什么
2. 是否满足 `track_on` README 依赖
3. 需要补哪些包

### Step A2. 修复到一个可运行 `track_on` 的环境

最低要求：

1. `import torch`
2. `import transformers`
3. `import mmcv`
4. `import mediapy`
5. `cd baselines/track_on && python -c "from model.trackon_predictor import Predictor"`

如果 `mmcv` 因 GPU 架构问题不能直接装 wheel，则按 `baselines/track_on/README.md` 的说明走源码编译。

这一步完成前，不要开始 repo-native eval。

### Step A3. 重新做 DINOv3 本地 load smoke

在**当前实际用于 Track-On 的环境**中重新执行：

1. `AutoConfig.from_pretrained(..., trust_remote_code=True, local_files_only=True)`
2. `AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only=True)`

重写：

- `outputs/dinov3_load_smoke.json`

要求至少补充：

1. `python_version`
2. `torch_version`
3. `transformers_version`
4. `config_ok`
5. `model_ok`
6. `error`

如果这里失败，不能继续假设 DINOv3 可用。

---

## 6. Phase B: DINOv3 本地 backbone 路线判定

### Step B1. 做 `track_on` import smoke

在 `baselines/track_on` 下，设置：

```bash
export DINOV3_LOCAL_DIR=/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

然后至少做：

1. `from model.trackon_predictor import Predictor`
2. 用 `config/test.yaml` 初始化 `Predictor`
3. 尽量只构建模型，不跑全量数据

输出到：

- `outputs/trackon_local_backbone_smoke.txt`

必须记录：

1. 是否成功 import
2. 是否成功构建 `Predictor`
3. 是否确实走了本地 `DINOV3_LOCAL_DIR`
4. 是否还有任何 HF 网络访问
5. 完整错误栈（如失败）

### Step B2. 做最小前向 smoke

如果模型构建成功，再补一个最小前向：

1. 随机构造一小段假视频 tensor
2. 构造少量 query
3. 调一次 `Predictor.forward(...)` 或最小可行路径

目的不是测指标，而是确认：

1. backbone 初始化真的没问题
2. `mmcv` CUDA op 没有立刻炸
3. `track_on` 不是只能 import 不能跑

把结果追加到：

- `outputs/trackon_local_backbone_smoke.txt`

### Step B3. 触发 fallback 的条件

以下任一成立，立即停止 DINOv3 路线：

1. 当前环境中 `AutoConfig` / `AutoModel` 仍无法本地加载 DINOv3
2. `track_on` 仍无法 import / 构建
3. `mmcv` 安装或源码编译超过合理 timebox 仍未通过
4. `Predictor` 最小前向仍依赖 HF 或直接失败
5. 修复依赖和 patch 时间明显超过这条线的价值

一旦触发：

1. 写 `outputs/fallback_decision.md`
2. 明确说明失败点
3. 立即切到 `Track-On2 DINOv2`

不要空耗。

---

## 7. Phase C: Track-On2 DINOv2 fallback 路线

如果进入 fallback，Claude 应直接执行：

### Step C1. 确认 DINOv2 本地路径

当前仓库已有本地 DINOv2 资产：

1. `baselines/hf/facebook_dinov2_small/`
2. `weights/dinov2/dinov2_vits14_pretrain.pth`

同时注意：

`baselines/track_on/model/vit_adapter/dinov2_adapter/dinov2_vit_adapter.py` 已支持：

- `TRACKON_DINOV2_MODEL_PATH`

所以 fallback 时优先走本地路径，不要依赖外网自动下载。

### Step C2. 用 `config/test_dinov2.yaml` 做 Track-On2 smoke

要求：

1. 设好 `TRACKON_DINOV2_MODEL_PATH`
2. 使用：
   - `baselines/track_on/checkpoints_trackon2_dinov2.pt`
   - `baselines/track_on/config/test_dinov2.yaml`
3. 完成 import / 构建 / 最小前向 smoke

### Step C3. 产出 fallback 决策文档

`outputs/fallback_decision.md` 至少写清：

1. 为什么 DINOv3 路线停止
2. 停在了哪个步骤
3. 是否建议未来重试
4. 当前切换到的 baseline 是什么
5. 这是否只 kill `Track-On-R DINOv3 路线`，还是 kill 整个 Track-On family

结论默认应是：

**只 kill 当前 DINOv3 / Track-On-R 路线，不自动 kill Track-On2 baseline replacement。**

---

## 8. Phase D: Track-On family Week 1 子阶段

只有在 Phase B 成功，或者 Phase C fallback 成功后，才继续这里。

### Step D1. 产出 checkpoint manifest

写：

- `outputs/track_on_checkpoint_manifest.md`

至少记录：

1. `track_on` 仓库路径
2. 使用的 config
3. 使用的 checkpoint 路径
4. checkpoint 文件大小
5. checkpoint SHA256
6. backbone 来源
   - DINOv3 local dir
   - 或 DINOv2 local path
7. 是否缺少 `Track-On-R` checkpoint
8. 是否缺少 `verifier` checkpoint

### Step D2. 判断 Track-On-R checkpoint 是否可得

当前本地只确认有：

1. `checkpoints_trackon2_dinov3.pt`
2. `checkpoints_trackon2_dinov2.pt`

未发现：

1. `track_on_r.pt`
2. `verifier.pt`

Claude 必须明确判断：

1. 本地是否已有可用 `Track-On-R` checkpoint
2. 若无，是否从官方源补下载
3. 若网络或权限不可行，是否把 Week 1 暂时降级为：
   - `Track-On2 only`

不要假设 Track-On-R checkpoint 已经在本地。

### Step D3. 跑 DAVIS repo-native eval

优先顺序：

1. `Track-On2`
2. 若 checkpoint 可得，再跑 `Track-On-R`

要求输出：

1. `outputs/track_on2_repo_native_metrics.json`
2. `outputs/track_onr_repo_native_metrics.json`（若可得）

必须记录：

1. dataset path
2. config path
3. checkpoint path
4. repo-native 指标原名
5. 与 README 参考值差异
6. wall time
7. peak memory
8. 实际是否为 DINOv3 版或 DINOv2 版

### Step D4. 跑 Kinetics repo-native eval

Kinetics 先允许：

1. `100 clips quick check`
2. 再决定全量

不要一上来直接全量。

### Step D5. 统一导出与 adapter sanity

Track-On family 至少要完成：

1. unified cache 导出
2. `attempt0_validate_cache.py --require-gt`
3. `attempt0_rescore_cache.py`
4. `attempt0_metric_parity.py`

并产出：

1. `outputs/trackon_adapter_sanity_report.json`
2. `outputs/trackon_metric_parity_report.json`
3. `outputs/track_on_unified_rescoring.json`

### Step D6. Week 1 partial decision

写：

- `outputs/week1_partial_decision.md`

只允许用以下结论模板：

1. `provisional go Track-On family`
2. `hold Track-On, investigate protocol/adapter gap`
3. `stop Track-On-R fine-tune, keep or drop Track-On2 separately`
4. `stop Track-On family, move to TAPNext++`

---

## 9. Phase E: 完整 Attempt 0

只有在 Track-On family 子阶段完成并且状态明确后，才进入完整 Attempt 0。

### Step E1. 初始化正式 Attempt 0 工作区

使用：

```bash
python scripts/init_attempt0_run.py --name attempt0_2026-06-14_recovery --out-root outputs
```

不要用旧的 `scripts/attempt0_init_run.py` 混出另一套目录规范。

### Step E2. 冻结 `CoTracker3 baseline`

先把锚点补齐：

1. repo-native reproduction
2. unified cache
3. unified rescoring
4. parity

### Step E3. 完成剩余 baseline

固定候选集仍然是：

1. `cotracker3_baseline`
2. `cotracker3_offline`
3. `trackon2`
4. `trackonr`
5. `tapnextpp`
6. `alltracker`

若某对象不可跑，必须在对应 status JSON 中明确写阻塞原因。

### Step E4. 每个 baseline 都要更新 status

更新：

- `outputs/<attempt0_run>/status/<model>.json`

不要只留终端日志。

### Step E5. 输出最终决策

写：

- `outputs/<attempt0_run>/final_decision.md`

必须建立在 unified rescoring table 基础上，不允许只根据 README 下结论。

---

## 10. Claude 的执行纪律

1. 先验证现有工件是否仍对应当前环境，不要盲信旧 JSON
2. 先补环境，再做 Track-On smoke，再决定是否 fallback
3. 只要 DINOv3 路线失败，就立刻形成书面 fallback 决策
4. 不要因为缺 `Track-On-R` checkpoint 而把整周卡死
5. `Track-On2` 可以单独作为 baseline replacement 候选
6. Attempt 0 完成前，不要发散到新训练、新 verifier、新 recovery 变体

---

## 11. 一句话交接给 Claude

直接把下面这段发给 Claude 即可：

先执行 `/gemini/code/FSPT/docs/claude_handoff_recovery_execution_checklist_2026-06-14.md`。

重点不是重新 brainstorm，而是恢复当前真实状态并继续执行：

1. 先验证当前实际环境是否还能支持本地 DINOv3 和 `track_on`
2. 现有 `outputs/dinov3_*` 与 `outputs/trackon_local_backbone_patch_notes.md` 说明 DINOv3 下载和 patch 以前做过，但**当前默认环境**实测仍缺 `mmcv`，且 `torch/transformers` 版本不满足本地 DINOv3 load
3. 先补环境并完成 `track_on` 本地 backbone smoke；若失败，立刻写 `outputs/fallback_decision.md` 并切 `Track-On2 DINOv2`
4. 然后继续 `Track-On family Week 1`
5. 若 Week 1 子阶段通过，再进入完整 Attempt 0

全过程必须输出文档要求的 manifest、smoke、repo-native metrics、adapter sanity、parity、rescoring、decision，不允许只留终端日志。
