# Track-On-R Execution Checklist (Week 1) (2026-06-13)

## 0. 目标

这份清单只服务于一个非常具体的目标：

**在 1 周内判断 `Track-On-R` 是否值得成为当前仓库下一阶段的主 baseline。**

更精确地说：

**这份清单是 Attempt 0 的 Track-On family 子阶段，不是完整 Attempt 0 的替代品。**

这里不做方法创新，不做大规模重训，不做 verifier 改造。

第一周只做 4 件事：

1. 环境跑通
2. checkpoint 跑通
3. `Track-On2 / Track-On-R` 的 repo-native eval 跑通
4. 导出统一预测并用本仓库 evaluator 重算主指标

---

## 1. 为什么第一周只做这些

因为 `Track-On-R` 的价值不在“理论上可能强”，而在于它已经有一条完整的公开闭环：

1. `Track-On2` 提供更强 online baseline
2. `Track-On-R` 提供 verifier-guided pseudo-label fine-tuning
3. 官方仓库提供统一 evaluation pipeline
4. 官方 README 已给出 repo-native 目标值

所以第一周不需要发明新东西，只需要回答：

**这条公开路线，在你的机器、你的数据路径、你的统一协议下，到底强不强。**

---

## 2. 第一周必须满足的前置条件

### 2.1 环境

`track_on` README 当前建议：

- `python=3.12`
- `pytorch=2.4.1`
- `torchvision=0.19.1`
- `pytorch-cuda=12.1`
- `mmcv==2.2.0`

还要注意一个现实风险：

- 如果 GPU 与预编译 `mmcv` wheel 不兼容，需要源码编译 `mmcv`

### 2.2 权重访问

根据官方仓库说明：

- `Track-On` checkpoint **不包含** `DINOv3` backbone 权重
- 需要用户先获得 `dinov3-vits16plus` 的访问权限
- 然后通过 `huggingface-cli login` 自动下载

这意味着：

**Day 0 / Day 1 必须先解决 DINOv3 权限问题。**

如果这一步卡住，不应该让整周空转。

正确处理方式：

1. Day 0 先申请 / 验证 `DINOv3` 访问
2. 最长只给 `24-48h` timebox
3. 若超时仍不可用：
   - 暂停 `Track-On-R` 子阶段
   - 并行切到 `TAPNext++` checkpoint 复现
4. 不要在权限问题上连续空等一周

### 2.3 数据路径

第一周先只准备评测集：

- `TAP-Vid DAVIS`
- `TAP-Vid Kinetics`

第一周不要求准备完整 real-world fine-tuning 数据。

---

## 3. 第一周的强制输出

到 Week 1 结束时，必须交付下面 8 个东西：

1. `track_on_env_manifest.md`
2. `track_on_checkpoint_manifest.md`
3. `trackon_adapter_sanity_report.json`
4. `trackon_metric_parity_report.json`
5. `track_on2_repo_native_metrics.json`
6. `track_onr_repo_native_metrics.json`
7. `track_on_unified_rescoring.json`
8. `week1_partial_decision.md`

没有这些工件，就视为第一周没有完成。

---

## 4. Day-by-Day 执行清单

### Day 0: 前置核验

目标：

- 在真正投入 Track-On family 之前，先核对最可能的单点阻塞

必须完成：

1. 核对 `track_on` 官方仓库 README 是否仍明确提供：
   - `Track-On2`
   - `Track-On-R`
   - 对应 checkpoint
2. 核对 `Track-On2 / Track-On-R` 的论文链接和 arXiv ID
3. 测试 `DINOv3` 权重访问是否可用
4. 冻结当前仓库 `CoTracker3 baseline` 作为 unified rescoring 锚点

如果 Day 0 就发现：

- Track-On-R checkpoint 不可得
- 论文 / README 信息不一致
- DINOv3 明显不可用

则不要硬推进，直接把 `TAPNext++` 提前到并行优先级。

### Day 1: 环境和依赖

目标：

- 克隆 `track_on`
- 建立独立环境
- 解决 `mmcv`
- 同步做 DAVIS / Kinetics dataset smoke

必须完成：

1. `git clone https://github.com/gorkaydemir/track_on.git`
2. 建环境
3. 安装依赖
4. 跑最小 import smoke
5. 记录 CUDA、PyTorch、mmcv、GPU 架构
6. 对 DAVIS / Kinetics 各读取 1 个 sample，确认数据路径和 shape 没问题

验收标准：

- `import mmcv`
- `import torch`
- `import evaluation.eval`
- 无 CUDA op 报错

如果 Day 1 结束都没过，优先排查环境，不准提前跳到别的 baseline。

### Day 2: Checkpoint 和 DAVIS repo-native eval

目标：

- 跑通 `Track-On2`
- 跑通 `Track-On-R`
- 先只看 DAVIS

官方仓库 README 给出的评测入口是：

```bash
torchrun --master_port=12345 --nproc_per_node=1 -m evaluation.eval \
  --model_names "trackon2" \
  --trackon_config_path config/test.yaml \
  --trackon_checkpoint_path /path/to/trackon/ckpt \
  --dataset_name davis \
  --dataset_path /path/to/davis
```

Day 2 要做两件事：

1. 用 `Track-On2` checkpoint 跑一次
2. 用 `Track-On-R` checkpoint 跑一次

验收标准：

- `Track-On2` DAVIS repo-native `delta_avg` 接近 `79.9`
- `Track-On-R` DAVIS repo-native `delta_avg` 接近 `80.3`

这里允许有小幅偏差，但如果差距大到 2 个点以上，不能直接继续，先定位：

- checkpoint 错了
- config 错了
- dataset 路径错了
- 评测协议不一致

### Day 3: Kinetics repo-native eval

目标：

- 在 Kinetics 上复现 `Track-On2 / Track-On-R`
- 如果全量太慢，先跑 quick subset

验收标准：

- `Track-On2` Kinetics repo-native `delta_avg` 接近 `69.3`
- `Track-On-R` Kinetics repo-native `delta_avg` 接近 `71.0`

建议：

- 先确认 Kinetics 评测集规模
- 必要时先跑 `100 clips quick check`
- 全量 Kinetics 可以放到 Week 1 后半或 Week 2 前半

到 Day 3 结束时，应当已经能回答：

- 这套公开仓库和 checkpoint 是否真实可跑
- 你本地复现是否基本对得上 README 量级

### Day 4: 统一导出预测

目标：

- 不再停留在 repo-native 表
- 开始为 unified rescoring 做适配

Day 4 只做导出层，不做新训练。

最小要求：

- 为 DAVIS 导出 `Track-On2` 预测
- 为 DAVIS 导出 `Track-On-R` 预测
- 字段统一成：
  - `video_id`
  - `query_points`
  - `pred_tracks`
  - `pred_visibility`
  - `original_size`

关键检查：

1. 坐标到底是 `(x, y)` 还是 `(y, x)`
2. 坐标是像素还是归一化
3. query frame 是不是保留了
4. visibility 是 bool 还是 score
5. query frame 上预测点与 query 坐标是否对齐
6. 随机可视化 3-5 个样本，确认轨迹没有整体翻转 / 缩放错误

如果导出不可靠，后面的统一重算全部失真。

建议：

- 导出适配器不要写在 `track_on` 仓库内部
- 在当前 FSPT 仓库维护统一 adapter 层

### Day 5: DAVIS unified rescoring

目标：

- 用本仓库 evaluator 重算 DAVIS
- 完成 evaluator parity check

统一使用本仓库协议：

- `query_mode = strided`
- `metric_resolution_mode = original`

对应本地代码：

- [datasets_code/metrics.py](/gemini/code/FSPT/datasets_code/metrics.py)
- [configs/fspt_base.yaml](/gemini/code/FSPT/configs/fspt_base.yaml)

Day 5 结束时，必须产出：

- `CoTracker3 baseline`
- `Track-On2`
- `Track-On-R`

三行在同一协议下的 DAVIS 对照表：

- `AJ`
- `OA`
- `<avg`
- `<4px`

并额外完成：

- 本仓库 evaluator vs 官方 evaluator parity 对照
- `long-occ>20` 子集表

### Day 6: Kinetics unified rescoring

目标：

- 对 Kinetics 做同样的统一重算

到 Day 6 结束时，应该已经有：

1. repo-native reproduction table
2. unified rescoring table
3. long-occ / re-entry 辅助表

这时才能真正判断：

- `Track-On-R` 是不是只是 README 好看
- 还是在你的协议下也确实更强

### Day 7: 决策

Day 7 不再跑新实验，只做决策和补缺。

必须回答 3 个问题：

1. `Track-On-R` 在 unified rescoring 后是否明显强于当前 `CoTracker3 baseline`？
2. `Track-On-R` 是否明显强于 `Track-On2`？
3. 如果不明显强，问题出在协议、适配，还是方法本身？

然后输出：

- `provisional go Track-On family`
- `hold Track-On, investigate protocol/adapter gap`
- `stop Track-On-R fine-tune, keep or drop Track-On2 separately`
- `stop Track-On family, move to TAPNext++`

这是 `partial decision`，不是完整 Attempt 0 final decision。

---

## 5. 第一周的裁决标准

### 5.1 Repo-native reproduction pass

满足下面两条，才算 repo-native 过关：

1. DAVIS / Kinetics 上的 `Track-On2 / Track-On-R` 数字接近官方 README 量级
2. 两者的相对排序与官方 README 一致，即 `Track-On-R >= Track-On2`

建议量化：

- 与官方 repo-native 数字差异不超过约 `1.5` 点

### 5.2 Unified rescoring pass

这里也必须拆成两层。

#### A. Track-On family baseline replacement pass

满足下面条件，才算值得继续：

1. `max(Track-On2, Track-On-R)` 在 `DAVIS` 上对 `CoTracker3 baseline` 有清晰优势
2. 不只看 `AJ`，还要看：
   - `OA`
   - `<avg`
   - `<4px`
3. 至少满足：
   - `AJ` 不低
   - `<avg` 或 `<4px` 至少一个明显更好
   - `OA` 不明显下降

#### B. Track-On-R fine-tune pass

满足下面条件，才算值得把 `Track-On-R` 升级为主线：

1. `Track-On-R` 对 `Track-On2` 至少在一套核心数据集上有实质优势；或
2. `Track-On-R` 在更接近真实视频的 benchmark 上更强，且 `DAVIS` 不显著退化

如果 B 不满足，不应误杀整个 Track-On family。

---

## 6. Kill Criteria

第一周内任一满足，直接停止把 `Track-On-R` 作为第一优先级主线：

1. `DINOv3` 权重在 `24-48h` 内仍不可用，且没有可行 fallback
2. repo-native eval 无法稳定复现，且 2 天内定位不了原因
3. unified rescoring 后对 `CoTracker3 baseline` 没有明确优势
4. `Track-On-R` 相比 `Track-On2` 没有清晰增益
5. adapter sanity check 或 evaluator parity check 未通过

注意：

这里的“停止”不是说 Track-On 系列完全没价值，而是说：

**不值得继续优先占用你下一轮主训练资源。**

---

## 7. 第一周不做的事

不要在 Week 1 做下面这些：

- 自己重训 Track-On2
- 自己重训 Track-On-R
- 自己改 verifier
- 自己扩 teacher pool
- 自己加新 pseudo-label 规则
- 把当前仓库的 reliability 模块硬塞进 Track-On-R

Week 1 的任务只是：

**验证公开强 baseline 是否真的值得接管主线。**

---

## 8. 如果 Week 1 成功，Week 2 的正确动作

只有当 Week 1 成功时，才进入下面动作：

1. 先确认是：
   - `Track-On family` 成功
   - 还是 `Track-On-R fine-tune` 也成功
2. 再决定是否固定 `Track-On-R` 为 Attempt 1 主 baseline
3. 开始补真实域微调数据准备
4. 评估是否要复现 verifier ensemble
5. 设计“只在 Track-On-R 之上做增量”的实验，而不是把旧仓库失败线强行迁过去

如果 Week 1 不成功，下一步不是继续硬练，而是：

- 转去 `TAPNext++`

---

## 9. 当前最合理的解释

截至 `2026-06-13`，`Track-On-R` 之所以排第一，不是因为它最“新”，而是因为它最符合当前目标：

1. 它已经把 `online tracker + memory + real-world pseudo-label fine-tuning` 做成了完整公开闭环。
2. 它和当前仓库已失败的 `CoTracker3 后处理 recovery` 路线是正交的。
3. 它最适合先拿来做统一协议下的 head-to-head baseline replacement。

因此，第一周最重要的不是创新，而是验证：

**它是否真的值得成为新的起点。**

---

## 10. 一手来源

- Track-On 官方仓库：<https://github.com/gorkaydemir/track_on>
- Track-On-R 论文：<https://arxiv.org/abs/2603.12217>
- Track-On2 论文：<https://arxiv.org/abs/2509.19115>
- Track-On 数据说明：<https://github.com/gorkaydemir/track_on/blob/main/dataset/README.md>
