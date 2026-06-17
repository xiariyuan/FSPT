# New Direction Ranking (2026-06-13)

## 0. 目标

这份文档不是为了继续包装当前仓库已有结果，而是为了重新确定：

**接下来最值得投入算力和工程成本、最有可能直接提升官方 point tracking 主指标的方向。**

这里的“主指标”只指：

- `TAP-Vid DAVIS`: `AJ / OA / <avg / <4px`
- `TAP-Vid Kinetics`: `AJ / OA / <avg / <4px`

其他指标例如：

- `selective AJ`
- `AURC`
- `ECE / Brier`
- `coverage-risk`
- `neighbor deviation AUC`

都只能作为分析材料，不再作为主成功标准。

---

## 1. 已被本仓库证伪的东西

### 1.1 CoTracker3 特征上的后处理 recovery / rerank 已经基本走完

以下路线都已经有硬负结果：

- local selector / scalar ranker
- vector ranker
- patch verifier
- geometry rerank
- direct recovery attention / DiReCT-like quick screen
- RouteA integrated relocal / verifier integration

本质结论：

1. `search` 可以找到候选，但 `ranking` 不能稳定泛化。
2. CoTracker3 当前特征空间不支持再靠后处理 recovery 模块获得稳定主指标提升。
3. 继续在这条线上投入，先验极低。

关键本地证据：

- `docs/experiment_review_table_2026-06-09.md`
- `docs/local_selector_stage2c_threshold_sweep_status_2026-06-12.md`
- `docs/local_geometry_rerank_status_2026-06-12.md`
- `docs/direct_recovery_attention_audit_v1_1_status_2026-06-12.md`

### 1.2 neighbor deviation 只能做 GT-track 分析，不能做训练信号

已经确认：

- 在 `GT-track` 分析上，`neighbor deviation` 是强信号
- 在 `predicted tracks` 上，它输给 `pred_visibility`
- 所以它不能作为 pseudo-label filtering 的主信号

关键本地证据：

- `docs/trajectory_manifold_verifier_v1_status_2026-06-12.md`
- `docs/trajectory_manifold_verifier_v1_loocv_status_2026-06-12.md`
- `docs/neighbor_deviation_deep_analysis_v1_1_status_2026-06-13.md`
- `docs/predicted_neighbor_deviation_signal_v1_status_2026-06-13.md`

### 1.3 reliability / calibration 线是分析贡献，不是主指标提升方法

这些结果有研究价值，但不应再被误当作“接下来最有希望的提升路径”：

- calibration
- selective tracking
- risk coverage
- failure taxonomy

它们能帮助理解问题，不能代替 `AJ/OA/<avg/<4px` 的真实增益。

---

## 2. 这轮重排的原则

这次排序只看四件事：

1. **能否快速打到官方主指标**
2. **是否与已失败的 CoTracker3 后处理路线正交**
3. **是否有现成公开代码/权重/训练链**
4. **能否在 1-2 周内给出明确 kill decision**

因此：

- 继续修当前 CoTracker3 后处理路线，不再进入候选
- 只优先考虑“换 base model / 换训练范式 / 换数据规模”的方向
- `BootsTAP`、`AJ_RD`、`real-video pseudo-labeling` 这些更适合作为**训练配方**，而不是单独的“方向名称”

---

## 3. 最终排序

### 排名 1：Track-On-R / Track-On2 路线

**结论：这是第一优先级，也是最应该先尝试的路线。**

#### 为什么排第一

相较于其它候选，它同时满足：

1. **最容易复现**
2. **最完整的公开代码/权重/评测管线**
3. **已经验证过 real-world pseudo-label fine-tuning 的有效性**
4. **与当前仓库失败路线正交**

一手来源显示：

- `track_on` 官方仓库同时提供 `Track-On2`、`Track-On-R`、verifier、teacher ensemble 和统一评测
- README 明确给出 `Track-On-R` 与 `Track-On2` 在多个 benchmark 上的指标
- `Track-On-R` 的核心机制正是 `verifier-guided pseudo-labeling`

关键来源：

- Track-On 官方仓库：<https://github.com/gorkaydemir/track_on>
- Track-On2 论文：<https://arxiv.org/abs/2509.19115>
- Track-On-R 论文：<https://arxiv.org/abs/2603.12217>

来自官方仓库 README 的直接证据：

- `Track-On-R` 是基于 `Track-On2` 的 `real-world fine-tuning using verifier-guided pseudo-labels`
- repo 提供 6 个 benchmark dataloader 和 8 个 baseline wrapper
- README 给出复现目标：
  - `Track-On-R`: DAVIS `80.3`, Kinetics `71.0`, RoboTAP `82.6`, EgoPoints `67.3`, Dynamic Replica `75.1`, PointOdyssey `53.4`（均为 `δ_avg^x`）

#### 为什么之前的失败没有否定它

当前仓库失败的是：

- 在 `CoTracker3` 单一特征空间上做后处理
- 用单模型置信度或 GT-only 分析信号去筛 pseudo labels

而 `Track-On-R` 的核心完全不同：

1. base model 不是 CoTracker3
2. pseudo-labeling 不是单模型自循环，而是 **多 teacher + verifier**
3. verifier 的目标是 **跨 teacher 逐帧选最可靠预测**
4. 真实视频微调本来就是它的主设计点，不是额外附加模块

#### 风险

主要不是“方法无效”，而是：

1. 公开指标主打 `δ_avg^x`，你仍然必须用统一协议重算 `AJ / OA / <avg / <4px`
2. DINOv3 backbone 权重有访问门槛
3. 多 teacher pipeline 部署稍重
4. Track-On-R 的最终优势必须在 unified rescoring 下确认，不能直接相信 README 表

#### 最小 PoC

第一阶段不要创新，先复现：

1. 跑 `Track-On2`
2. 跑 `Track-On-R`
3. 用统一的 `TAP-Vid DAVIS / Kinetics` 官方协议重算 `AJ / OA / <avg / <4px`
4. 与当前 CoTracker3 baseline 做 head-to-head 对比

#### Kill Criteria

任一满足即停止把它作为主线：

1. 1 周内无法稳定复现 Track-On2 / Track-On-R 官方量级结果
2. 按统一协议重算后，对官方主指标没有形成清晰优势
3. `Track-On-R` 相比 `Track-On2` 在你的协议下没有实质增益

#### 判断

如果你的目标是“最快拿到更强 baseline 并开始真正做增益实验”，这是最优先路线。

---

### 排名 2：TAPNext++ 路线

**结论：这是第二优先级，也是最值得作为“长遮挡 / re-detection 专项主线”的路线。**

#### 为什么排第二

它的研究针对性极强：

1. 明确指出 `re-detection` 是当前文献 blind spot
2. 提出 `AJ_RD`
3. 用 `periodic roll` 和 `occluded-point supervision` 专门打 long-occlusion / re-entry
4. 支持 1024-frame 长序列训练

关键来源：

- TAPNext++ 论文：<https://arxiv.org/abs/2604.10582>
- TAPNext++ 官方页：<https://tap-next-plus-plus.github.io/>
- TAP / TAPNext / BootsTAP 官方仓库：<https://github.com/google-deepmind/tapnet>

来自一手来源的直接证据：

- TAPNext++ 论文明确说：
  - 当前 literature 对 re-detection 存在 blind spot
  - 提出 `AJ_RD`
  - 引入 `periodic roll` 来模拟 re-entry
  - 通过 1024-frame training + sequence parallelism 强化长程鲁棒性
- 官方 repo 明确写到：
  - `TAPNext` 的 best checkpoint 本身就使用了 `BootsTAP`
  - `TAPNext++` checkpoint 已经 fine-tuned on `PointOdyssey + Kubric-1024`

#### 为什么不是第一

不是因为它不强，而是因为：

1. 训练工程复杂度高于 Track-On-R
2. 长序列 1024-frame 训练和 sequence parallelism 的门槛更高
3. 当前最稳妥的第一步仍然是先复现一个更完整、可立即跑通的 real-world fine-tune baseline

#### 为什么之前的失败没有否定它

当前失败的是 `post-hoc recovery`。

`TAPNext++` 改的是：

- base architecture
- long-sequence training
- occlusion/re-entry augmentation
- explicit redetection evaluation

它根本不是“在 CoTracker3 上再补一个 recovery head”。

#### 风险

1. 训练链更重
2. 需要较大算力才能认真做长序列实验
3. 如果 checkpoint 足够强，进一步继续往上抬可能 margin 不大

#### 最小 PoC

1. 用官方 checkpoint 在 `DAVIS / Kinetics` 上重跑
2. 除标准指标外，额外建立 `AJ_RD` / long-occ / re-entry 子集评估
3. 看它是否在当前仓库最关注的痛点场景上显著优于 CoTracker3 和 Track-On2

#### Kill Criteria

1. 1 周内无法稳定复现官方 checkpoint 数字
2. 在 long-occ / re-entry 上没有体现出相对 CoTracker3 的结构性优势
3. 训练改造的成本明显高于其潜在收益

#### 判断

如果你最在意的是“从根上解决长遮挡和重检测”，它是最强的研究方向；但从**先拿结果**的角度，它排在 Track-On-R 后面更合理。

---

### 排名 3：AllTracker 路线

**结论：这是第三优先级，也是最有价值的“范式正交备选”和 teacher 候选。**

#### 为什么排第三

它最大的价值不在于“立刻替代一切”，而在于：

1. 架构范式和现在的稀疏 point tracker 完全不同
2. 稠密 flow-field 形式天然适合做 teacher 或对照
3. 官方 repo 完整、训练 recipe 公开、评测入口清楚

关键来源：

- AllTracker 论文：<https://arxiv.org/abs/2506.07310>
- 官方项目页：<https://alltracker.github.io/>
- 官方代码：<https://github.com/aharley/alltracker>

一手来源直接证据：

- 官方 repo 说明：
  - AllTracker 两阶段训练：`Kubric -> mixed datasets`
  - 第二阶段联合 point tracking datasets 和 optical flow datasets
  - 直接给出 DAVIS 评估例子：
    - `da: 76.3`
    - `aj: 63.3`
    - `oa: 90.0`
  - 更高分辨率还可继续涨到 `aj: 67.2`
- 论文的 Table 3 还给出 TAP-Vid 平均 AJ：
  - `AllTracker 68.9`
  - `CoTracker3 63.1`

#### 为什么之前的失败没有否定它

因为当前仓库从来没有真正走过“dense flow + point tracking 联合训练”这条路。

它失败的是：

- sparse point tracker 上的后处理
- GT-track signal 迁移到 predicted-track training

`AllTracker` 则是：

- dense correspondence
- optical-flow + tracking joint training
- 高分辨率 dense-to-sparse evaluation

这是正交路线。

#### 为什么不排更高

1. 相比 Track-On-R / TAPNext++，它对你当前关注的“官方 sparse TAP 主指标”路径没那么直接
2. 高分辨率稠密输出转回 sparse TAP setting 仍有适配成本
3. 它更适合作为：
   - 新 baseline
   - teacher
   - 后续融合对象

#### 最小 PoC

1. 直接跑官方 checkpoint
2. 用 repo 自带 `test_dense_on_sparse.py` 在 DAVIS / Kinetics 上评估
3. 建立：
   - standalone baseline
   - 作为 Track-On-R teacher 的潜力判断

#### Kill Criteria

1. 官方 checkpoint 在你的协议下明显不 competitive
2. 高分辨率优势不能稳定转化为 official sparse metrics 的优势
3. 作为 teacher 也不提供额外信息增益

#### 判断

它不是最该第一个试的，但它是最值得保留的“正交备选”和 multi-teacher 资源。

---

## 4. 不列入第一轮主线的方向

### 4.1 BootsTAP 不是单独方向，而是训练配方

`BootsTAP` 的价值很大，但它更应该被当成：

- `Track-On-R` / `TAPNext++` 的训练 recipe 参考

而不是一个独立主方向。

原因：

1. 单独说“做 BootsTAP”太抽象
2. 真正决定结果的是：
   - base model
   - teacher quality
   - data scale
   - training protocol

补充：

- `Track-On-R verifier`
- `TAPNext++ roll augmentation`
- `BootsTAP real-video pseudo-labeling`

这三者在方法论上是可叠加的，后续若前两轮 baseline 复现成功，这比直接跳 3D 更适合作为第二阶段 2D 研究储备。

### 4.2 3D / TAPVid-3D 先降为第二阶段研究储备

虽然本仓库的 `Stage0 GT 3D oracle` 很强，外部也有：

- TAPVid-3D
- PointSt3R
- SpatialTracker / DELTA

但当前用户目标是：

**先把 2D 官方主指标打上去。**

因此 3D 方向现在不应抢第一轮资源。

它适合在下面情况下再启动：

1. 当 Track-On-R / TAPNext++ / AllTracker 的 2D 提升边际变小
2. 当你需要更强研究创新故事，而不是更快拉主指标

---

## 5. 强制尝试顺序

### Attempt 0：统一外部 baseline 复现梯队

这个步骤是强制的，不可跳过。

统一协议下先重跑：

1. 当前 CoTracker3 baseline
2. CoTracker3-Offline
3. Track-On2
4. Track-On-R
5. TAPNext++
6. AllTracker

统一主表只看：

- DAVIS `AJ / OA / <avg / <4px`
- Kinetics `AJ / OA / <avg / <4px`

同时强制保留辅助子表：

- `long-occ>20`
- `re-entry first-frame`
- `AJ_RD`，如果支持

不要先看 selective / calibration。

目标：

1. 建立真正统一的横向表
2. 防止被论文里的不同协议误导
3. 快速确定“新的 base 应该是谁”

### Attempt 1：Track-On-R

如果复现顺利，先把它作为第一主线。

原因：

- 工程闭环最完整
- real-world pseudo-label fine-tune 已实现
- 最容易最先出主指标结果

### Attempt 2：TAPNext++

如果：

- Track-On-R 已复现
- 但长遮挡 / re-entry 仍是核心痛点

则转到 TAPNext++ 专项线。

### Attempt 3：AllTracker

如果：

- 你需要更强的架构正交性
- 或需要更强 teacher pool

再正式开 AllTracker 线。

---

## 6. 推荐的实际执行口径

### 如果目标是“最快提升”

先做：

1. `Track-On-R`
2. `TAPNext++`
3. `AllTracker`

### 如果目标是“创新优先”

先做：

1. `TAPNext++`
2. `AllTracker`
3. `3D / TAPVid-3D`

### 如果目标是“又要快又要稳”

最佳折中顺序仍然是：

1. `Track-On-R`
2. `TAPNext++`
3. `AllTracker`

---

## 7. 当前正式结论

当前仓库下一步**不应该**再是：

- 继续修 CoTracker3 后处理
- 继续做 neighbor-deviation pseudo-label filtering
- 继续在 reliability/calibration 上兜圈

当前仓库下一步**应该**是：

1. **先统一复现外部更强 baseline**
2. **第一优先级尝试 Track-On-R**
3. **第二优先级尝试 TAPNext++**
4. **第三优先级尝试 AllTracker**

补充：

- `Track-On-R Week 1` 只应视为 `Attempt 0` 的 Track-On family 子阶段
- `CoTracker3-Offline` 必须加入 Attempt 0，作为强公开 2D 锚点

一句话总结：

**不要再在当前 CoTracker3 后处理空间里找增益；直接换到更强的公开基础模型和训练范式，并按 Track-On-R -> TAPNext++ -> AllTracker 的顺序逐个尝试。**

---

## 8. 一手来源

- Track-On 官方仓库：<https://github.com/gorkaydemir/track_on>
- Track-On2 论文：<https://arxiv.org/abs/2509.19115>
- Track-On-R 论文：<https://arxiv.org/abs/2603.12217>
- TAP / TAPNext / BootsTAP / TAPNext++ 官方仓库：<https://github.com/google-deepmind/tapnet>
- TAPNext 官方页：<https://tap-next.github.io/>
- TAPNext++ 论文：<https://arxiv.org/abs/2604.10582>
- TAPNext++ 官方页：<https://tap-next-plus-plus.github.io/>
- BootsTAP 官方页：<https://bootstap.github.io/>
- AllTracker 官方仓库：<https://github.com/aharley/alltracker>
- AllTracker 论文：<https://arxiv.org/abs/2506.07310>
- AllTracker 项目页：<https://alltracker.github.io/>
- TAP-Vid 官方页：<https://tapvid.github.io/>
- TAPVid-3D 官方页：<https://tapvid3d.github.io/>
