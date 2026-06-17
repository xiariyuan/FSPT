# Neighbor-Deviation Baseline-First Pseudo-Label Checklist (2026-06-13)

## 0. 结论先行

继续做阶段 2，但只能按下面这条线继续：

**predicted-track neighbor deviation -> teacher-side signal diagnosis -> coverage-matched pseudo-label filtering -> official baseline metrics**

不要再把 selective / calibration 当主成功标准。它们现在只保留为辅助解释。

---

## 1. 先同步一个关键修正

`neighbor_deviation` 在伪标签训练里原本被错误地当成了“confidence 越大越好”的分数直接阈值化。这个方向是错的，因为它本质上是 `risk`，语义是：

- `higher neighbor_deviation = higher risk = worse`
- 训练筛选需要的是 `confidence / reliability`

当前最新实现已经修正：

- [advanced_training.py](/gemini/code/FSPT/utils/advanced_training.py)

现在 trainer 会自动写入：

- `neighbor_deviation_raw_risk`
- `neighbor_deviation_risk`
- `neighbor_deviation_confidence`
- `neighbor_deviation_reliability`

其中真正用于伪标签 hard filter 的应是：

- `neighbor_deviation_confidence = 1 - normalized_risk`

所以：

1. 所有在这次修正前跑的 `neighbor_deviation` pseudo-label filtering 训练结果都不要作为正式证据。
2. 阶段 2 之后的所有训练，必须基于当前最新版代码继续。

---

## 2. 主目标

这条线唯一的主判据是：

**能不能在原 baseline 官方主指标上稳定赢。**

主指标只看：

- `AJ`
- `OA`
- `<avg`
- `<4px`

优先数据集：

1. `TAP-Vid DAVIS`
2. `TAP-Vid Kinetics`，如果 baseline 论文/当前仓库的主表也同时报告它

以下内容不是主成功标准：

- selective AJ
- AURC
- ECE
- Brier
- long-occ selective 表

这些只能在主指标有正增益以后，作为补充分析写进文中。

---

## 3. 冻结协议

在开跑前先冻结这几个东西，不要边跑边换：

### 3.1 Base 训练协议

沿用现有 verifier-guided pseudo-label training 路径，不重开新 trainer：

- 入口：
  - [train.py](/gemini/code/FSPT/train.py)
  - [advanced_training.py](/gemini/code/FSPT/utils/advanced_training.py)
- 现有参考 config：
  - [fspt_verifier_guided_pseudo_labeling.yaml](/gemini/code/FSPT/configs/fspt_verifier_guided_pseudo_labeling.yaml)
  - [fspt_verifier_guided_pseudo_labeling_pred_visibility.yaml](/gemini/code/FSPT/configs/fspt_verifier_guided_pseudo_labeling_pred_visibility.yaml)

### 3.2 评估协议

固定：

- `query_mode = strided`
- `metric_resolution_mode = original`
- 与当前 baseline 保持完全一致的 eval protocol

### 3.3 主对照组

必须保留这几个对照：

1. `No filter / All visible`
2. `pred_visibility hard filter`
3. `verifier_scores hard filter`
4. `neighbor_deviation hard filter`
5. `neighbor_deviation soft weighting`，仅当前 4 通过后再做

---

## 4. 阶段 2：teacher-side signal diagnosis

这一阶段不训练学生模型，只回答一个问题：

**predicted-track neighbor deviation 在 teacher 预测轨迹上，是否真能提供适合 pseudo-label 筛选的无 GT 风险信号？**

### 4.1 要做什么

在一个有 GT 的验证集上运行 teacher，基于 teacher 的 `pred_tracks + pred_visibility` 计算：

- `neighbor_deviation_raw_risk`
- `neighbor_deviation_confidence = 1 - risk`

然后和真实 tracking error 对齐，做信号质量诊断。

### 4.2 数据

至少先做：

- `TAP-Vid DAVIS`

如果成本不高，再补：

- `TAP-Vid Kinetics`

### 4.3 需要产出的指标

对每个 score source 都报：

- `pred_visibility`
- `verifier_scores`，如果 teacher 路径里可直接取到
- `neighbor_deviation_raw_risk`
- `neighbor_deviation_confidence`

必报：

- overall `ROC-AUC` for `high_error`
- overall `Spearman(error, score)`
- long-occ subset `ROC-AUC`
- coverage-aligned accept table：`90% / 70% / 50% coverage`
- accept set 的 mean error / median error / `<4px`
- reject set 的 high-error enrichment

`high_error` 阈值先固定为当前已经使用过的：

- `error > 16px`

### 4.4 成功标准

只有满足下面条件才进入阶段 3：

1. `neighbor_deviation` 在 `long_occ` 上明显优于 `pred_visibility`
2. overall 上至少不劣于 `pred_visibility`
3. coverage-aligned accept 表里，`neighbor_deviation` 的 accept set error quality 不能明显差于 `pred_visibility`

### 4.5 失败即停

如果阶段 2 发现：

- overall 和 `pred_visibility` 基本持平甚至更差
- long-occ 优势不稳定
- accept set 纯度没有优势

那就停止这条线，不进入训练矩阵。

不要用 selective/calibration 再包装成“仍然有研究价值”。

### 4.6 建议工件

- `outputs/predicted_neighbor_deviation_signal_v1/`
- `docs/predicted_neighbor_deviation_signal_v1_status_2026-06-13.md`

---

## 5. 阶段 2.5：在 pseudo dataset 上做 coverage 对齐

这是必须做的，不能跳过。

原因很简单：

- 训练时真正被筛的是 pseudo dataset，不是 DAVIS
- 不同分数源的数值尺度不同
- 直接拿同一个数值阈值做对比不公平

### 5.1 要做什么

在实际 pseudo-label 数据源上扫描 teacher 输出分布，按 source 分别计算 quantile 阈值。

source 至少包括：

1. `pred_visibility`
2. `verifier_scores`
3. `neighbor_deviation_confidence`

目标 coverage 先固定为：

- `0.90`
- `0.70`

`0.50` 只在前两档出现明显正信号时再加。

### 5.2 输出

每个 source 都产出：

- coverage -> threshold 对照表
- 实际保留比例
- 各分位统计

### 5.3 建议工件

- `outputs/pseudo_confidence_quantiles_v1/`
- `docs/pseudo_confidence_quantiles_v1_status_2026-06-13.md`

---

## 6. 阶段 3：训练矩阵

阶段 3 的目标不是大规模扫参，而是最小化实验数回答：

**neighbor_deviation filtering 是否比 baseline filtering 更能提升官方主指标。**

### 6.1 第一轮：hard filter smoke

先只做短程 smoke，不直接 full train。

#### 组 A：90% coverage

1. `A0_all_visible`
   - 所有 predicted visible 帧都保留
   - 可实现为 `pred_visibility` + 极低阈值，等价于“不做额外筛选”
2. `A1_pred_visibility_c90`
3. `A2_verifier_scores_c90`
4. `A3_neighbor_deviation_c90`

#### 组 B：70% coverage

5. `B1_pred_visibility_c70`
6. `B2_verifier_scores_c70`
7. `B3_neighbor_deviation_c70`

### 6.2 第一轮 smoke 的判据

只看官方主指标：

- `AJ`
- `OA`
- `<avg`
- `<4px`

保留规则：

1. `neighbor_deviation` 变体必须至少优于 `A0_all_visible`
2. 同时至少不差于 `pred_visibility` 对照
3. 如果 `AJ` 没增益但 `<4px` 和 `<avg` 明显更好，可以保留为候选
4. 如果 `AJ`、`<4px`、`<avg` 都没有优势，直接停，不进入 full run

### 6.3 第二轮：full run

只把第一轮 smoke 最好的两个 variant 提升到 full run。

建议最多：

1. `best_neighbor_deviation_variant`
2. `best_baseline_control_variant`

每个至少跑：

- `2 seeds`

### 6.4 Soft weighting 何时做

`neighbor_deviation soft weighting` 只有在 hard filter 版本已经出现正向主指标增益时才做。

否则不要提前开。

原因：

- 当前 trainer 的 hard filter 逻辑已经现成
- soft weighting 需要额外改损失权重路径
- 如果 hard filter 都没信号，soft weighting 大概率只是更贵的噪声

---

## 7. 阶段 4：主结果判据

最终是否继续这条线，只按下面规则判断。

### 7.1 继续条件

满足任一即可继续深挖：

1. `AJ` 相比 `A0_all_visible` 和 `pred_visibility` 对照都有稳定正增益
2. `AJ` 至少小幅正增益，同时 `<4px` 和 `<avg` 也同步改善
3. 2 个 seed 上方向一致，没有明显 `OA` 伤害

### 7.2 停机条件

出现下面任一项就停止：

1. 最好变体的 `AJ` 仍然没有清晰正增益
2. 只有 selective / long-occ / calibration 好看，但官方主指标没有赢
3. 结果严重依赖单 seed
4. `OA` 明显下降，属于用过滤换取位置指标的假提升

### 7.3 建议的量化门槛

为了避免拿噪声当提升，建议用这套标准：

- 若 2-seed mean `AJ` 增益 `< +0.001`，视为不够强
- 或者虽未到 `+0.001`，但 2 个 seed 都同时满足：
  - `AJ > control`
  - `<4px > control`
  - `OA` 不低于 control `-0.001`

否则停止。

---

## 8. 主结果之后才能补的分析

只有当阶段 4 通过，下面这些分析才值得补到论文：

1. `long_occ` 子集增益
2. pseudo-label 保留样本的 error 分布变化
3. reliability / calibration 附表
4. risk-coverage 曲线

顺序不能反过来。

---

## 9. 具体交付物

Claude 这一轮执行后，至少要交付：

### 9.1 阶段 2

- `outputs/predicted_neighbor_deviation_signal_v1/`
- `docs/predicted_neighbor_deviation_signal_v1_status_2026-06-13.md`

### 9.2 阶段 2.5

- `outputs/pseudo_confidence_quantiles_v1/`
- `docs/pseudo_confidence_quantiles_v1_status_2026-06-13.md`

### 9.3 阶段 3-4

- `outputs/neighbor_deviation_pseudolabel_baselinefirst_v1/`
- `docs/neighbor_deviation_pseudolabel_baselinefirst_v1_status_2026-06-13.md`

这个最终 status 文档必须包含：

1. 训练矩阵表
2. 每个 variant 的 official metrics
3. 最佳 variant vs baseline 的 delta 表
4. 是否继续/停止的明确结论

---

## 10. 可以直接发给 Claude 的一句话

继续做阶段 2，但必须切到 baseline-first 协议：先在 teacher 预测轨迹上验证 `predicted neighbor deviation` 是否真是有效的无 GT risk signal，再在 pseudo dataset 上按 coverage 对齐阈值，只用官方 `AJ / OA / <avg / <4px` 比较 `all_visible / pred_visibility / verifier_scores / neighbor_deviation` 几组 hard filter；如果主指标没有清晰稳定正增益，就停止，不要再用 selective 或 calibration 包装成功。
