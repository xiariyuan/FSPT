# Online Recovery Status Report (2026-06-10)

## 结论摘要

截至 2026-06-10，基于 CoTracker3 的在线 recovery 主线已经完成了从接口联通到方法瓶颈定位的完整闭环。

当前最稳、最可对外复述的结论是：

1. **真实在线 recovery 链路已打通**
   - 已在 TAP-Vid DAVIS 上验证 `trigger -> relocalization search -> confidence -> retracking splice` 全链路工作。

2. **DINO 作为 recovery-only 特征源，显著提升了 recovery confidence**
   - 在同口径 10-batch DAVIS 审计中，DINO 的 `relocal_conf` 约为 CoTracker 原生特征的 `8x~9x`。

3. **更高的 confidence 能转化为更强的 policy 行为**
   - 在 `min_conf=0.005` 下，CoTracker recovery 无法触发 retracking，而 DINO recovery 仍可触发全部 74 个已命中的 queries。

4. **但最终 recovery 精度尚未改善**
   - 当前 checkpoint 和当前 search/readout 策略下，DINO relocalization 生成的 anchor 与 base tracker 原始位置几乎相同。
   - retracking 从这个 anchor 重启后，也没有在已触发子集上带来稳定的误差下降。

5. **当前瓶颈已定位为方法层面的 signal bottleneck，而非简单的配置或接口问题**
   - `argmax` 与 `softargmax` 几乎等价
   - 单独下调 `conf_threshold` 无效
   - 低 temperature 会显著恶化结果
   - 这说明在当前 query/template 构造下，DINO cosine similarity map 对有效位移的判别力不足

因此，当前方向最合适的定位不是 “已验证的 integrated tracker improvement”，而是：

> **A fully wired online recovery pipeline with strong diagnostic evidence, plus a negative but informative finding: stronger recovery-only features improve confidence and policy activation, but do not yet produce effective geometric correction under the current search/readout design.**

## 一、问题定义与当前边界

本阶段聚焦的问题是：

- 点在 query 时可见
- 后续发生较长遮挡
- 点重新出现时，tracker 是否能在第一时间恢复

当前在线 recovery 主线已经摆脱了 oracle/post-hoc 的推理设定：

- 触发来自 tracker 自身的 visibility/reappearance 规则
- search 在真实视频帧上运行
- retracking 使用 base tracker 重新启动并 splice 尾部

但当前仍**不能**声称：

- recovery 模块已在最终 tracking 指标上稳定提升
- integrated tracker 已具备经过验证的恢复能力

## 二、核心实验工件

### 1. 真实数据链路审计

- `outputs/online_recovery_real_eval_audit.json`
- `outputs/online_recovery_real_eval_lowconf10_audit.json`
- `outputs/online_recovery_cotracker_conf0005_audit.json`
- `outputs/online_recovery_dino_real_eval_audit.json`
- `outputs/online_recovery_dino_conf0005_audit.json`

### 2. Anchor / Retracking 诊断

- `outputs/online_recovery_cotracker_anchor_diag.json`
- `outputs/online_recovery_dino_anchor_diag.json`
- `outputs/anchor_diag_dino_argmax.json`
- `outputs/anchor_diag_dino_softargmax_lowtemp.json`

### 3. 状态文档

- `docs/online_recovery_real_eval_status_2026-06-09.md`
- `docs/online_recovery_dino_status_2026-06-09.md`

## 三、已经被证实的事实

### 3.1 在线 recovery 入口在真实数据上真实生效

来自 `outputs/online_recovery_real_eval_audit.json`：

- DAVIS 20 batches
- `total_queries = 3835`
- `relocal_mask.nonzero_queries = 123`

说明：

- online recovery 不是只在 smoke input 上可跑
- 它在真实数据上确实会识别 reappearance 事件并进入 search

### 3.2 DINO 比 CoTracker 原生特征提供更强的 recovery confidence

同口径 10-batch DAVIS 对比：

- CoTracker:
  - `outputs/online_recovery_real_eval_lowconf10_audit.json`
- DINO:
  - `outputs/online_recovery_dino_real_eval_audit.json`

关键数字：

| 指标 | CoTracker | DINO |
|------|-----------|------|
| total queries | 1856 | 1856 |
| relocal queries | 74 | 74 |
| relocal elements | 75 | 75 |
| relocal_conf mean | 0.00208 | 0.01797 |
| relocal_conf max | 0.00252 | 0.01882 |

解释：

- `relocal_mask` 完全一致，说明触发逻辑一致，差异不是来自 trigger
- DINO 的 `relocal_conf` 提升约 `8.6x`

### 3.3 DINO 的更高 confidence 可以转化为更强的 policy 行为

在 `min_conf=0.005` 下：

- CoTracker:
  - `outputs/online_recovery_cotracker_conf0005_audit.json`
  - `retracking_mask_nonzero_total = 0`
- DINO:
  - `outputs/online_recovery_dino_conf0005_audit.json`
  - `retracking_mask_nonzero_total = 74`

解释：

- CoTracker 的 recovery conf 不足以穿过这个门槛
- DINO 的 recovery conf 足以稳定触发 retracking

这是一个非常干净的结论：

> stronger recovery-only features do improve policy activation

## 四、关键负结果：最终 recovery 精度没有改善

虽然 DINO 触发了 retracking，但从最终精度看，没有观察到可信改善。

这一点不能再用 trigger 或 conf 代替，需要看 anchor / tail 误差。

## 五、Anchor / Tail 诊断结果

### 5.1 触发子集不是“都很简单”，而是强重尾分布

来自：

- `outputs/online_recovery_cotracker_anchor_diag.json`
- `outputs/online_recovery_dino_anchor_diag.json`

以 DINO 为例：

- `base_t0_error median = 2.67 px`
- `base_t0_error mean = 47.04 px`
- `base_t0_error p95 = 243.86 px`

这说明：

- 已触发的 74 个 queries 并非都容易
- 其中存在明显的高误差长尾
- 因此不能仅用 `median=2.67px` 得出“这些 query 本来就很准，所以 recovery 没意义”

更准确的说法是：

> triggered subset is highly bimodal / heavy-tailed

### 5.2 DINO anchor 与 base 位置几乎相同

DINO 诊断：

- `base_t0_error median = 2.6706`
- `anchor_t0_error median = 2.6702`
- `post_retracking_t0_error median = 2.6702`

CoTracker 诊断也类似。

这说明：

- relocalization 生成的 anchor 几乎没有把轨迹从 base tracker 位置拉开
- retracking 在 `t0` 帧也没有引入额外伤害，但也没有修正

### 5.3 Retracking tail 没有产生有效收益

DINO：

- `base_tail_error median = 3.4461`
- `post_retracking_tail_error median = 3.4499`

CoTracker：

- `base_tail_error median = 3.2793`
- `post_retracking_tail_error median = 3.2399`

解释：

- 两者都没有出现“由 recovery anchor 带来的稳定 tail 改善”
- DINO even when triggered does not improve the tail

## 六、低成本消融的最终判断

### 6.1 单独下调 conf threshold 无效

Claude 已对 `relocalization.conf_threshold` 做过扫描，结果几乎不变。

这说明：

- 不是简单地因为 `gate` 太严才导致 anchor 不动
- 即使 gate 放松，候选 step 本身也没有提供有效位移

### 6.2 argmax 与 softargmax 基本等价

来自 `outputs/anchor_diag_dino_argmax.json`：

- `anchor_t0 median = 2.6647`
- `mean = 47.0173`
- `p95 = 243.8632`

与默认 softargmax 几乎一致。

这说明：

- 问题不是 softargmax 抹平了峰值
- similarity peak 本身就位于 base tracker 位置附近

### 6.3 低 temperature 会显著恶化结果

来自 `outputs/anchor_diag_dino_softargmax_lowtemp.json`：

- `anchor_t0 median = 32.36 px`
- `post_retracking_tail_error median = 28.52 px`

这说明：

- 粗暴 sharpen softmax 并不能恢复正确几何
- 相反会放大错误峰值

## 七、当前最可信的技术结论

把以上证据串起来，当前最可信的解释是：

1. DINO recovery-only 特征确实提高了 confidence signal
2. 但在当前 query/template 构造和当前 dense cosine search 下，GT 位置与 base 位置的相似度差异不足以产生有效位移
3. 因此：
   - conf 更高
   - policy 可以更积极
   - 但几何修正几乎没有发生
4. 这不是一个简单的配置错误，也不是一个已知接口 bug
5. 它更像是一个**特征空间与 readout 设计的结构性瓶颈**

## 八、对外可说与不可说

### 可以说

- 我们已经实现并验证了一个真实在线的 recovery pipeline
- DINO 作为 recovery-only feature source 显著提升了 recovery confidence，并能在更合理的阈值下触发 retracking
- 但在当前 search/readout 设计下，这种更强的特征并未转化为有效的几何 correction
- `argmax / temperature / threshold` 这类低成本改动不能解决问题

### 不可以说

- integrated tracker 已被验证在 recovery 指标上稳定提升
- DINO recovery branch 只差一点训练就能工作
- 当前主要瓶颈已经被唯一确定为 trigger coverage

最后这一条尤其重要。当前证据支持：

- coverage 有限
- triggered subset 上的 correction 也无效

因此不能把问题简化成“只需要覆盖更多 query”

## 九、最合理的后续路线

### Route A: 诊断论文 / benchmark 叙事

这是当前最稳的路线。

论文定位可以是：

> benchmark + diagnosis + online recovery wiring + negative finding on current feature/readout design

优点：

- 证据已经很完整
- 口径诚实
- 不需要再投入大规模训练

### Route B: 继续方法线，但需显著工程投入

只有在你明确愿意追加工程预算时才建议继续。

候选方向：

1. 更强 recovery backbone
   - 例如 DINOv2 ViT-B / ViT-L

2. 可训练的 recovery head
   - 不再直接依赖 raw cosine + soft/argmax
   - 学习一个 DINO-conditioned matching / correction head

3. 更强 query/template 构造
   - multi-scale support
   - patch bank
   - multi-crop matching

4. 重新设计 search target
   - 不是“从当前 base 附近做微偏移”
   - 而是面向真正 re-detection 的 coarse-to-fine candidate generation

但这些都已经超出“简单修接口/调参数”的范围。

## 十、最终建议

当前应停止继续试探式方法推进，转入整理与收敛：

1. 保留当前代码与诊断工件
2. 统一口径，写清三层集合：
   - all long-occ / re-entry subset
   - relocal-triggered subset
   - retracked subset
3. 对外采用保守叙事：
   - pipeline works
   - confidence improves
   - geometry does not yet improve
4. 如果继续做方法，必须把目标明确为：
   - a new trainable recovery head / stronger feature backbone
   - 而不是继续扫阈值和小配置
