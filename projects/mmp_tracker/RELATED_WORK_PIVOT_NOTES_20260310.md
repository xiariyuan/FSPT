# Related Work → Pivot Notes (2026-03-10)

目的：把“全网搜到的强相关顶会/强工作”的**可执行共识**固定在 repo 里，避免后续在讨论中丢失重点；并把它们直接映射到当前 MMP 主线的真实瓶颈与下一步最小可跑 pivot。

## 当前两类失败（对齐我们自己的现象）

1) **候选里有 headroom，但路由/选择器转不成最终指标**
- v2 典型：gate/selector 召回不足（selected_when_global_better 极低）。
- v3 典型：把 gate “推开”之后，整体反而更差 → 问题变成 **precision/判别力不足**，不是阈值没调好。

2) **long-occlusion re-track 需要更强匹配先验 + 更稳的记忆更新**
- 当前 global retrieval 主要是 descriptor 相似度 + visibility/recency 权重，memory 还没有显式运动一致/几何一致的更新与传播。

## 强相关工作给出的“共识解法”（只保留能直接映射到我们代码/实验的点）

### A. 不要再指望很浅的 gate/scorer “把弱候选选准”
- **ReTracker (ICCV 2025)**：把 long-occlusion re-track 更明确地当作图像匹配问题做，强调更强 backbone + 更强 matching 表征 + 多尺度 refinement，而不是靠浅层的路由器调参来“救精度”。citeturn0search0
- **经验映射**：这与我们 v2→v3 的现象高度一致：当候选本身质量/表征不够强时，gate 只会在 recall/precision 之间来回摆。

### B. 真正拉动局部 refinement 精度的往往是 correlation 表征，而不是更多 heuristic 特征
- **LocoTrack (ECCV 2024)**：主打局部 all-pair correspondence / correlation 表征来抗歧义与重复纹理；核心不是多一个 loss 或多几个标量特征。citeturn0search2
- **经验映射**：我们当前 LocalMatcher/global_refine_matcher 是 “点描述子 vs 搜索窗 patch” 的相关性；patch 模板信息很弱，容易在歧义区域 precision 崩。

### C. memory 真要提升 long-occlusion，更新机制要“可解释且运动一致”
- **SPOT (ICCV 2025)**：强调 streaming memory reading + 更结构化的更新/传播（visibility-guided、motion/flow 相关）。citeturn0search1
- **经验映射**：我们当前 memory 更接近 descriptor queue；positions 虽存，但 retrieval 没用它做几何一致性推理。

### D. 真实视频 / pseudo-label recipe 是硬杠杆（往往比小改架构 ROI 高）
- **BootsTAP (2024)**：teacher-student / bootstrapped 训练在 TAP-Vid 上能显著拉分，即使架构不大改。citeturn0search4
- **CoTracker3 (2025)**：也强调 multi-teacher pseudo label 与真实视频自训练的重要性。citeturn0search3
- **经验映射**：如果目标是外部竞争（而不是“内部机制验证”），recipe 往往会压过小结构改动带来的收益。

## 结合当前 repo：最小可跑 pivot（只做 smoke，不烧大训）

### Pivot-B（先做）：Patch-template 的局部相关性 refine（对齐 LocoTrack 的“相关性表征”方向）
- 动机：在不换 backbone/不引入大模型的前提下，先把 global refine 的“验证/定位”做得更像“局部 all-pair”，改善 precision。
- 落地：新增 `PatchMatcher` + `global_refine_template_mode: patch`，模板用 query-frame 的局部 patch token 集合；匹配时对 token 集合做 log-mean-exp 聚合，再 softmax 得到 offset。
- 配置入口：`projects/mmp_tracker/configs/localglobal_nocommit_patchrefine_dev.yaml`

### Pivot-A（后做）：matching prior / backbone 级别升级（对齐 ReTracker 的“先学匹配”）
- 动机：如果 oracle headroom 本身就弱，或 Pivot-B 也救不了 precision，说明 descriptor/matching prior 真的不够，应该先把 matching 表征升级，再谈路由/记忆。
- 风险：需要引入更强特征（可能涉及外部预训练权重/依赖/数据配方），不适合作为“今晚必须跑完”的最小改动。

## 当前约定（避免证据链混乱）

- 先跑完 full lock-in（`local` vs `nocommit`）锁主模型版本。
- scorer/gate v4 级别微调暂停，除非出现新的强 oracle headroom 证据。
- Pivot-B 只作为 quick smoke：如果它在 dev 上连趋势都没有，就直接 kill，不升级为 full。

