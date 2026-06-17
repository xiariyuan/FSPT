# FSPT 文献综述

这份综述按“和我们当前主线的相似度”来排，不再围绕旧的频率/语义叙事。

当前主线是：

**长遮挡后的 point relocalization / re-detection，配合 MegaDepth 两视图几何预训练，并在 TAP-Vid 官方协议下严格可比地评估。**

---

## 1. 必须对齐的基准论文

- **TAP-Vid: A Benchmark for Tracking Any Point in a Video**，NeurIPS 2022，https://arxiv.org/abs/2211.03726  
  核心作用：定义 TAP 任务、评估协议和官方指标。  
  和我们的关系：这是我们必须严格对齐的 benchmark，不允许为了好看而改协议。  
  和我们的不同：它是基准和评测定义，不是方法论文。

- **TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement**，ICCV 2023，https://arxiv.org/abs/2306.08637  
  核心作用：经典强 baseline，提出逐帧初始化 + 时序精调。  
  和我们的关系：它是最基础的 tracker 参照物之一。  
  和我们的不同：它没有专门针对“长遮挡后重定位”的外部几何预训练故事。

- **PIPs: Persistent Independent Particles**，ECCV 2022，https://arxiv.org/abs/2204.04153  
  核心作用：把 tracking 看成粒子迭代更新。  
  和我们的关系：提供了迭代式更新的思路。  
  和我们的不同：它不是以 re-localization 或两视图几何预训练为中心。

---

## 2. 最接近我们当前方向的近期工作

- **CoTracker**，ECCV 2024，https://arxiv.org/abs/2307.07635  
  核心作用：联合追踪多个点，利用点间交互增强稳定性。  
  和我们的关系：它证明了“把点放在一起建模”可以显著增强鲁棒性。  
  和我们的不同：它的核心是 joint tracking，不是长遮挡后重新找回同一目标。

- **LocoTrack: Local 4D Correlation for Point Tracking**，ECCV 2024，https://arxiv.org/abs/2407.15420  
  核心作用：用局部 4D correlation 做高效 point tracking。  
  和我们的关系：它是强且高效的本地相关性 baseline。  
  和我们的不同：它主要解决局部匹配效率，不是借助 MegaDepth 两视图几何去学重定位能力。

- **CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos**，ICCV 2025，https://arxiv.org/abs/2410.11831  
  核心作用：用真实视频伪标签大幅提升 tracking。  
  和我们的关系：这是必须对齐的强 baseline，也是“现实视频数据规模化”的代表。  
  和我们的不同：它的主线是伪标签扩规模，不是外部几何 pair 预训练；我们更关注 long-occlusion reappearance。

- **ReTracker: Exploring Image Matching for Robust Online Any Point Tracking**，ICCV 2025，论文见本地文件 `references/papers/retracker_iccv2025.pdf`。  
  核心作用：把 image matching 和 robust online tracking 连起来。  
  和我们的关系：这是最贴近“重定位/重检测”语义的一篇近期工作之一。  
  和我们的不同：它强调 online matching；我们更像是离线几何预训练 + 长遮挡恢复，目标是把 reappearance 找回来并保持 TAP-Vid 可比性。

- **Track-On: Transformer-based Online Point Tracking**，ICLR 2025，https://arxiv.org/abs/2501.18487  
  核心作用：在线、因果式 point tracking。  
  和我们的关系：它讨论了流式追踪下的鲁棒性。  
  和我们的不同：它关注 causal online inference，不是我们这里的 long-occlusion relocalization。

- **TAPTRv3: Spatial and Temporal Context Foster Robust Tracking**，arXiv 2024，https://arxiv.org/abs/2411.18671  
  核心作用：用空间上下文和长时序注意力对抗漂移。  
  和我们的关系：它说明“长时序 context”是强信号。  
  和我们的不同：它仍然是上下文增强 tracker；我们更强调“遮挡后重找回”的几何恢复机制。

- **TAPNext++: What’s Next for Tracking Any Point**，arXiv 2026，论文见本地文件 `references/papers/tapnextpp_arxiv2604.10582.pdf`。  
  核心作用：把 tracking 继续往 re-detection / longer-horizon 方向推。  
  和我们的关系：它和我们的主题最接近，尤其在“再出现后怎么找回”这一点上。  
  和我们的不同：它更偏序列建模和 token decoding；我们当前的核心抓手是 MegaDepth 几何 pair 预训练 + TAP-Vid 迁移。

- **Real-World Point Tracking with Verifier-Guided Pseudo-Labeling**，arXiv 2026，论文见本地文件 `references/papers/verifier_guided_pseudolabel_arxiv2603.12217.pdf`。  
  核心作用：通过 verifier 筛选真实视频伪标签。  
  和我们的关系：它说明真实视频扩规模依然是重要方向。  
  和我们的不同：它的核心是伪标签筛选，而不是两视图几何监督。

---

## 3. 几何预训练相关的外部灵感

- **DUSt3R**，CVPR 2024，https://arxiv.org/abs/2312.14132  
  核心作用：从图像对中学习密集几何对应/pointmap。  
  和我们的关系：它证明了“图像对几何监督”这条线是成立的。  
  和我们的不同：它目标是 3D 几何重建，不是 point tracking 的 long-occlusion recovery。

- **MASt3R**，arXiv 2024，https://arxiv.org/abs/2406.09756  
  核心作用：继续推进图像对匹配/重建的几何表示。  
  和我们的关系：它强化了两视图几何预训练作为通用视觉表征的可行性。  
  和我们的不同：它还是偏 3D / matching，不是 TAP-Vid tracking。

---

## 4. 这些工作和我们到底差在哪里

如果把现有工作分成三类，我们属于第三类，而且要把第三类讲清楚：

- 第一类是 **benchmark / baseline tracking**：TAP-Vid、TAPIR、PIPs、CoTracker、LocoTrack、CoTracker3。  
  这类工作的主目标是“把 point tracking 做得更准、更快、更稳”。

- 第二类是 **online / context / pseudo-label scaling**：Track-On、TAPTRv3、ReTracker、Verifier-Guided Pseudo-Labeling、TAPNext++。  
  这类工作的主目标是“把 tracking 做得更强、更长、更适应真实世界”。  
  它们非常接近我们，但通常不是以 MegaDepth 两视图几何预训练为主轴。

- 第三类是我们要做的 **geometry-pretrained long-occlusion relocalization**。  
  关键点不是“要不要输出”或“怎么门控”，而是：
  - 长遮挡后能否找回同一个点
  - 这种能力能否从大规模两视图几何监督中学到
  - 迁移到 TAP-Vid 后能否在官方协议下稳定提升

这就是我们和大多数相似工作的本质区别。

---

## 5. 对我们最有用的论文结论

从这些论文里，最值得我们拿来服务自己故事的结论有四个：

1. **强 tracker 都依赖明确的归纳偏置。**  
   纯堆数据不够，必须有清晰的结构或训练信号。

2. **长时序 context 有用，但不等于 re-localization。**  
   仅仅更会“记住前文”，不一定能把长遮挡后的点重新找回来。

3. **真实视频/伪标签可以增强泛化。**  
   但伪标签路线和几何 pair 路线是不同的故事，不能混着讲。

4. **两视图几何预训练是可以成立的。**  
   这是我们现在最值得压重的方向，因为它和 long-occlusion recovery 的逻辑是兼容的。

---

## 6. 当前判断

如果我们最终能证明：

- MegaDepth 两视图预训练提升了长遮挡后的重定位能力，
- 在 TAP-Vid 官方指标下仍然保持严格可比，
- 对比 CoTracker3 / LocoTrack / ReTracker / TAPTRv3 有稳定优势，

那么这篇论文就不是“又一个 tracker 改进”，而是一个更清晰的故事：

**用大规模几何 pair supervision 学到可恢复的 tracking 表征。**

