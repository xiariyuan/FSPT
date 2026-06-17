# Recovery Head Context Check (2026-06-10)

## 已确认的事实

1. **在线 recovery 链路已打通**
   - trigger -> relocalization search -> confidence -> retracking splice 在真实 DAVIS 数据上工作
   - 配置: `configs/fspt_online_recovery_dino_real_eval256.yaml`
   - 审计: `outputs/online_recovery_dino_conf0005_audit.json`

2. **DINO 比 CoTracker 原生特征更强**
   - relocal_conf: DINO 0.018 vs CoTracker 0.002 (~9x)
   - 在 min_conf=0.005 下，DINO 能独占性触发 retracking（74 queries），CoTracker 全部不通过

3. **当前 DINO raw cosine search 没有产生有效几何修正**
   - anchor_t0_error median ≈ base_t0_error median (都是 2.67px)
   - argmax/softargmax/temperature 调整均无效
   - 说明 similarity map peak 就在 base 位置附近，不是 readout 问题

4. **当前不应训练 acceptor/verifier**
   - 现有 acceptor 只做 accept/reject，不回归坐标
   - 训练它只会得到更会"决定是否采用"，但没有新的几何位移可以采用

5. **应训练一个新的 DINO-conditioned geometric recovery head**
   - 冻结 CoTracker3，冻结 DINO backbone
   - 先做 offline anchor prediction
   - 目标：learned head 能否比 raw DINO cosine 更准地预测 re-entry 位置

## 第一轮任务

1. 写 context check（本文件）✓
2. 实现 `scripts/build_online_recovery_anchor_dataset.py` — 数据构建器
3. 小样本 smoke 测试
4. 实现 `scripts/eval_offline_recovery_anchor_baselines.py` — baseline 评测器
