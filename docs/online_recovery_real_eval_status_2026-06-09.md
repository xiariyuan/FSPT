# Online Recovery Real Eval Status (2026-06-09)

## 实验设置

- Config: `configs/fspt_online_recovery_real_eval256.yaml` (default), `configs/fspt_online_recovery_real_eval_lowconf256.yaml` (low conf)
- Checkpoint: `checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth`
- Dataset: TAP-Vid DAVIS (30 videos, 20 batches)
- Audit: `outputs/online_recovery_real_eval_audit.json`, `outputs/online_recovery_real_eval_lowconf_audit.json`, `outputs/online_recovery_real_eval_lowconf10_audit.json`

## 核心结果

| 项目 | min_conf=0.1 | min_conf=0.001 |
|------|-------------|----------------|
| 总 queries | 3835 | 3835 |
| relocal_mask 非零 queries | 123 | 123 |
| **retracking 触发 batches** | **0** | **9** |
| **retracking mask nonzero** | **0** | **123** |

## Relocal_conf 分布（relocal_mask 非零位置）

| 统计量 | 值 |
|--------|-----|
| count | 124 |
| mean | 0.0021 |
| std | 0.0001 |
| min | 0.0018 |
| max | 0.0025 |

## Min-conf 阈值扫描

| 阈值 | 全部 124 个 conf 值是否通过 |
|------|---------------------------|
| 0.1000 | 全部不通过 |
| 0.0100 | 全部不通过 |
| 0.0050 | 全部不通过 |
| 0.0030 | 全部不通过 |
| 0.0020 | 部分通过（max=0.0025） |
| 0.0010 | **全部通过** |
| 0.0005 | 全部通过 |

## 结论

| 问题 | 答案 |
|------|------|
| online_recovery 入口是否在真实数据上生效？ | **是**，123/3835 queries 触发 |
| retracking 是否在真实数据上触发？ | **是**（min_conf=0.001 时），123 queries 处理 |
| cotracker 特征是否足够做 recovery 主特征？ | relocalization 能触发但 conf 极低（0.002），阈值需极低才能让 retracking 生效 |
| dino 分支是否更适合作为 recovery-only 特征源？ | 已在独立文档中验证；同口径 10-batch 对照下 DINO conf 明显更高 |

## 最新更新

后续对比实验已经补完：

1. `min_conf=0.005` 下：
   - CoTracker recovery 无法触发 retracking
   - DINO recovery 仍可触发全部已命中的 retracked queries

2. 但进一步的 anchor / tail 诊断表明：
   - 已触发子集上，relocalization 生成的 anchor 与 base tracker 位置几乎相同
   - retracking 后的 tail 误差也没有出现稳定改善

因此，当前主线结论不应表述为“已实现最终 recovery 精度提升”，而应表述为：

- CoTracker 原生特征的 recovery conf 太低，不足以支持积极的 policy
- DINO 能显著提升 conf 并触发 policy
- 但当前几何 correction 仍然无效

完整总结请见：

- `docs/online_recovery_status_report_2026-06-10.md`
