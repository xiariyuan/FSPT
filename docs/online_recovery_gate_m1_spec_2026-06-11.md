# M1 Online Recovery Gate Specification (2026-06-11)

## Status: BLOCKED by upstream candidate degeneracy

当前 M1 gate 暂停。原因：online relocalization 产出的 anchor 与 base 几乎相同（差异 < 0.01px），无法构成 accept/reject 决策。只有当 raw top1 non_ambiguous_frac@2px >= 10% 或 oracle_topk better_frac@2px on base_bad >= 20% 时，M1 才能恢复。

当前阶段转为 M0.5 candidate audit，详见 `docs/online_relocal_candidate_audit_m05_spec_2026-06-11.md`。

---

## 目标

在真实 online pipeline 中，学习一个 lightweight gate，决定"是否用 relocal anchor 替换 baseline tracker 输出"。

## 非目标（明确不做）

- 不做 unconditional learned anchor 替换
- 不做 top-k candidate selector
- 不做 end-to-end full model finetune
- 不做新的 sequence/temporal verifier 支线
- 不做 residual refinement（留到 M2）

## 输入特征（全部可在线获得）

从真实 online path 导出，训练时用 offline cache replay：

| 字段 | 来源 | 说明 |
|------|------|------|
| base_xy_norm | base_tracks[b,n,t] | baseline 位置 [0,1] |
| anchor_xy_norm | relocal_anchor_tracks[b,n,t] | relocal anchor 位置 [0,1] |
| delta_xy | anchor_xy - base_xy | 位移向量 |
| delta_norm | \|\|delta_xy\|\| | 位移幅度 |
| relocal_conf | relocal_conf_nt[b,n,t] | relocalization 置信度 |
| base_vis | base_visibility[b,n,t] | base tracker visibility |
| occ_len | 从 relocal_mask 推导 | 遮挡长度（帧数） |
| support_descriptor | DINO pooled support features | support memory 描述子 |
| baseline_patch_embedding | DINO on baseline crop | baseline patch 特征 |
| anchor_patch_embedding | DINO on anchor crop | anchor patch 特征 |

**注意**：第一版先用标量特征（前 6 个），embedding 特征作为扩展。标量特征足够建立 baseline gate。

## 标签定义

| 字段 | 定义 |
|------|------|
| base_err_px | \|\|base_xy - gt_xy\|\| in pixels |
| anchor_err_px | \|\|anchor_xy - gt_xy\|\| in pixels |
| improve_px | base_err_px - anchor_err_px |
| accept_label | 1 if improve_px >= 2px; 0 if improve_px <= -2px; ambiguous if \|improve_px\| < 2px |
| base_good | base_err_px <= 4 |
| base_bad | base_err_px > 16 |
| base_very_bad | base_err_px > 32 |

## 训练指标

- BCE loss on accept_label (ambiguous 样本降权)
- base_bad / base_very_bad 样本增权
- 输出 threshold sweep

## 上线验收标准

| 指标 | 门槛 |
|------|------|
| BaseBad (>16px) median 改善 | >= 5% |
| BaseVeryBad (>32px) better_frac | >= 0.55 |
| BaseGood (<=4px) false override rate | <= 10% |
| 3 seeds 方向一致 | 必须 |

## M1 通过标准

- 数据契约完全无泄漏，parity audit 通过
- offline 上 base_bad 子集有稳定改善
- online smoke 上 gate 优于 AlwaysAccept
- real eval 上全局 AJ 不退化，bad subset 有方向一致改善

## M1 完成后允许进入 M2

- top-k candidates + abstain selector
- residual refinement
- joint finetune（视情况）
