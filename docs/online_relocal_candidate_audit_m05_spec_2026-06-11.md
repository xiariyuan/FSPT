# M0.5: Online Relocal Candidate Audit & Rescue (2026-06-11)

## 背景

M1 gate 暂停。原因：candidate degeneracy — 当前 online relocalization 产出的 anchor 与 base 几乎相同（CoTracker 差异=0，DINO 差异<0.01px），无法构成有意义的 accept/reject 决策。

## 任务总目标

在真实 online path 上回答三个问题：

1. 当前 relocalization 的 raw top-1 候选到底有没有离开 base
2. 如果有，是被后处理（gate/fusion/clamp）压回去，还是一开始 similarity map 就塌在 base 附近
3. 如果 top-1 不行，top-k oracle gap 里是否仍存在可利用的机会

## 明确禁止

- 不继续 gate 训练
- 不降 margin 继续训
- 不用 overall AJ 做决策指标
- 不只比较 base 和 final anchor

## Phase 0: 冻结 M1

- M1 gate 状态: BLOCKED
- 原因: candidate degeneracy
- 恢复条件: non-ambiguous candidate 达到最小门槛（raw_top1 non_ambiguous@2px >= 10% 或 oracle_topk better_frac@2px on base_bad >= 20%）

## Phase 1: 给模型加 raw candidate 导出

在 cotracker_refiner.py 的 relocalization 路径中，在 gate 之前和之后分别导出 raw/gated 位移和 top-k candidates。

字段: relocal_raw_step, relocal_gated_step, relocal_topk_positions, relocal_topk_scores, relocal_center_prob, relocal_best_prob_raw, relocal_margin_raw, relocal_gate_value

开关: config flag `relocalization_export_candidates_debug: true`

## Phase 2: Candidate audit 数据集

每样本包含:
- 位置: base_xy, raw_top1_xy, gated_top1_xy, gt_xy, topk_xy
- 误差: base_error_px, raw_top1_error_px, gated_top1_error_px, oracle_topk_error_px
- 分数: center_prob, best_prob_raw, margin_raw, gate_value, topk_scores
- 位移: raw_top1_delta_px, gated_top1_delta_px

## Phase 3: Failure mode 归因

- Type A: raw top1 ≈ base → 问题在 similarity map / feature
- Type B: raw top1 会动，被 gate 压回去
- Type C: top-k oracle 有好候选，但 top1 选错

## 恢复训练的门槛

- raw_top1 non_ambiguous_frac@2px >= 0.10，或
- oracle_topk better_frac@2px on base_bad >= 0.20
