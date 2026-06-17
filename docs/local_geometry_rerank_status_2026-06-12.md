# Local Geometry Rerank Status (2026-06-12)

## 三句话

1. 这是最后一个正交实验：不训练任何新模型，只用 escort neighbor 几何一致性在现有 top-k 里 rerank。
2. 所有 geometry variants 均未能超过 Stage-2c baseline。
3. **Geometry consistency rerank FAIL。Online recovery 新方法研究线到此结束。**

## Sanity Check（全部通过）

| 检查项 | 结果 |
|---|---|
| Stage-2c reference 复现 | overall diff=-0.09（与 results.json 一致） |
| 样本数 | n=97 ✓ |
| 视频集合 | bike-packing, breakdance, dance-twirl, parkour, pigs ✓ |
| positive n | 20 ✓ |
| base>16 n | 38 ✓ |
| base>32 n | 19 ✓ |

## 主表

| Variant | overall diff | pos_med | b16_med | b16_b2px | oracle_acc | changed | pos_impr | pos_worse |
|---|---|---|---|---|---|---|---|---|
| **stage2c_reference** | **-0.09** | **20.34** | **22.87** | **0.526** | 0.350 | 0 | 0 | 0 |
| geo_translation_nearest | -1.84 | 20.68 | 25.95 | 0.421 | 0.500 | 22 | 6 | 9 |
| geo_affine_nearest | -0.85 | 20.68 | 25.95 | 0.421 | 0.500 | 22 | 5 | 9 |
| geo_affine_nearest_dino_tiebreak | -0.85 | 22.31 | 26.35 | 0.421 | 0.400 | 25 | 5 | 11 |
| geo_affine_top2_dino (exploratory) | -0.85 | 20.68 | 25.95 | 0.421 | 0.450 | 23 | 5 | 10 |

## PASS/FAIL 判定（4 条全部 FAIL）

| Criterion | 需要 | 实际最佳 | 结果 |
|---|---|---|---|
| base>16 better_2px >= 0.526 | 0.526 | 0.421 | **FAIL** |
| positive final median <= 18.0 | 18.0 | 20.68 | **FAIL** |
| overall diff <= 1px | 1.0 | -0.85 | PASS (但无意义) |
| exact oracle acc > 0.25 | 0.25 | 0.500 | PASS (但无意义) |

**核心指标全部 FAIL：geometry rerank 让 base>16 更差，positive median 更高。**

## Findings

1. **Geometry rerank 让 base>16 显著恶化**：b16_b2px 从 0.526 降到 0.421，b16_med 从 22.87 升到 25.95-26.35。所有 variant 一致。

2. **Positive samples 上改善和恶化并存**：5-6 个 positive 改善，但 9-11 个恶化。净效果为负。

3. **Affine ≈ translation**：affine center 和 translation center 选的候选几乎一样。Affine + DINO tiebreak 反而更差（pos_med=22.31）。

4. **Escort neighbor 的几何预测不等于最优候选位置**：geometry center 可能接近某个候选，但那个候选未必是 oracle。当前 top-k 候选的分布与 geometry center 的关系不稳定。

## Changed Positive Cases（5 个代表）

| sample_id | video | s2c_rank/s2c_err | geo_rank/geo_err | oracle_rank/oracle_err | aff_valid | n_escorts | 改善? |
|---|---|---|---|---|---|---|---|
| — | dance-twirl | — | — | — | — | — | — |
| — | parkour | — | — | — | — | — | — |
| — | pigs | — | — | — | — | — | — |

（详细 case 见 `outputs/local_geometry_rerank_audit/debug_changed_cases.json`）

## Open Questions / Assumptions

无。所有已探索的 ranker 方向（scalar score、vector embedding、patch verifier、geometry consistency）均未能在泛化评估中超过 Stage-2c。

## Change Summary

- 新增脚本：`scripts/audit_local_geometry_rerank.py`
- 新增输出：`outputs/local_geometry_rerank_audit/`
- 判定：**FAIL**
- **Online recovery 新方法研究线正式结束。下一步只剩 integration smoke。**
