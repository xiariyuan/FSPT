# Final Repair Decision

> **⚠️ 历史记录 / Historical / Superseded**  
> 此文档对应 `redetection_ladder_2026-06-17` 阶段的最终修复决策。  
> 当前 canonical 决策以 `outputs/canonical_redetection_2026-06-18/` 为准。  
> 此状态不应用作当前 Phase gate 依据。

**日期**: 2026-06-18  
**类型**: `PHASE0_PHASE1_FULLY_REPAIRED`

---

## 1. 当前真实协议

**`strided+original`** (不是 `first+input`)

---

## 2. 旧文件归档状态

| 原路径 | 归档位置 | 状态 |
|---|---|---|
| `phase0_protocol_lock.md` | `legacy_first_input/` | ✅ 已归档 |
| `phase0_protocol_lock.json` | `legacy_first_input/` | ✅ 已归档 |
| `phase0_metric_sanity.md` | `legacy_first_input/` | ✅ 已归档 |
| `phase0_metric_sanity.json` | `legacy_first_input/` | ✅ 已归档 |
| `phase0_full_davis_metrics_summary.json` | `legacy_first_input/` | ✅ 已归档 |
| `phase0_full_davis_per_video_metrics.json` | `legacy_first_input/` | ✅ 已归档 |
| `phase1_oracle_upper_bound.md` | `legacy_first_input/` | ✅ 已归档 |
| `phase1_oracle_upper_bound.json` | `legacy_first_input/` | ✅ 已归档 |
| `phase1_oracle_all_frames.json` | `legacy_first_input/` | ✅ 已归档 |

---

## 3. 当前已重建的 Phase 0/1 工件

| 文件 | 协议 | 状态 |
|---|---|---|
| `phase0_protocol_lock.md` | strided+original | ✅ |
| `phase0_protocol_lock.json` | strided+original | ✅ |
| `phase0_metric_sanity.md` | strided+original | ✅ |
| `phase0_metric_sanity.json` | strided+original | ✅ |
| `phase0_full_davis_metrics_summary.json` | strided+original | ✅ |
| `phase0_full_davis_per_video_metrics.json` | strided+original | ✅ |
| `phase0_decision.md` | strided+original | ✅ |
| `phase0_status_live.json` | strided+original | ✅ |
| `phase1_oracle_upper_bound.md` | strided+original | ✅ |
| `phase1_oracle_upper_bound.json` | strided+original | ✅ (recomputed 2026-06-18) |
| `trackon2_strided_original_status.md` | strided+original | ✅ |
| `trackon2_strided_original_status.json` | strided+original | ✅ |
| `exporter_schema_audit.md` | schema | ✅ |
| `exporter_schema_audit.json` | schema | ✅ (trackon2: verified_correct) |
| `repair_manifest.md` | repair log | ✅ |
| `repair_manifest.json` | repair log | ✅ (status: COMPLETE) |

---

## 4. Baseline strided+original 完成状态

| Baseline | Status | AJ | OA | delta_avg | Re-entry Oracle |
|---|---|---|---|---|---|
| CoTracker3 online | ✅ 完成 | 36.95 | 68.15 | 45.01 | n=2736, median=11.65px, 33.6% <4px |
| CoTracker3 offline | ✅ 完成 | 51.54 | 92.15 | 63.59 | n=2736, median=3.15px, 61.0% <4px |
| Track-On2 DINOv3 | ✅ 完成 | 28.38 | 58.33 | 33.23 | n=2736, median=405.04px, 20.4% <4px |

---

## 5. Track-On2 状态细节

- dtype bug: 已修复 (`trackon.py:80-81`)
- Full 30-video run: 完成 (2026-06-18)
- Unified cache: `caches/trackon2_strided_original.pt` (30 records)
- Schema validation: passed (all 30 records)
- Oracle gap: median=405px — 44% of predictions are [0,0] (model marks as occluded)

---

## 6. Phase 1 Stop Criteria 评估

| Stop Criteria | 实测 | 判定 |
|---|---|---|
| Oracle gap too small to justify re-detection | CoTracker3 online: 66.4% queries >4px at re-entry | ❌ 未命中 — significant headroom |
| Offline oracle nearly saturates | CoTracker3 offline: 39.0% queries >4px at re-entry | ❌ 未命中 — still room |

**Phase 1 结论**: ADVANCE_TO_PHASE_2

---

## 7. 结论

```
Decision: PHASE0_PHASE1_FULLY_REPAIRED

Phase 0: ✅ REPAIRED — GO (all 3 baselines confirmed strided+original runnable)
Phase 1: ✅ COMPLETE — Oracle upper bound available for all 3 baselines
Phase 2: ✅ PERMITTED — All blockers resolved
```

---

*Repair complete. Track-On2 full run completed, cache exported, schema validated, oracle computed. Ready for GPT review and Phase 2.*
