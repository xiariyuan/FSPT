# Final Repair Decision

**日期**: 2026-06-17  
**类型**: `PHASE0_REPAIRED_BUT_PHASE1_PARTIAL`

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
| `phase0_protocol_lock.md` | strided+original | ✅ 新写 |
| `phase0_protocol_lock.json` | strided+original | ✅ 新写 |
| `phase0_metric_sanity.md` | strided+original | ✅ 新写 |
| `phase0_metric_sanity.json` | strided+original | ✅ 新写 |
| `phase0_full_davis_metrics_summary.json` | strided+original | ✅ 新写 |
| `phase0_full_davis_per_video_metrics.json` | strided+original | ✅ 新写 |
| `phase0_decision.md` | strided+original | ✅ 新写 |
| `phase0_status_live.json` | strided+original | ✅ 新写 |
| `phase1_oracle_upper_bound.md` | strided+original | ✅ 新写 |
| `phase1_oracle_upper_bound.json` | strided+original | ✅ 新写 (in-place update) |
| `phase1_oracle_all_frames.json` | strided+original | ✅ 新写 (in-place update) |
| `trackon2_strided_original_status.md` | strided+original | ✅ 新写 |
| `trackon2_strided_original_status.json` | strided+original | ✅ 新写 |
| `exporter_schema_audit.md` | schema | ✅ 新写 |
| `exporter_schema_audit.json` | schema | ✅ 新写 |
| `repair_manifest.md` | repair log | ✅ 新写 |
| `repair_manifest.json` | repair log | ✅ 新写 |

---

## 4. Baseline strided+original 完成状态

| Baseline | Status | AJ | Re-entry Oracle Gap |
|---|---|---|---|
| CoTracker3 online | ✅ **完成** | 36.95 | mean=14.73px, 48.7% >4px |
| CoTracker3 offline | ✅ **完成** | 51.54 | mean=12.21px, 46.2% >4px |
| Track-On2 DINOv3 | ⚠️ **needs_rerun** | — | — (dtype bug fixed locally, probe shows 14+ npz) |

---

## 5. Phase 1 Stop Criteria 评估

| Stop Criteria | CoTracker3 offline 实测 | 判定 |
|---|---|---|
| long-occ AJ barely improves | oracle gap mean ≈ 12px | ❌ 未命中 |
| re-entry error barely decreases | oracle gap mean ≈ 12-21px | ❌ 未命中 |

**Phase 1 结论**: ADVANCE_TO_PHASE_2

---

## 6. Track-On2 状态

Track-On2 DINOv3 在 `strided+original` 协议下存在 dtype runtime bug，但已由用户本地修复，修复后 probe 可运行（已写出 14+ npz）。当前状态为 `needs_rerun`，需完整 30-video run + unified cache export + oracle metrics。不能在 Track-On2 strided probe 状态重新落账前直接放行 Phase 2。

---

## 7. 单一真相版本

| 协议 | 有效工件 |
|---|---|
| `legacy_first_input/` | 归档（旧版本，仅供参考） |
| **根目录 `*.md / *.json`** | **单一真相（strided+original）** |

---

## 8. 结论

```
Decision: PHASE0_REPAIRED_BUT_PHASE1_PARTIAL

Phase 0: ✅ REPAIRED — GO (CoTracker3 confirmed strided+original runnable)
Phase 1: ⚠️ PARTIAL — CoTracker3-only oracle available, Track-On2 needs rerun
Phase 2: ❌ NOT PERMITTED — Track-On2 strided probe status needs re-accounting
```

**Phase 2 放行条件**: Track-On2 strided+original 完整 run 结果落盘，并纳入 Phase 1 oracle comparison，之后才可放行。

---

*Repair complete. Track-On2 status corrected from failed_unsupported to needs_rerun. Ready for GPT review.*
