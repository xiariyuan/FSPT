# Repair Manifest — Protocol Artifact Reconciliation

**日期**: 2026-06-17  
**问题类型**: 协议口径混乱（不是实验失败）  
**状态**: 🚧 IN PROGRESS

---

## 1. 问题描述

Phase 0 / Phase 1 的工件内部存在**两套互相冲突的真相**：

| 问题 | 详情 |
|---|---|
| 旧文件残留 | 旧 `first+input (256-space)` Phase 0/1 文件（已归档到 `legacy_first_input/`） |
| 新文件存在但不完整 | 新 `strided+original` Phase 0/1 文件部分存在，但 Track-On2 状态未知 |
| 冲突结论 | `phase0_decision.md` 写过 `STOP_AT_PHASE_0`，后又改成 `GO`，导致混乱 |
| Phase 1 基于旧协议 | `phase1_oracle_upper_bound.*` 是 `first+input` 结果，当前无效 |

---

## 2. 根本原因

1. 最初发现 DAVIS pkl GCS 404 → 写了 `STOP_AT_PHASE_0`
2. 后发现 pkl 实际在 `/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl`
3. CoTracker3 strided+original 可跑通 → 但没统一重写所有旧文件
4. Track-On2 strided+original **失败**（固定 384×512 输入），但未明确记录
5. Phase 1 oracle 用的是旧 `first+input` 结果，未更新

---

## 3. 归档清单

所有旧 `first+input` Phase 0/1 文件已移至 `legacy_first_input/`:

| 原路径 | 归档路径 | 协议 |
|---|---|---|
| `phase0_protocol_lock.md` | `legacy_first_input/` | first+input |
| `phase0_protocol_lock.json` | `legacy_first_input/` | first+input |
| `phase0_metric_sanity.md` | `legacy_first_input/` | first+input |
| `phase0_metric_sanity.json` | `legacy_first_input/` | first+input |
| `phase0_full_davis_metrics_summary.json` | `legacy_first_input/` | first+input |
| `phase0_full_davis_per_video_metrics.json` | `legacy_first_input/` | first+input |
| `phase1_oracle_upper_bound.md` | `legacy_first_input/` | first+input |
| `phase1_oracle_upper_bound.json` | `legacy_first_input/` | first+input |
| `phase1_oracle_all_frames.json` | `legacy_first_input/` | first+input |

---

## 4. 当前真实状态

| Baseline | strided+original 状态 | 原因 |
|---|---|---|
| CoTracker3 online | ✅ **完成** (AJ=36.95, 30 videos) | model_input_size = original_size |
| CoTracker3 offline | ✅ **完成** (AJ=51.54, 30 videos) | model_input_size = original_size |
| Track-On2 | ❌ **失败** (SystemExit(1)) | 固定 384×512 输入，无法处理 original resolution |

---

## 5. 修复目标

恢复 `redetection_ladder_2026-06-17` 的**单一协议真相**：
- 协议：`strided+original`
- 已完成：CoTracker3 online + offline
- 未完成：Track-On2（不支持）
- Phase 1 oracle：基于 CoTracker3 strided+original cache 计算

---

## 6. Phase 2 状态

**❌ 禁止继续 Phase 2**，直到修复完成并得到 GPT 复核通过。

---

*Repair manifest created. Legacy files archived. New unified artifacts in progress.*
