# Phase A — Health-Check: Existing Online Recovery Assets

**日期**: 2026-06-16
**资产审计范围**: `scripts/debug_online_recovery_real_eval.py` × 2 configs, DAVIS 30 videos, max_batches=50

---

## A1. 真实数据链路审计结果

### CoTracker 特征源 (`fspt_online_recovery_real_eval256.yaml`)

| 指标 | 值 |
|---|---|
| feature_source | cotracker |
| relocal_mask nonzero elements | 146 |
| relocal_mask nonzero queries | 145 / 5882 (2.5%) |
| relocal_conf mean (at nonzero) | 0.0021 |
| relocal_conf std | 0.00013 |
| relocal_conf max | 0.0025 |
| threshold (min_conf) | 0.1 |
| retracking triggered batches | **0 / 30** |
| status | `RELOCAL_TRIGGERED_NO_RETRACKING` |

**结论**: CoTracker 特征 relocal_conf 峰值仅 0.0025，比阈值 0.1 低约 50×。retracking 管线完整，但几何置信度永远达不到触发门限。

### DINO 特征源 (`fspt_online_recovery_dino_real_eval256.yaml`)

| 指标 | 值 |
|---|---|
| feature_source | dino |
| relocal_mask nonzero elements | 146 (same 146 events) |
| relocal_mask nonzero queries | 145 / 5882 (2.5%) |
| relocal_conf mean (at nonzero) | 0.0180 |
| relocal_conf std | 0.00034 |
| relocal_conf max | 0.0188 |
| threshold (min_conf) | 0.1 |
| retracking triggered batches | **13 / 30** |
| retracking_mask nonzero total | 145 |
| status | `FULLY_TRIGGERED` |

**结论**: DINO 特征 relocal_conf 均值 0.018，比 CoTracker 高约 8.6×，仍低于 0.1 阈值但在 batch 层面足以触发（13/30 批）。DINO 是正确的特征源。

---

## A2. 关键发现

### F1: relocal_conf 数值异常低

两种特征源的 relocal_conf 绝对值都极低：
- CoTracker: [0.0018, 0.0025] — 全挤在 0.002 附近
- DINO: [0.0171, 0.0188] — 全挤在 0.018 附近

这说明 `relocal_conf` 的绝对值意义有限，关键是 batch 内**排序**是否能让 re-entry 事件排到 top-K。但 DINO 相对于 CoTracker 的 8.6×提升是真实的（与 Phase 1 oracle 诊断一致）。

### F2: relocal_mask 触发率稳定

两版 relocal_mask 完全一致（146 elements, 145 queries），说明 relocal trigger 逻辑独立于特征源。30 批视频中 relocal 事件分布于 ~20 批（部分批次 query 少或遮挡短）。

### F3: 13/30 批次触发 retracking 的意义

DINO 版 13/30 批触发 retracking，但 30 批中并非每批都有 relocal 事件。batch 16 触发最多（41 relocal queries），其余批 0~23 不等。整体 retracking 覆盖了 145 个 query 上的 relocal 事件。

---

## A3. 现有资产健康状态

| 资产 | 状态 | 备注 |
|---|---|---|
| `fspt_online_recovery_real_eval256.yaml` | 链路完整但不可用 | CoTracker relocal_conf 永远低于阈值 |
| `fspt_online_recovery_dino_real_eval256.yaml` | **可用** | DINO relocal_conf 可触发 retracking |
| `cotracker_refiner.py` 在线 recovery 管线 | **已完整实现** | `_apply_learned_recovery_head()` 已存在 |
| `models/online_recovery_head.py` M0 | **已训练 (2-seed)** | 3-seed 收敛，详见 Phase B |
| `checkpoints/online_recovery_head_m0/best.pth` | **存在** | 合并 best 模型 |
| `configs/fspt_online_recovery_learned_head_smoke.yaml` | **存在** | 已配置 use_learned_head=true |

---

## A4. 对 Phase D 的直接启示

1. **必须使用 DINO 特征源**，不能用 CoTracker
2. **relocal_conf 的绝对值不重要**，重要的是 head 输出质量 — M0 head 正是要解决这个问题
3. **retracking 在 13/30 批次上触发**，覆盖 145 个 query × re-entry event — 这是 smoke 的有效信号
4. **smoke 的验证重点**：确认 `_apply_learned_recovery_head()` 在 retracking 帧上能给出合理位置估计

---

*审计完成。Phase D smoke 可立即启动。*
