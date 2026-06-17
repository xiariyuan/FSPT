# Phase 0 — Protocol Lock Attempt

**日期**: 2026-06-17  
**目录**: `outputs/redetection_ladder_2026-06-17/`  
**状态**: ❌ **BLOCKED — STOP AT PHASE 0**

---

## 1. Phase 0 目标回顾

Phase 0 要求锁死 DAVIS `strided+original` 评测口径，并产出 7 个 mandatory artifacts。

---

## 2. 已完成的盘点

### 2.1 Prediction Cache 盘点

| Baseline | 协议 | 目录 | 记录数 | 状态 |
|---|---|---|---|---|
| trackon2 | first+input (256-space) | `attempt0_2026-06-15_recovery/prediction_caches/` | 30 | ✅ 完整 |
| cotracker3_online | first+input (256-space) | `attempt0_2026-06-15_recovery/prediction_caches/` | 30 | ✅ 完整 |
| cotracker3_offline | first+input (256-space) | `attempt0_2026-06-15_recovery/prediction_caches/` | 30 | ✅ 完整 |
| trackon2 | strided+original | — | 0 | ❌ **不存在** |
| cotracker3_online | strided+original | `cotracker3_baseline_strided_original_cache/` | 0 | ❌ 目录为空 |
| cotracker3_offline | repo-native strided | `cotracker3_offline_davis_cache/davis/cotracker3_offline/` | 30 | ✅ 存在 |

### 2.2 已知限制

**Track-On2 原生限制**（来自 `scripts/attempt0_eval_strided_original.py` 第371行注释）：
> "Track-On2 uses fixed 384x512 input (from config) and cannot handle original video resolution."

这意味着 Track-On2 端到端无法处理 original resolution，必须 resize 后再处理。

### 2.3 DAVIS pkl 可用性

| 来源 | 状态 | 详情 |
|---|---|---|
| GCS (storage.googleapis.com) | ❌ **404 Not Found** | `https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl` |
| GitHub Raw | ⏱️ Timeout | 连接超时 |
| HuggingFace | ⏱️ Timeout | 连接超时 |
| 本地 `/tmp/tapvid_davis_one_video.pkl` | ✅ 存在 | 仅 1 个视频 (bike-packing)，非完整数据集 |
| 本地 `/gemini/code/FSPT/datasets/tapvid_davis/` | ❌ 不存在 | 目录为空 |

**结论**: DAVIS pkl 在所有已知来源均不可用（GCS 404，境外网络超时）。

---

## 3. 根因分析

### 3.1 strided+original 协议无法验证的根本原因

strided+original 评测依赖以下数据：
1. **DAVIS pkl** — 提供 GT points/occluded + original video frames（用于 extract original_size）
2. **模型在 original resolution 下的 predictions** — npz 格式 tracks/visibility

当前状态：
- DAVIS pkl: GCS 404 → **不可用**
- Track-On2 strided+original: 模型固定 384×512 resize → **无法原生支持**
- CoTracker3 strided+original cache: 目录存在但为空 → **从未成功运行**

### 3.2 first+input 协议可以作为替代吗？

**可以，但不满足 Phase 0 的严格要求。**

`first+input` 协议的已知属性：
- 已在 unified cache 中验证与 repo-native 完全一致（AJ delta = 0.00）
- 已在 unified cache 中验证 CoTracker3 与 Track-On2 的 repo-native 对齐

**但 `first+input` ≠ `strided+original`**：
- `first+input`: first-frame query + 256×256 resize + original-size metric evaluation
- `strided+original`: strided queries + original video resolution evaluation
- 两者 query 协议不同（first vs strided），无法互相替代作为 Phase 0 锁死的口径

---

## 4. Phase 0 Stop Criteria 对照

| Stop Criteria | 状态 | 判定 |
|---|---|---|
| 如果 full DAVIS `strided+original` 仍然跑不通，停止后续全部新主线实验 | DAVIS pkl GCS 404 + Track-On2 原生不支持 | **命中** |

**Phase 0 结论: STOP AT PHASE 0**

---

## 5. 已有资产价值确认

虽然 Phase 0 stop，但以下资产仍然有效：

### 5.1 first+input unified caches（已验证）
- Track-On2: AJ=67.04, delta_avg=79.84, OA=92.09 — 与 repo-native 0.00 diff
- CoTracker3 online: AJ=64.89, delta_avg=77.36, OA=91.80
- CoTracker3 offline: repo-native metrics 待确认

### 5.2 re-entry metric protocol（已标准化）
Phase C 的 7 个 re-entry 指标协议可用于所有 future recovery experiments。

### 5.3 CoTracker3 head-to-head 结果
Track-On2 vs CoTracker3 re-entry comparison 已在统一口径下完成。

---

## 6. 决策：是否在 first+input 协议下继续？

Phase 0 Stop Criteria 命中，但需要决策：**是否在 first+input 协议下继续 Phase 1？**

**风险**：
- first+input 的 long-occ 指标可能与 strided+original 有显著差异
- 在错误的口径下做实验 → 结论可能无法泛化到 paper 标准

**建议**：
- **先尝试解决 DAVIS pkl 问题**：
  - 联系 maintainer 更新 GCS URL
  - 在有网络的环境中下载后传输
  - 从其他来源获取
- **如果 24h 内无法解决**：
  - 明确文档化 DAVIS pkl blocker
  - 基于 first+input unified caches 继续，但标注协议限制
  - 不声称达到 paper-level strided+original 标准

---

## 7. 立即可执行的工作

即使 Phase 0 blocked，以下工作不依赖 DAVIS pkl：

1. **使用 first+input unified caches 进行 Oracle diagnostic（Phase 1 的降级版）**
   - 仅测 `first+input` 协议下的 long-occ oracle gap
   - 无法得到 paper-level strided+original 结果，但可以评估方向

2. **Anchor retrieval smoke（Phase 2 预备）**
   - `eval_global_retrieval.py` 可能可以独立于 pkl 运行
   - 需要确认

3. **CoTracker3 offline repo-native metrics 确认**
   - 已有 30 npz，需要用 repo-native 脚本跑出 CoTracker3 offline 的基准数字

---

*Phase 0 blocked by DAVIS pkl unavailability. Next action: attempt DAVIS pkl recovery, or pivot to first+input-based execution with clear protocol caveats.*
