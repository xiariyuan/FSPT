# Phase C — Re-Entry Metric Protocol Standardization

**日期**: 2026-06-16
**协议版本**: v1.0

---

## C1. 背景

Phase 1 oracle 诊断揭示了核心矛盾：oracle_mask 在 long_occ_AJ 上 +3pp，但 re-entry 恶化 68-79%。Phase D integration smoke 必须用**统一、可比较的 re-entry 指标**评估 head 效果。

---

## C2. Re-Entry 事件定义

### C2.1 Re-Entry Frame

对每个 query `n` 在视频 `b` 中，以 GT visibility `V[b,n,t]`（True=visible）为 ground truth：

1. 从 query time `t_q` 开始往后扫描
2. 遇到第一个 `V[b,n,t]=True` 且 `t > t_q`：这不是 re-entry（query 本来就在可见）
3. 遇到第一个 `V[b,n,t]=False`：标记为 in-occlusion
4. 之后遇到的第一个 `V[b,n,t]=True`：标记为 **re-entry frame** `t_re`
5. 重复以上过程得到所有 re-entry frames

### C2.2 Re-Entry Query

有至少一个 re-entry frame 的 query。long-occ re-entry query: occlusion run length ≥ min_occlusion_len（通常 20 帧）。

---

## C3. 标准 Re-Entry 指标（7 个）

### C3.1 基础计数

| 指标名 | 定义 | 用途 |
|---|---|---|
| `n_queries_with_reentry` | 有 re-entry frame 的 query 总数 | 信号丰富度 |
| `n_retracked` | 经 retracking 的 re-entry query 数 | head 覆盖率 |
| `n_non_retracked` | 未经 retracking 的 re-entry query 数 | baseline 对照 |

### C3.2 误差指标

所有误差在 **256×256 像素空间**计算（input resolution），归一化坐标转换后乘以 256。

| 指标名 | 定义 | 用途 |
|---|---|---|
| `reentry_first_error_mean_px` | 所有 re-entry query 的 `error(t_re[0])` 的均值 | 主指标，re-entry 精度 |
| `reentry_first_error_median_px` | 所有 re-entry query 的 `error(t_re[0])` 的中位数 | 对离群值鲁棒的主指标 |
| `reentry_<4px` | `error(t_re[0]) < 4px` 的 re-entry query 占比（×100） | 精确度达标率 |
| `reentry_<8px` | `error(t_re[0]) < 8px` 的 re-entry query 占比（×100） | 宽松精度达标率 |
| `reentry_p95_px` | 所有 `error(t_re[0])` 的 95th percentile | 尾部精度 |

### C3.3 Retracking 对比

| 指标名 | 定义 | 用途 |
|---|---|---|
| `retracked_first_error_median_px` | 经 retracking 的 re-entry query 的 first-error 中位数 | head 贡献 |
| `non_retracked_first_error_median_px` | 未 retracking 的 re-entry query 的 first-error 中位数 | baseline 对照 |

---

## C4. 精度对比协议

Phase D 输出**必须**包含以下对比：

```
reentry_first_error_median_px
  vs. base（无 learned head）
  vs. M0 head（learned head）

retracked_first_error_median_px   ← M0 head 贡献
non_retracked_first_error_median_px  ← 未覆盖的 query

n_retracked / n_queries_with_reentry  ← head 触发率
```

**成功标准（Phase D smoke）**：
- `retracked_first_error_median < non_retracked_first_error_median` 且差值 > 0（方向正确）
- retracking 覆盖 re-entry query 的合理比例（非 0 也非 100%）

---

## C5. 与 Phase 1 协议对齐

Phase 1 decision.md 中的 re-entry 指标定义：

| Phase 1 指标 | 本协议对应 | 说明 |
|---|---|---|
| `reentry_mean_error_px` | `reentry_first_error_mean_px` | 完全对齐 |
| `reentry_median_error_px` | `reentry_first_error_median_px` | 完全对齐 |
| `reentry_<4px` | `reentry_<4px` | 完全对齐 |
| `reentry_<8px` | `reentry_<8px` | 完全对齐 |
| (count=1.5) | `n_queries_with_reentry` | 扩展为可报告绝对数 |

Phase 1 的 `count=1.5` 是 DAVIS 上 long-occ re-entry 事件极其稀疏导致的均值。Phase D smoke 应报告绝对数量，并在 full eval 时报告均值。

---

## C6. 实施

### C6.1 现有脚本

`scripts/eval_recovery_position_error.py` 已实现本协议 7 个指标中的 6 个（缺 `reentry_p95_px`）。Phase D 集成 smoke 可直接复用，输出 JSON 包含：

- baseline 条件 re-entry 指标
- dino_recovery 条件 re-entry 指标
- （Phase D smoke 扩展）learned_head 条件 re-entry 指标

### C6.2 输出格式

```json
{
  "reentry_metrics": {
    "n_queries_with_reentry": 45,
    "n_retracked": 22,
    "n_non_retracked": 23,
    "reentry_first_error_mean_px": 18.4,
    "reentry_first_error_median_px": 12.3,
    "reentry_<4px": 31.1,
    "reentry_<8px": 44.4,
    "reentry_p95_px": 52.7,
    "retracked_first_error_median_px": 9.1,
    "non_retracked_first_error_median_px": 15.8
  }
}
```

---

## C7. Stop Criteria（re-entry 维度）

以下任一情况 → 建议停止对 M0 head 的继续投入：

| 条件 | 阈值 |
|---|---|
| retracked median ≥ non-retracked median | head 方向错误 |
| `n_retracked / n_queries_with_reentry` | ≥95% 或 ≤5% 覆盖不合理 |
| head 输出 p95 > 64px | 几何精度极差 |
| improvement vs non-retracked | < 2px median 改善 |

---

*协议标准化完成。Phase D smoke 将输出符合此协议的 JSON + 人类可读摘要。*
