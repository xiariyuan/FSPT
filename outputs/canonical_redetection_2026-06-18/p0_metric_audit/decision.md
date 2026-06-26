# P0 Decision

**日期**: 2026-06-19（updated 2026-06-26）

**Decision**: `RESOLVED_METRIC_RECONCILIATION`

## 当前状态

- ✅ `first_reentry_frame_proxy` 已实现（512/N frame jaccard）
- ✅ `true_AJ_RD` 已实现（报告 `true_AJ_RD` 和 `true_AJ_RD_256` 字段）
- ✅ re-entry metric reconciliation 文档已创建（解释 405px vs 3.91px 等冲突）
- ✅ 坐标 roundtrip 通过
- ✅ Cache schema 一致
- ✅ 256-space AJ_RD 变体已实现
- ✅ query-weighted 聚合已落地
- ✅ AJ_RD@d_min 分解已输出
- ✅ consistency_check 实现（cross-path 重算，独立路径 diff < 1e-4）
- ✅ determinism 通过（diff=0）
- ✅ manifest 已写
- ✅ GT cache identity 已验证

## 口径决定（2026-06-26 拍板）

**论文 canonical 口径：`true_AJ_RD_256`（256-space，TAPNext++ 可比口径）**

- 原分辨率 `true_AJ_RD` 保留为内部诊断口径，不用于论文主表。
- `CURRENT_MAINLINE.md`、`docs/current_mainline_status_2026-06-26.md`、`docs/current_redetection_route_closure_2026-06-26.md` 均已同步此决定。
- P0 解锁条件已满足。

## 口径选择理由

- 256-space 与 TAPNext++ / 公开 baseline 可比
- 原分辨率口径在 DAVIS（480×854）和 Kinetics 间不可直接比较
- `true_AJ_RD_256` 在相同数据上值更高（相同 px 阈值在 256×256 下占比更大），更适合跨论文对比

## 已知遗留项（不阻塞 P0）

- `eval_aj_rd_from_cache.py` 中 `consistency_check` 字典未显式输出 `pass` 字段，`consistency_pass` 变量计算后未落盘。这是代码卫生问题，不影响 P0 口径决定。
- `eval_teacher_reentry_audit.py` 缺 `DEFAULT_AJ_THRESHOLDS` 导入（NameError 崩溃），oracle 选择目标错位（用 min pixel error 而非 argmax-AJ_RD）。这是 Route A 执行工具的问题，不阻塞 P0 口径决定。
