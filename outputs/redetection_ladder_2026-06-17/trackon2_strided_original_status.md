# Track-On2 strided+original Status

**日期**: 2026-06-17  
**结论**: ⚠️ **Runtime bug detected, partial probe success, needs rerun**

---

## 1. 事件时间线

| 阶段 | 状态 | 详情 |
|---|---|---|
| 初次执行 | ❌ 运行时报错 | dtype mismatch: `Index put requires the source and destination dtypes match` |
| 修复后 probe | ✅ 开始产出 npz | 本地验证写出了 14+ npz 文件 |
| 完整运行 | ❌ 尚未完成 | 需要完整 30-video run + unified cache export |

---

## 2. 根因（经本地验证修正）

### 2.1 原始报错栈

- `baselines/track_on/model/trackon.py:89`
- `baselines/track_on/model/trackon_predictor.py:192`
- 报错: `Index put requires the source and destination dtypes match`
- 原因: `point_memory` 和 `q_features` 的 dtype 不一致

### 2.2 已有的修复（由用户本地验证）

- 位置: `baselines/track_on/model/trackon.py:80` 附近
- 内容: 让 `q_features` 对齐到 `point_memory.dtype`
- 验证: 修复后 Track-On2 开始实际运行 strided+original，写出了 npz 文件

---

## 3. 修复后 probe 结果

| 检查项 | 结果 |
|---|---|
| 修复后运行 | ✅ 开始写 npz |
| 写出文件数 | 至少 14 个 (`/tmp/trackon2_strided_probe2/davis/trackon2/*.npz`) |
| 完整 30-video run | ❌ 尚未完成（probe 中途观察） |
| unified .pt cache | ❌ 不存在（需要 exporter） |

---

## 4. 当前状态

**`needs_rerun`** — 不是 unsupported，是需要在 bug 修复后完整重跑。

| 检查项 | 结果 |
|---|---|
| 协议天然不支持？ | ❌ **不是** — dtype bug 修复后可运行 |
| 完整 run 完成？ | ❌ — 只跑了部分 |
| cache 存在？ | ❌ — `/tmp` 为 probe 临时目录 |
| eval script 需要修改？ | ⚠️ — eval script 本身没问题（第371行 SystemExit 是前一次调用时的旧逻辑，已被绕过） |

---

## 5. 需要的操作

| 步骤 | 操作 |
|---|---|
| 1 | 确认 `trackon.py` dtype fix 已保留（非临时） |
| 2 | 完整运行 30-video strided+original eval |
| 3 | 导出 unified .pt cache |
| 4 | 计算 Phase 1 oracle metrics |

---

## 6. 对 Phase 0/1 判定的影响

- Phase 0: ✅ 不受影响（至少一个 baseline 已跑通）
- Phase 1: 当前仅基于 CoTracker3，**不完整**（缺少 Track-On2 对比）
- Phase 2: ⏸️ **不能放行** — 需要等 Track-On2 strided probe 状态重新落账

---
