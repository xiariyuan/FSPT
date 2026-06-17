# Exporter Schema Audit

**日期**: 2026-06-17

---

## 1. Schema 定义

Exporter `attempt0_export_strided_original_cache.py` 第197-198行：

```python
"original_size": np.array(original_size_hw, dtype=np.int32),
"model_input_size": np.array(original_size_hw, dtype=np.int32),  # same as original
```

**Schema 假设**: `model_input_size == original_size`（no resize step）。

---

## 2. 按 Baseline 评估

| Baseline | model_input_size | exporter schema 正确性 |
|---|---|---|
| CoTracker3 online (cotracker3_video) | = original (384×512+ 实际视频) | ✅ 正确 |
| CoTracker3 offline (cotracker3_window) | = original | ✅ 正确 |
| Track-On2 DINOv3 | = original（如果支持） | ⚠️ 假设正确，但从未执行 |

---

## 3. CoTracker3 验证

从 `cotracker3_online_strided_original.pt` 确认：

```python
original_size: [480, 854]
model_input_size: [480, 854]  # 两者相等 ✅
```

---

## 4. Track-On2 风险

如果未来 Track-On2 要支持 strided+original，需要：
1. 修改 eval script 支持 resize（不只是 SystemExit）
2. 修改 exporter 接收正确的 `model_input_size`（不是 original_size）
3. 修改 metric evaluation 使用 `model_input_size` 缩放

**当前状态**: Track-On2 已在 eval 阶段失败，exporter 未被调用，schema 风险未触发。

---

## 5. 结论

| 项目 | 状态 |
|---|---|
| CoTracker3 exporter schema | ✅ 正确 |
| Track-On2 exporter schema | ⚠️ 未执行，风险待确认 |
| 当前 Phase 0/1 不受影响 | ✅ |
