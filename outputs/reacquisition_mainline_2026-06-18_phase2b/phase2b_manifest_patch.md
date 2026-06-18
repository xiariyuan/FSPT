# Phase 2b — Manifest Patch: Corrected Conclusions

**日期**: 2026-06-18  
**补丁原因**: 真实协议 (`strided+original`) 下原有结论失效，需修正

---

## 1. 已修正的结论

### 1.1 Full CoTracker3 Hybrid 路线判断

**❌ 旧结论** (来自 `first+input` 协议):
> "head-to-head: Track-On2 总体更强，CoTracker3 局部信号不足"

**✅ 新结论** (来自 `strided+original` 真实协议):
> **旧 head-to-head 结论作废。** 真实协议下 CoTracker3 offline 在 re-entry 帧上显著强于 Track-On2 (median 3.2px vs 6.5px, <8px: 83.3% vs 56.1%)。但当前 DINOv2 query-frame anchor whole-frame retrieval 变体同样失败 (top5@16px=1.6%)，说明问题不在 base model 选择，而在 candidate generation 方法本身。

### 1.2 Track-On2 Strided+Original 协议可行性

**❌ 旧结论**:
> "Track-On2 uses fixed 384x512 input and cannot handle original video resolution"

**✅ 新结论**:
> **Track-On2 完全支持 original resolution。** 输入坐标由 original H/W 归一化，输出坐标由 model H/W 反归一化回 original 空间。dtype bug 修复后 30-video full run 成功 (AJ=28.38)。该脚本注释为误导性描述，不影响实际执行。

### 1.3 Cache 状态

**❌ 旧**:
> Track-On2 strided+original cache: ❌ 不存在  
> CoTracker3 offline strided+original cache: ❌ 不存在

**✅ 新**:
> 全部三个 baseline 的 strided+original unified cache 均已完成：
> - `caches/trackon2_strided_original.pt` ✅ (30 records)
> - `caches/cotracker3_online_strided_original.pt` ✅ (在 `outputs/redetection_ladder_2026-06-17/caches/`)
> - `caches/cotracker3_offline_strided_original.pt` ✅ (在 `outputs/redetection_ladder_2026-06-17/caches/`)

---

## 2. 当前 Phase 2 状态

### 2.1 已执行的 Phase 2 变体

| 变体 | 方法 | Top5@16px | 状态 |
|---|---|---|---|
| query-frame anchor whole-frame retrieval | DINOv2 query-frame GT patch → whole-frame dense match | 1.6% | ❌ **STOP** (两数量级失败) |
| last-visible multi-support (mean/max) | Last-visible/multi-frame anchor → whole-frame dense match | 0.0-2.3% | ❌ **STOP** (纯 DINOv2 整帧检索死路) |
| CoTracker3 offline centered local search | Last-visible anchor → local search around CT-offline pred | TBD | ⏳ 执行中 |

### 2.2 清晰的问题归属

```
问题不在: candidate ranking / verifier / base model
问题在:   candidate generation 方法 → GT 没进 top-k，verifier 没有输入
```

---

## 3. 未被禁用的有效路线

| 方向 | 依据 | 状态 |
|---|---|---|
| CoTracker3-offline-first candidate source | median 3.2px re-entry error, 83.3% < 8px | ✅ 已验证 |
| Local-global hybrid w/ CoTracker3 prior | 缩小搜索空间到 ~64px 半径 | ⏳ 执行中 |
| Learned feature matching at re-entry | 当前 DINOv2 template match 远弱于期望 | 🤔 需评估 |

---

## 4. 对 `00_execution_manifest.md` 的建议修改

- 第 20 行 "Full CoTracker3 hybrid" 条目 → 改为 "旧结论作废，真实协议下 CoTracker3 offline 更强，但当前 anchor retrieval 变体仍失败"
- 第 69-74 行 cache 状态 → 更新为全部三个 baseline 已存在
- 第 89-95 行 Track-On2 384x512 限制 → 删除或修正为"已修复并验证可行"
- 第 4 节 "禁止事项" → 第 3 条 "不要做 full CoTracker3 replacement" 保留，但理由改为新结论
