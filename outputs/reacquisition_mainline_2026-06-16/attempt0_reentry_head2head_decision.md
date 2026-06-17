# Attempt0 Re-Entry Head-to-Head Decision

**日期**: 2026-06-16  
**数据来源**: `outputs/attempt0_2026-06-15_recovery/prediction_caches/*.pt`  
**协议**: `tapvid_davis`, `first+input`, 统一 bridge cache, first re-entry frame

---

## 1. 这次验证回答的问题

Phase E 的口头建议是：

> 下一步优先看 `CoTracker3 Hybrid`，因为 CoTracker3 可能在 re-entry 帧的几何精度上优于 Track-On2。

这次验证做的就是把这个判断变成硬对比：

- `Track-On2 DINOv3`
- `CoTracker3 online`
- `CoTracker3 offline`

全部放到 **同一批 DAVIS re-entry queries** 上，直接比较 first re-entry frame 误差。

---

## 2. 结果

### 2.1 全部 re-entry queries（n=259）

| Model | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|
| Track-On2 | **1.60** | **6.91** | 75.7% | **85.7%** | **31.47** |
| CoTracker3 online | 1.73 | 8.75 | 73.0% | 81.5% | 48.00 |
| CoTracker3 offline | 1.63 | 7.62 | **76.1%** | 83.4% | 40.49 |

结论：

- **总体上 Track-On2 没有输给 CoTracker3**
- CoTracker3 offline 很接近，但并没有形成 clear win
- CoTracker3 online 在总体上反而更差

### 2.2 long-occ re-entry（occ >= 20, n=24）

| Model | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|
| Track-On2 | 3.97 | 21.85 | 50.0% | 62.5% | 118.17 |
| CoTracker3 online | **2.99** | 24.47 | **62.5%** | **70.8%** | 121.88 |
| CoTracker3 offline | 3.19 | **21.54** | 54.2% | 66.7% | **90.48** |

结论：

- 在 **小规模** `occ >= 20` 子集上，CoTracker3 确实出现了局部优势信号
- 但样本只有 `24`，还不足以支撑“直接切主线到 CoTracker3 Hybrid”

---

## 3. 对 “CoTracker3 Hybrid” 的判断

### 不成立的强结论

下面这句话当前**不成立**：

> CoTracker3 在 DAVIS re-entry 上显著优于 Track-On2，因此应该直接做通用 CoTracker3 替换式 hybrid。

原因：

1. 全部 re-entry queries 上，Track-On2 总体更强或至少不弱
2. CoTracker3 offline 只有局部接近，没有 clear margin
3. CoTracker3 online 明显不能作为更强替代

### 仍然成立的弱结论

下面这句话当前**成立**：

> 在 `occ >= 20` 的 long-occ re-entry 小子集上，CoTracker3 有值得继续验证的局部优势信号。

这意味着：

- **不要做通用 CoTracker3 Hybrid**
- 如果继续做，只能做 **selective / gated hybrid**
- gate 必须以 `long-occ / hard re-entry` 为前提，而不是全量替换

---

## 4. 当前最合理的下一步

优先级建议：

1. **停止“CoTracker3 全量替换 Track-On2”这个版本**
2. 如果要继续 hybrid，只做：
   - `occ >= 20` 或更 hard re-entry 条件下的 selective comparison
   - 优先比较 `Track-On2` vs `CoTracker3 offline`
3. 如果短期只选一个主 teacher / baseline：
   - **仍然保留 Track-On2 DINOv3 作为主线**

---

## 5. 最终判断

**本次 head-to-head 的结论是：**

> `CoTracker3 Hybrid` 不能作为当前主线的默认下一步，因为在统一 DAVIS re-entry 协议下，CoTracker3 并没有总体性地超过 Track-On2。  
> 唯一值得保留的是：`CoTracker3 offline` 在小规模 long-occ 子集上的局部优势信号，可作为后续 selective hybrid 的探索分支，而不是主线替换。

