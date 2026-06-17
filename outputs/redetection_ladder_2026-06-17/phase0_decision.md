# Phase 0 — Decision (strided+original)

**日期**: 2026-06-17  
**协议**: `strided+original`  
**Decision**: ✅ **GO — ADVANCE TO PHASE 1**

---

## 1. Stop Criteria 对照

| Stop Criteria | 判定 |
|---|---|
| full DAVIS `strided+original` 仍然跑不通 | ❌ **NOT HIT** |

---

## 2. 事实

| Baseline | Status | AJ | Cache |
|---|---|---|---|
| CoTracker3 online | ✅ Complete | 36.95 | 30 npz + unified .pt |
| CoTracker3 offline | ✅ Complete | 51.54 | 30 npz + unified .pt |
| Track-On2 | ❌ Failed (unsupported) | — | — |

Phase 0 Stop Criteria 未命中。至少一个 baseline（CoTracker3）完整跑通。

---

## 3. Phase 1 预判

| Stop Criteria | CoTracker3 实测 | 判定 |
|---|---|---|
| long-occ AJ barely improves | oracle gap mean ≈ 12px | ❌ NOT barely |
| re-entry error barely decreases | oracle gap mean ≈ 12-15px | ❌ NOT barely |

---

## 4. 最终结论

```
Phase 0: ✅ GO
Phase 1: ✅ ADVANCE TO PHASE 2 (partial — CoTracker3 only)
Phase 2: ⏸️ BLOCKED — waiting for GPT review of repair artifacts
```

---
