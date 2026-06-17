# Round 3 Correction Notes (2026-06-15)

## What was corrected

### 1. Missing artifact: `outputs/fallback_decision.md`
- **Created**: `outputs/fallback_decision.md`
- Documents that DINOv3 route passed, no fallback needed

### 2. CoTracker3 naming/mapping fix
- **Before**: `cotracker3_baseline.json` had offline metrics (62.66) with offline checkpoint, and `cotracker3_offline.json` had online metrics (64.89) with offline checkpoint path — both were mislabeled
- **After**: 
  - `cotracker3_baseline.json` → online variant (64.89, `scaled_online.pth`)
  - `cotracker3_offline.json` → offline variant (62.66, `scaled_offline.pth`)
- **Root cause**: The eval script ran `cotracker3_offline` first and `cotracker3_online` second, but the status names were assigned by Attempt 0 row order, not eval execution order

### 3. Status JSON completions
- `trackon2.json`: `rescoring_status` changed from `complete_via_evaluator_native` → `pending`, with clear note about why unified rescoring is incompatible
- `cotracker3_baseline.json`: Added `checkpoint_sha256`, `variant`, `variant_note`, `official_reference_numbers`, `delta_vs_official`, `keep_as_teacher_candidate`, `notes`
- `cotracker3_offline.json`: Same fields added
- `tapnextpp.json`, `alltracker.json`: Changed from skeleton `pending` → `blocked` with specific missing-asset info

### 4. Final decision downgrade
- **Before**: Titled "Final Decision — Attempt 0" with language suggesting Attempt 0 completion
- **After**: Titled "Partial Repo-Native Decision — Attempt 0 Recovery" with three-layer structure (Confirmed Working / Blocked / Not Yet Resolved)
- Explicitly states unified rescoring is NOT completed
- Explicitly states this is a partial decision, not a final ranking

## Conclusions downgraded

| Statement | Before | After |
|---|---|---|
| Attempt 0 status | Implied complete | Explicitly partial |
| `rescoring_status` (trackon2) | `complete_via_evaluator_native` | `pending` |
| Unified rescoring | Not mentioned as incomplete | Three distinct unresolved items listed |
| Decision type | "Proceed with trackon2_dinov3" | "current repo-native best, pending unified rescoring resolution" |
