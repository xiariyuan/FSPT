# Partial Repo-Native Decision — Attempt 0 Recovery (2026-06-15)

**Type**: Repo-native partial decision with first/input unified bridge (NOT a completed Attempt 0 final ranking)

## Three-Layer Summary

### Layer 1: Confirmed Working (repo-native reproduction complete)

| Baseline | DAVIS AJ | DAVIS δ_avg | DAVIS OA | vs Track-On2 |
|---|---|---|---|---|
| **trackon2_dinov3** | **67.04** | **79.84** | **92.09** | — |
| cotracker3_baseline (online variant) | 64.89 | 77.36 | 91.80 | AJ −2.15, δ −2.48 |
| cotracker3_offline (strong 2D anchor) | 62.66 | 77.22 | 88.15 | AJ −4.38, δ −2.62 |

All three baselines reproduce official reference δ_avg within noise.

### Layer 2: Blocked by Missing Assets

| Baseline | Missing | Source |
|---|---|---|
| Track-On-R | `track_on_r.pt`, `verifier.pt` | HF: `gorkaydemir/track_on_r` (network unreachable) |
| TAPNext++ | No checkpoint found in repo | — |
| AllTracker | No checkpoint found in repo | — |

### Layer 3: Unified Bridge Status (Round 4)

**Bridge protocol**: first-query + input-space (256×256)

**Current status**:
- Track-On2: ✅ **bridge complete** — 0.00 diff vs repo-native
- CoTracker3 baseline: ✅ **bridge complete** — 0.00 diff vs repo-native
- CoTracker3 offline: ✅ **bridge complete** — 0.00 diff vs repo-native

| Baseline | Unified Bridge AJ | Unified Bridge δ_avg | Unified Bridge OA | vs Repo-Native Diff | Bridge Status |
|---|---|---|---|---|---|
| **trackon2_dinov3** | **67.04** | **79.84** | **92.09** | **0.00** (perfect match) | ✅ complete |
| cotracker3_baseline | 64.89 | 77.36 | 91.80 | 0.00 | ✅ complete |
| cotracker3_offline | 62.66 | 77.22 | 88.15 | 0.00 | ✅ complete |

**Resolved CoTracker3 bridge issue**: The earlier 1.75–2.47pt AJ gap was caused by a bug in [`datasets/metrics.py`](/gemini/code/FSPT/datasets/metrics.py). The unified metric wrapper used a `max_abs <= 1.5` heuristic to decide whether tracks were normalized before converting them back to TAP-Vid raster space. That heuristic fails for CoTracker3 because normalized trajectories can legitimately extrapolate beyond frame bounds, producing values greater than `1.0` or less than `0.0` while still remaining normalized. As a result, the wrapper skipped the required `* (size - 1)` scaling and evaluated CoTracker3 in the wrong coordinate space. After fixing that logic, both CoTracker3 bridges match repo-native exactly.

**Long-occ sub-metrics** (first+input bridge, min_run=20):

| Baseline | Long-occ AJ | Long-occ δ_avg |
|---|---|---|
| trackon2_dinov3 | 45.27 | 68.11 |
| cotracker3_baseline | 45.73 | 64.71 |
| cotracker3_offline | 39.30 | 63.98 |

**Status**: `first_input_bridge_complete_for_all_runnable_baselines`

True `strided + original` Attempt 0 main protocol is still **pending**.

## Reproduction Parity Check

| Baseline | Our δ_avg | Official/Reference | Δ | Status |
|---|---|---|---|---|
| trackon2_dinov3 | 79.84 | 79.9 | −0.06 | ✅ match |
| cotracker3_baseline | 77.36 | ~77.4 | −0.04 | ✅ match |
| cotracker3_offline | 77.22 | ~77.2 | +0.02 | ✅ match |

## Variant Mapping (CoTracker3)

This section documents the actual variant→artifact mapping to prevent confusion:

| Attempt 0 row | Predictor variant | Checkpoint | Eval metrics |
|---|---|---|---|
| `cotracker3_baseline` | `CoTrackerPredictor(offline=False, window_len=16)` | `scaled_online.pth` | AJ=64.89, δ=77.36, OA=91.80 |
| `cotracker3_offline` | `CoTrackerPredictor(offline=True, window_len=60)` | `scaled_offline.pth` | AJ=62.66, δ=77.22, OA=88.15 |

**Note**: `cotracker3_baseline` maps to the online variant. The offline variant is the separate strong 2D anchor.

## Partial Conclusion

Among the currently runnable repo-native baselines, `trackon2_dinov3` achieves the highest DAVIS δ_avg (79.84) and is the strongest candidate. This is a **repo-native partial decision with first/input unified bridge** — it does not constitute a completed Attempt 0 final ranking.

To advance this to a full Attempt 0 decision, the following must be resolved:
1. Track-On-R checkpoint procurement (or confirmed skip)
2. TAPNext++ / AllTracker checkpoint procurement (or confirmed skip)
3. True `strided + original` Attempt 0 main protocol (requires re-running predictors with strided queries)
4. Long-occ sub-metric computation under the true `strided + original` main protocol

## Artifacts Produced

| Artifact | Status |
|---|---|
| `outputs/track_on_env_manifest.md` | ✅ Environment documented |
| `outputs/dinov3_load_smoke.json` | ✅ DINOv3 load verified |
| `outputs/fallback_decision.md` | ✅ No fallback triggered |
| `outputs/trackon_local_backbone_smoke.txt` | ✅ Backbone smoke passed |
| `outputs/trackon_adapter_sanity_report.json` | ✅ Adapter verified |
| `outputs/track_on_checkpoint_manifest.md` | ✅ Checkpoints documented |
| `outputs/track_on2_repo_native_metrics.json` | ✅ DAVIS metrics |
| `outputs/track_on2_kinetics_metrics.json` | ✅ Kinetics 10-clip smoke |
| `outputs/cotracker3_offline_repo_native_metrics.json` | ✅ Offline eval |
| `outputs/cotracker3_online_repo_native_metrics.json` | ✅ Online eval |
| `outputs/attempt0_2026-06-15_recovery/prediction_caches/*_first_input_bridge.pt` | ✅ Round 4: corrected unified bridge caches |
| `outputs/attempt0_2026-06-15_recovery/unified_rescoring/*_first_input_validation.json` | ✅ Round 4: bridge validation passed |
| `outputs/attempt0_2026-06-15_recovery/unified_rescoring/*_first_input_parity.json` | ✅ Round 5: official parity passed after metric-wrapper fix |
| `outputs/attempt0_2026-06-15_recovery/unified_rescoring/*_first_input_rescore.json` | ✅ Round 5: unified rescoring matches repo-native for all runnable baselines |
| `outputs/attempt0_2026-06-15_recovery/reports/unified_protocol_bridge_round4.md` | ✅ Round 4: protocol bridge doc |
| `outputs/attempt0_2026-06-15_recovery/reports/*_first_input_repo_vs_unified.json` | ✅ Round 5: all diff reports now show 0.00 bridge gap |
| `outputs/attempt0_2026-06-15_recovery/status/*.json` | ✅ Round 5: updated with resolved CoTracker3 bridge results |
| `outputs/attempt0_2026-06-15_recovery/final_decision.md` | ✅ This file (updated) |
| `scripts/attempt0_export_tapvid_repo_cache.py` | ✅ Round 4: new exporter script |
| `scripts/attempt0_validate_cache.py` | ✅ Round 4: updated with --norm-tol, --query-anchor-tol-px |
| `scripts/attempt0_rescore_cache.py` | ✅ Round 4: updated with --norm-tol |
| `scripts/attempt0_metric_parity.py` | ✅ Round 5: updated with metric-resolution selection |
| `datasets/metrics.py` | ✅ Round 5: fixed normalized-coordinate rescaling for out-of-bounds CoTracker3 tracks |
