# Paper Experiment Tables Consolidation — 2026-06-29

## Positioning

The final paper should frame B2-W16-P2 as a **localized re-entry reliability intervention**, not as a new universal SOTA point tracker. The core claim is: global or online re-entry-strong branches improve re-entry but damage standard tracking; B2-W localizes their usage to predicted re-entry windows, improving AJ_RD while preserving most standard AJ.

Recommended main method name:

```text
B2-W16-P2: Predicted Re-entry Windowed Override with 2-frame override persistence
```

## Table 1 — DAVIS main result

| method | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ | role |
|---|---:|---:|---:|---:|---|
| CoTracker3 offline | 0.5546 | 70.0510 | 92.1544 | 82.3206 | standard-strong base |
| global B1 vis4_gated288 | 0.6279 | 47.4046 | 72.6762 | 76.4913 | re-entry strong but globally damaging |
| B2 full-post | 0.6278 | 68.9702 | 91.1851 | 82.4604 | localized trigger, W=∞ |
| B2-W16 | 0.6266 | 68.9512 | 91.1924 | 82.4480 | finite-window local override |
| B2-W16-P2 | 0.6251 | 69.0119 | 91.2382 | 82.4425 | finite-window + 2-frame persistence guard |
| B2 GT-window oracle | 0.6290 | 70.1821 | 92.6439 | 82.3494 | diagnostic upper bound |

**Use this as the first main table.** It shows the central tradeoff: global B1 nearly matches B2 on AJ_RD but collapses standard AJ; B2-W16-P2 preserves most of the re-entry gain while restoring standard tracking.

## Table 2 — RGB fresh20-49 main validation

Protocol: frozen B2-W16-P2, evaluated on `rgb_stacking_000020` ... `rgb_stacking_000049`; excludes dev 0-9 and consumed diagnostic 10-19.

| method | videos | queries | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 30 | 37221 | 0.3816 | 79.5944 | 91.4636 | 88.4854 |
| CoTracker3 online | 30 | 37221 | 0.4121 | 44.6933 | 55.7585 | 72.8684 |
| B2-W16 | 30 | 37221 | 0.4456 | 79.0634 | 92.8891 | 88.4346 |
| B2-W16-P2 | 30 | 37221 | 0.4454 | 79.0664 | 92.8804 | 88.4385 |

Main RGB gains:

```text
p2_vs_offline_AJ_RD_256 = +0.0638
p2_vs_offline_AJ_256 = -0.5280
p2_vs_online_AJ_RD_256 = +0.0333
p2_vs_online_AJ_256 = +34.3731
p2_vs_w16_AJ_RD_256 = -0.0002
p2_vs_w16_AJ_256 = +0.0030
```

Per-video stability on RGB fresh20-49:

```text
n_videos = 30
p2_improves_AJRD_vs_offline = 24
p2_improves_AJRD_vs_offline_ge_0p01 = 23
p2_AJ_drop_vs_offline_le_1 = 20
p2_AJ_drop_vs_offline_gt_2 = 3
p2_beats_online_AJRD = 23
p2_beats_online_AJ_by_10 = 30
p2_improves_AJRD_vs_w16 = 12
p2_improves_AJ_vs_w16 = 17
```

## Table 3 — DAVIS window / persistence ablation

| variant | AJ_RD_256 ↑ | AJ_256 ↑ | ΔAJ_RD vs full-post | ΔAJ vs full-post | interpretation |
|---|---:|---:|---:|---:|---|
| B2 full-post / W=∞ | 0.6278 | 68.9702 | +0.0000 | +0.0000 | W=∞ reference |
| B2-W16 | 0.6266 | 68.9512 | -0.0012 | -0.0190 | finite window |
| B2-W32 | 0.6265 | 68.9596 | -0.0013 | -0.0106 | finite window |
| B2-W64 | 0.6277 | 68.9706 | -0.0001 | +0.0004 | upper finite-window limit |
| B2-W16-P2 | 0.6251 | 69.0119 | -0.0027 | +0.0417 | main robust variant |

Takeaway: finite-window override is not a RGB-specific patch. On DAVIS, W16/W64 preserve almost all full-post AJ_RD, and P2 trades a small AJ_RD reduction for cleaner standard AJ and trigger precision.

## Table 4 — DAVIS occlusion-length bucket

| bucket | fixed | global B1 | B2 full-post | B2-W16 | B2-W16-P2 | P2-fixed | events |
|---|---:|---:|---:|---:|---:|---:|---:|
| occ_1_4 | 0.6246 | 0.6624 | 0.6619 | 0.6605 | 0.6590 | +0.0344 | 810 |
| occ_5_8 | 0.6261 | 0.6397 | 0.6396 | 0.6375 | 0.6385 | +0.0124 | 340 |
| occ_9_16 | 0.4500 | 0.5529 | 0.5533 | 0.5517 | 0.5509 | +0.1010 | 351 |
| occ_17_32 | 0.4926 | 0.5980 | 0.5980 | 0.5975 | 0.5940 | +0.1014 | 330 |
| occ_33_plus | 0.2914 | 0.4097 | 0.4097 | 0.4103 | 0.4106 | +0.1192 | 32 |

This table is important because it proves that B2-W16-P2 gains are concentrated in the intended long-occlusion / re-entry regime.

## Table 5 — Trigger / false-trigger analysis

### DAVIS trigger comparison

| method | triggered tracks | true-trigger tracks | false-trigger tracks | missed re-entry tracks | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| B2-W16 | 2517 | 1334 | 1183 | 51 | 0.5300 | 0.9632 |
| B2-W16-P2 | 2395 | 1314 | 1081 | 71 | 0.5486 | 0.9487 |

### RGB fresh20-49 trigger comparison

| method | trigger events | triggered tracks | true-trigger tracks | false-trigger tracks | missed re-entry tracks | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB fresh20-49 B2-W16 | 32957 | 10415 | 5780 | 4635 | 681 | 0.5550 | 0.8946 |
| RGB fresh20-49 B2-W16-P2 | 31218 | 10147 | 5674 | 4473 | 787 | 0.5592 | 0.8782 |

DAVIS false-trigger cost for B2-W16-P2:

```text
false_triggers = 1081
low_cost_rate = 0.511563
harmful_rate = 0.39408
severe_rate = 0.247919
false_delta_full_mean = -0.026044
false_delta_full_median = -0.006472
true_trigger_delta_post_mean = 0.025076
```

## Recommended paper figure plan

| figure | content | purpose |
|---|---|---|
| Figure 1 | Method diagram: base tracker, override branch, predicted re-entry trigger, W-frame local override, return to base | Explain B2-W clearly |
| Figure 2 | DAVIS qualitative success: fixed misses re-entry, global/B1 recovers but damages standard, B2-W recovers locally | Show mechanism |
| Figure 3 | RGB fresh success: online has re-entry but bad standard AJ, B2-W16-P2 preserves standard | Cross-dataset qualitative |
| Figure 4 | Failure case: false trigger or weak override branch causes standard AJ loss | Honest limitation |

## Main claims that are supported

Supported:

```text
1. Global re-entry fusion improves AJ_RD but damages standard tracking.
2. Localized/windowed override recovers most re-entry benefit while preserving standard AJ.
3. B2-W16-P2 generalizes across DAVIS and RGB fresh20-49.
4. Gains are concentrated in re-entry / long-occlusion regimes.
5. P2 is a conservative false-trigger-cost reducer, not a large new performance module.
```

Not supported / should not claim:

```text
1. New SOTA point tracker.
2. Universal standard-AJ improvement.
3. Complete RGB generalization of the DAVIS 4-teacher B1 branch.
4. Solving all long-term point tracking failures.
```

## Next required writing step

Create a paper skeleton with these sections: Abstract, Introduction, Related Work, Method, Experiments, Ablation, Failure Analysis, Limitations. Use Table 1 and Table 2 as the main empirical backbone.

