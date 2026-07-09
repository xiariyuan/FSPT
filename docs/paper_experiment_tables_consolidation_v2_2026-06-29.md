# Paper Experiment Tables Consolidation v2 — 2026-06-29

## What changed in v2

v2 keeps the original B2-W16-P2 main story and adds two paper-strengthening components:

```text
1. Baseline parity audit for TrackOn2 / TAPNext.
2. Supplementary plug-in generality experiment on reproduced TrackOn2 first-query/input-resolution protocol.
```

The main method is still B2-W16-P2. The TrackOn2 experiment should be used as supplementary evidence, not mixed directly into the main strided-original DAVIS/RGB tables.

## Table 1 — DAVIS main result, strided-original protocol

| method | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ | role |
|---|---:|---:|---:|---:|---|
| CoTracker3 offline | 0.5546 | 70.0510 | 92.1544 | 82.3206 | standard-strong base |
| global B1 vis4_gated288 | 0.6279 | 47.4046 | 72.6762 | 76.4913 | re-entry strong but globally damaging |
| B2 full-post | 0.6278 | 68.9702 | 91.1851 | 82.4604 | localized trigger, W=∞ |
| B2-W16 | 0.6266 | 68.9512 | 91.1924 | 82.4480 | finite-window local override |
| B2-W16-P2 | 0.6251 | 69.0119 | 91.2382 | 82.4425 | main robust variant |
| B2 GT-window oracle | 0.6290 | 70.1821 | 92.6439 | 82.3494 | diagnostic upper bound |

Main conclusion: global re-entry fusion nearly matches B2-W in AJ_RD but collapses standard AJ. B2-W16-P2 retains most re-entry gain while recovering standard tracking.

## Table 2 — RGB fresh20-49 main validation

Protocol: frozen B2-W16-P2, evaluated on `rgb_stacking_000020` ... `rgb_stacking_000049`; excludes dev 0-9 and consumed diagnostic 10-19.

| method | videos | queries | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 30 | 37221 | 0.3816 | 79.5944 | 91.4636 | 88.4854 |
| CoTracker3 online | 30 | 37221 | 0.4121 | 44.6933 | 55.7585 | 72.8684 |
| B2-W16 | 30 | 37221 | 0.4456 | 79.0634 | 92.8891 | 88.4346 |
| B2-W16-P2 | 30 | 37221 | 0.4454 | 79.0664 | 92.8804 | 88.4385 |

RGB gains:

```text
p2_vs_offline_AJ_RD_256 = +0.0638
p2_vs_offline_AJ_256 = -0.5280
p2_vs_online_AJ_RD_256 = +0.0333
p2_vs_online_AJ_256 = +34.3731
p2_vs_w16_AJ_RD_256 = -0.0002
p2_vs_w16_AJ_256 = +0.0030
```

RGB per-video stability:

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

## Table 4 — DAVIS occlusion-length bucket

| bucket | fixed | global B1 | B2 full-post | B2-W16 | B2-W16-P2 | P2-fixed | events |
|---|---:|---:|---:|---:|---:|---:|---:|
| occ_1_4 | 0.6246 | 0.6624 | 0.6619 | 0.6605 | 0.6590 | +0.0344 | 810 |
| occ_5_8 | 0.6261 | 0.6397 | 0.6396 | 0.6375 | 0.6385 | +0.0124 | 340 |
| occ_9_16 | 0.4500 | 0.5529 | 0.5533 | 0.5517 | 0.5509 | +0.1010 | 351 |
| occ_17_32 | 0.4926 | 0.5980 | 0.5980 | 0.5975 | 0.5940 | +0.1014 | 330 |
| occ_33_plus | 0.2914 | 0.4097 | 0.4097 | 0.4103 | 0.4106 | +0.1192 | 32 |

This table supports the central mechanism: gains are concentrated in intended re-entry / long-occlusion regimes.

## Table 5 — DAVIS trigger / false-trigger analysis

| method | triggered tracks | true-trigger tracks | false-trigger tracks | missed re-entry tracks | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| B2-W16 | 2517 | 1334 | 1183 | 51 | 0.5300 | 0.9632 |
| B2-W16-P2 | 2395 | 1314 | 1081 | 71 | 0.5486 | 0.9487 |

B2-W16-P2 false-trigger cost:

```text
false_triggers = 1081
low_cost_rate = 0.511563
harmful_rate = 0.39408
severe_rate = 0.247919
true_trigger_delta_post_mean = 0.025076
```

## Table 6 — Baseline parity audit for TrackOn2 / TAPNext

This table should go to appendix or experimental protocol section. It explains why current TrackOn2/TAPNext strided-original caches are not used as official main-table baselines.

| artifact | protocol | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | status |
|---|---|---:|---:|---:|---:|---|
| TrackOn2 first-input bridge | first-query/input-resolution | 0.5444 | 67.0406 | 93.0615 | 79.8418 | parity-valid, strong but different protocol |
| TrackOn2 existing strided-original cache | strided/original | 0.5383 | 37.5303 | 58.3304 | 42.7415 | not main-table-ready |
| TAPNext existing strided-original cache | strided/original | 0.5199 | 34.5573 | 65.9795 | 43.4950 | parity unresolved / not main-table-ready |

Safe interpretation: TrackOn2 is reproduced and strong under its first-input protocol, but the existing strided-original bridge is not parity-established. Do not claim B2-W generally beats TrackOn2/TAPNext from weak bridge caches.

## Table 7 — Supplementary plug-in generality on reproduced TrackOn2 protocol

Protocol: DAVIS first-query/input-resolution bridge, 650 queries. This is a supplemental experiment, not directly mixed with main strided-original results.

| method | base | override | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | delta_avg_256 ↑ | trigger precision | trigger recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline first-input | - | - | 0.4525 | 62.1660 | 89.0107 | 77.2244 | - | - |
| TrackOn2 first-input | - | - | 0.5444 | 67.0406 | 93.0615 | 79.8418 | - | - |
| B2-W16 | TrackOn2 | CoTracker3 offline | 0.5513 | 67.1750 | 93.2513 | 79.8543 | 0.7700 | 0.6332 |
| B2-W16-P2 | TrackOn2 | CoTracker3 offline | 0.5509 | 67.1260 | 93.2224 | 79.8202 | 0.7892 | 0.6216 |
| B2-W16-P2 | CoTracker3 offline | TrackOn2 | 0.5495 | 64.4295 | 92.7140 | 77.8145 | 0.6132 | 0.8263 |

Key supplementary conclusion: using TrackOn2 as the strong base, B2-W16-P2 improves TrackOn2 by `+0.0065 AJ_RD_256` and `+0.0854 AJ_256`. This is a small but positive plug-in generality result under a parity-valid TrackOn2 protocol.

Per-video stability for TrackOn2-base B2-W16-P2:

```text
n_videos = 30
p2_trackon_base_improves_AJRD_vs_trackon = 12
p2_trackon_base_improves_AJ_vs_trackon = 13
p2_trackon_base_improves_both_vs_trackon = 9
p2_trackon_base_AJRD_drop_gt_0p01 = 6
p2_trackon_base_AJ_drop_gt_1 = 3
p2_cotracker_base_improves_AJRD_vs_cotracker = 20
p2_cotracker_base_improves_AJ_vs_cotracker = 20
```

## Updated claim boundaries after v2

Supported:

```text
1. B2-W16-P2 improves re-entry reliability while preserving most standard tracking under the main DAVIS/RGB strided-original protocol.
2. Gains concentrate in long-occlusion / re-entry regimes.
3. P2 is a conservative false-trigger-cost reducer.
4. Baseline parity for TrackOn2 first-input protocol is validated.
5. B2-W gives a small positive plug-in gain on reproduced TrackOn2 first-input protocol.
```

Still unsupported:

```text
1. B2-W beats TrackOn2/TAPNext generally.
2. B2-W is a new universal SOTA tracker.
3. Current TrackOn2/TAPNext strided-original bridge results are official comparable baselines.
4. B2-RV is ready to replace B2-W16-P2.
```

## Recommended paper placement

```text
Main paper: Tables 1-5 + qualitative figures.
Appendix/Supplement: Table 6 baseline parity audit + Table 7 TrackOn2 plug-in generality.
Method limitation: current B2-W requires a compatible base/override pair and reliable visibility signals.
```
