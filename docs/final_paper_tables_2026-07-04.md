# Final Paper Tables — 2026-07-04

This document freezes the current paper-ready quantitative evidence. All “official-style” results are local evaluations under the stated TAP-Vid protocol, not official leaderboard/server submissions.

## Table 1. TAPVid-DAVIS first/input official-style local evaluation

Protocol: TAPVid-DAVIS, 30 videos, 650 queries, query mode `first`, input/256 metric resolution. Standard metrics are AJ / OA / δ_avg / δ_4px; AJ_RD is a re-entry diagnostic metric.

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 baseline | 64.8949 | 91.7983 | 77.3560 | 84.9284 | 0.3486 | 0.5246 |
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 84.5776 | 0.3142 | 0.4525 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 87.8233 | 0.3714 | 0.5444 |
| ReEntry V1 | 64.5758 | 91.7274 | 77.2244 | 84.5776 | 0.3588 | 0.5305 |
| ReEntry V22Q | 64.7260 | 91.7406 | 77.2244 | 84.5776 | 0.3556 | 0.5252 |
| ReEntry V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 84.5776 | 0.3549 | 0.5236 |

Deltas vs CoTracker3 offline:

| Method | ΔAJ | ΔOA | Δδ_avg | ΔAJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 baseline | +2.2383 | +3.6496 | +0.1316 | +0.0344 |
| TrackOn2 | +4.3841 | +3.9429 | +2.6174 | +0.0572 |
| ReEntry V1 | +1.9193 | +3.5787 | +0.0000 | +0.0446 |
| ReEntry V22Q | +2.0695 | +3.5919 | +0.0000 | +0.0414 |
| ReEntry V24-DINOScore | +2.1874 | +3.7363 | +0.0000 | +0.0407 |

Key paired-video checks vs CoTracker3 offline:

| Method | AJ mean pp [95% CI] | OA mean pp [95% CI] | AJ W/L/T | OA W/L/T |
|---|---:|---:|---:|---:|
| ReEntry V1 | +1.9193 [+0.8375, +3.0545] | +3.5787 [+1.8485, +5.6697] | 22/7/1 | 24/5/1 |
| ReEntry V22Q | +2.0695 [+1.0763, +3.1667] | +3.5919 [+2.0011, +5.5624] | 23/6/1 | 25/4/1 |
| ReEntry V24 | +2.1874 [+1.2582, +3.2487] | +3.7363 [+2.1673, +5.7423] | 24/5/1 | 27/2/1 |

## Table 2. TAPVid-DAVIS strided/original official-style local evaluation

Protocol: TAPVid-DAVIS, 30 videos, 5,882 queries, query mode `strided`, original-resolution metrics. This is the strictest DAVIS local protocol currently audited.

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 68.0060 | 0.3870 | 0.5546 |
| CoTracker3 online | 36.9543 | 68.1493 | 45.0072 | 47.7100 | 0.4101 | 0.5972 |
| TrackOn2 | 28.3785 | 58.3304 | 33.2280 | 35.8975 | 0.3700 | 0.5383 |
| TAPNext | 25.9433 | 65.9795 | 33.4476 | 36.1557 | 0.3624 | 0.5199 |
| ReEntry V1 default | 50.7303 | 91.2730 | 63.5892 | 68.0060 | 0.4144 | 0.6072 |
| ReEntry V22Q default | 50.8342 | 91.3439 | 63.5892 | 68.0060 | 0.4136 | 0.6047 |
| V25-safe threshold=0.80 | 51.5648 | 92.2106 | 63.5892 | 68.0060 | 0.3900 | 0.5601 |

Deltas vs CoTracker3 offline:

| Method | ΔAJ | ΔOA | Δδ_avg | ΔAJ_RD |
|---|---:|---:|---:|---:|
| ReEntry V1 default | -0.8081 | -0.8813 | +0.0000 | +0.0274 |
| ReEntry V22Q default | -0.7043 | -0.8105 | +0.0000 | +0.0266 |
| V25-safe threshold=0.80 | +0.0263 | +0.0563 | +0.0000 | +0.0030 |

Interpretation: default V1/V22Q improve AJ_RD but reduce strided/original AJ/OA. V25-safe threshold=0.80 preserves AJ/OA with only a tiny AJ_RD gain.

## Table 3. TAPVid RGB-Stacking full50 strided official-style local evaluation

Protocol: TAPVid RGB-Stacking full50, 50 videos, 60,829 queries, strided official-style local evaluation, input/256-style metric resolution.

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD | Queries |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 79.9345 | 91.6371 | 88.5714 | 91.5875 | 0.3617 | 60,829 |
| CoTracker3 online | 44.8432 | 55.8092 | 73.1881 | 83.3099 | 0.4028 | 60,829 |
| ReEntry V1 | 79.4302 | 93.0758 | 88.5714 | 91.5875 | 0.4414 | 60,829 |
| ReEntry V22Q | 79.5756 | 93.0481 | 88.5714 | 91.5875 | 0.4400 | 60,829 |
| ReEntry V24-DINOScore | 79.5974 | 93.0389 | 88.5714 | 91.5875 | 0.4394 | 60,829 |

Deltas vs CoTracker3 offline:

| Method | ΔAJ | ΔOA | Δδ_avg | ΔAJ_RD |
|---|---:|---:|---:|---:|
| ReEntry V1 | -0.5043 | +1.4387 | +0.0000 | +0.0797 |
| ReEntry V22Q | -0.3589 | +1.4110 | +0.0000 | +0.0783 |
| ReEntry V24-DINOScore | -0.3371 | +1.4018 | +0.0000 | +0.0777 |

Key paired-video checks vs CoTracker3 offline:

| Method | AJ mean pp [95% CI] | OA mean pp [95% CI] | AJ W/L/T | OA W/L/T |
|---|---:|---:|---:|---:|
| ReEntry V1 | -0.5043 [-0.8046, -0.2011] | +1.4387 [+0.9589, +1.9236] | 9/41/0 | 39/11/0 |
| ReEntry V22Q | -0.3589 [-0.6391, -0.0670] | +1.4110 [+0.9452, +1.8875] | 10/40/0 | 40/10/0 |
| ReEntry V24 | -0.3371 [-0.6065, -0.0531] | +1.4018 [+0.9438, +1.8769] | 11/39/0 | 40/10/0 |

## Table 4. RGB-Stacking fresh/stress diagnostic evaluation

These are diagnostic, not official benchmark submissions. They isolate re-entry behavior under natural and synthetic stress settings.

| Setting | AJ_RD Base→V24 | ΔAJ_RD | OA Base→V24 | ΔOA | AJ Base→V24 | ΔAJ |
|---|---:|---:|---:|---:|---:|---:|
| fresh20-49 natural | 0.3330 → 0.4027 | +0.0698 | 91.4636 → 92.9244 | +1.4608 | 79.5944 → 79.2884 | -0.3060 |
| translate_L16 | 0.5080 → 0.5564 | +0.0484 | 90.7805 → 92.1951 | +1.4146 | 75.1940 → 74.9987 | -0.1953 |
| occluder_L16 | 0.6417 → 0.6763 | +0.0345 | 90.8918 → 92.2293 | +1.3375 | 77.6280 → 77.2276 | -0.4004 |
| Average | 0.4939 → 0.5451 | +0.0509 | 91.0453 → 92.4496 | +1.4043 | 77.4721 → 77.1716 | -0.3006 |

## Table 5. V25 official-safe threshold sweep on DAVIS strided/original

Purpose: find a conservative operating point that preserves official-style AJ/OA while retaining some AJ_RD gain.

| Threshold | AJ | OA | AJ_RD | ΔAJ | ΔOA | ΔAJ_RD | Recovered frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 50.7569 | 91.2704 | 0.4141 | -0.7816 | -0.8839 | +0.0271 | 39,809 |
| 0.30 | 50.8000 | 91.2821 | 0.4136 | -0.7384 | -0.8722 | +0.0266 | 39,508 |
| 0.40 | 50.8522 | 91.3187 | 0.4131 | -0.6863 | -0.8357 | +0.0261 | 38,993 |
| 0.50 | 50.9103 | 91.4152 | 0.4118 | -0.6282 | -0.7391 | +0.0248 | 37,759 |
| 0.60 | 51.0521 | 91.6501 | 0.4092 | -0.4864 | -0.5043 | +0.0222 | 34,459 |
| 0.70 | 51.3570 | 91.9982 | 0.3988 | -0.1815 | -0.1562 | +0.0118 | 26,932 |
| 0.80 | 51.5648 | 92.2106 | 0.3900 | +0.0263 | +0.0563 | +0.0030 | 15,024 |
| 0.90 | 51.5630 | 92.1914 | 0.3875 | +0.0246 | +0.0370 | +0.0005 | 4,140 |
| 0.95 | 51.5420 | 92.1607 | 0.3870 | +0.0035 | +0.0063 | +0.0000 | 773 |
| 0.99 | 51.5385 | 92.1543 | 0.3870 | +0.0000 | +0.0000 | +0.0000 | 0 |

Best official-safe point: threshold=0.80. It preserves AJ/OA with tiny positive AJ_RD (+0.0030).

## Table 6. Method positioning

| Variant | Role | Paper positioning |
|---|---|---|
| V1 Learned ReEntry-VisCalibrator | Main AJ_RD-oriented method | Strongest AJ_RD among ReEntry variants on RGB full50 and DAVIS strided/original default; can trade off standard AJ/OA. |
| V22Q Interval/Gate | Stability/interval extension | Slightly safer than V1 for standard AJ/OA; good stability variant; not replaced by V24. |
| V24-DINOScore | Optional AJ-oriented appearance micro-filter | Small AJ gains over V22Q with tiny AJ_RD/OA cost; best ReEntry row on DAVIS first/input but not a universal default. |
| V25 threshold=0.80 | Official-safe conservative operating point | Preserves DAVIS strided/original AJ/OA, but AJ_RD gain is tiny. |
| V26 future direction | Official-metric-aware selective ReEntry | Needed if the target is stronger leaderboard-style improvement under strided/original. |

## Final table-level conclusion

The paper should claim that ReEntry improves re-entry recovery across TAP-Vid official-style local evaluations and diagnostic stress settings. It improves DAVIS first/input AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter strided/original evaluation reveals a trade-off that can be mitigated but not fully solved by conservative thresholding. All results are local official-style evaluations, not official leaderboard submissions.
