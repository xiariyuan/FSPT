# ReEntry-TAP Paper Tables — 2026-07-01

Generated from JSON summaries by `scripts/make_reentry_tap_paper_tables_and_figures.py`.

## Table 1. Natural RGB fresh20-49 validation

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | #videos | #queries |
| --- | --- | --- | --- | --- | --- | --- |
| CoTracker3 offline | 0.3816 | 79.5944 | 91.4636 | 88.4854 | 30 | 37221 |
| CoTracker3 online | 0.4121 | 44.6933 | 55.7585 | 72.8684 | 30 | 37221 |
| B2-W16-P2 | 0.4454 | 79.0664 | 92.8804 | 88.4385 | 30 | 37221 |

B2-W16-P2 vs offline: AJ_RD 0.0638, AJ -0.5280.  
B2-W16-P2 vs online: AJ_RD 0.0333, AJ 34.3731.

## Table 2. Translate severity curve on RGB dev0-9

| Family | L | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
| --- | --- | --- | --- | --- | --- | --- |
| translate | 8 | cotracker3_offline | 0.4709 | 77.0301 | 90.9260 | 86.9612 |
| translate | 8 | cotracker3_online | 0.5188 | 44.0925 | 56.6368 | 68.6245 |
| translate | 8 | b2_w16_p2 | 0.5513 | 76.9534 | 92.7714 | 87.1168 |
| translate | 16 | cotracker3_offline | 0.4708 | 75.8656 | 90.7947 | 86.1429 |
| translate | 16 | cotracker3_online | 0.5194 | 43.9771 | 57.0539 | 67.2819 |
| translate | 16 | b2_w16_p2 | 0.5485 | 75.8344 | 92.6729 | 86.3117 |
| translate | 32 | cotracker3_offline | 0.4614 | 74.0875 | 90.0318 | 84.9765 |
| translate | 32 | cotracker3_online | 0.5139 | 43.5656 | 57.7568 | 64.8982 |
| translate | 32 | b2_w16_p2 | 0.5452 | 74.1657 | 92.0251 | 85.1856 |

## Table 3. Moving-occluder severity curve on RGB dev0-9

| Family | L | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
| --- | --- | --- | --- | --- | --- | --- |
| occluder | 8 | cotracker3_offline | 0.6417 | 78.5238 | 90.9933 | 88.1378 |
| occluder | 8 | cotracker3_online | 0.6547 | 44.1678 | 57.0715 | 72.4975 |
| occluder | 8 | b2_w16_p2 | 0.6875 | 78.0956 | 92.7550 | 88.0913 |
| occluder | 16 | cotracker3_offline | 0.6422 | 77.6880 | 90.6437 | 87.6125 |
| occluder | 16 | cotracker3_online | 0.6477 | 43.6800 | 58.2391 | 72.0218 |
| occluder | 16 | b2_w16_p2 | 0.6873 | 77.1682 | 92.2855 | 87.5420 |
| occluder | 32 | cotracker3_offline | 0.6388 | 76.2831 | 90.1738 | 86.8624 |
| occluder | 32 | cotracker3_online | 0.6173 | 42.3135 | 60.5086 | 70.9439 |
| occluder | 32 | b2_w16_p2 | 0.6825 | 75.5675 | 91.6044 | 86.7668 |

## Table 4. Stress gains on RGB dev0-9

| Family | L | Comparison | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
| --- | --- | --- | --- | --- | --- |
| translate | 8 | online_vs_offline | 0.0479 | -32.9376 | -34.2892 |
| translate | 8 | b2_vs_offline | 0.0804 | -0.0767 | 1.8454 |
| translate | 8 | b2_vs_online | 0.0325 | 32.8609 | 36.1346 |
| translate | 16 | online_vs_offline | 0.0486 | -31.8885 | -33.7408 |
| translate | 16 | b2_vs_offline | 0.0777 | -0.0312 | 1.8782 |
| translate | 16 | b2_vs_online | 0.0291 | 31.8573 | 35.6190 |
| translate | 32 | online_vs_offline | 0.0525 | -30.5219 | -32.2750 |
| translate | 32 | b2_vs_offline | 0.0838 | 0.0782 | 1.9933 |
| translate | 32 | b2_vs_online | 0.0313 | 30.6001 | 34.2683 |
| occluder | 8 | online_vs_offline | 0.0130 | -34.3560 | -33.9218 |
| occluder | 8 | b2_vs_offline | 0.0458 | -0.4282 | 1.7617 |
| occluder | 8 | b2_vs_online | 0.0328 | 33.9278 | 35.6835 |
| occluder | 16 | online_vs_offline | 0.0055 | -34.0080 | -32.4046 |
| occluder | 16 | b2_vs_offline | 0.0451 | -0.5198 | 1.6418 |
| occluder | 16 | b2_vs_online | 0.0396 | 33.4882 | 34.0464 |
| occluder | 32 | online_vs_offline | -0.0215 | -33.9696 | -29.6652 |
| occluder | 32 | b2_vs_offline | 0.0437 | -0.7156 | 1.4306 |
| occluder | 32 | b2_vs_online | 0.0652 | 33.2540 | 31.0958 |

## Table 5. Stress-induced-only AJ_RD_256 on RGB dev0-9

| Stress | stress-induced event rate | offline | online | B2-W16-P2 | B2-offline | B2-online |
| --- | --- | --- | --- | --- | --- | --- |
| translate L8 | 0.1762 | 0.6980 | 0.7345 | 0.7408 | 0.0428 | 0.0063 |
| translate L16 | 0.1791 | 0.6763 | 0.7204 | 0.7242 | 0.0479 | 0.0038 |
| translate L32 | 0.1888 | 0.6340 | 0.6771 | 0.6912 | 0.0572 | 0.0141 |
| occluder L8 | 0.5532 | 0.7587 | 0.7538 | 0.7828 | 0.0241 | 0.0290 |
| occluder L16 | 0.5986 | 0.7520 | 0.7410 | 0.7772 | 0.0252 | 0.0362 |
| occluder L32 | 0.6190 | 0.7420 | 0.6954 | 0.7705 | 0.0285 | 0.0751 |

## Table 6. Intersection-query strict audit

| Family | L | #intersection queries | B2-offline ΔAJ_RD | B2-offline ΔAJ | online-offline ΔAJ |
| --- | --- | --- | --- | --- | --- |
| translate | 8 | 11808 | 0.0780 | -0.1127 | -33.5104 |
| translate | 16 | 11808 | 0.0766 | -0.0481 | -32.3509 |
| translate | 32 | 11808 | 0.0838 | 0.0782 | -30.5219 |
| occluder | 8 | 10825 | 0.0443 | -0.4532 | -35.2690 |
| occluder | 16 | 10825 | 0.0446 | -0.5532 | -34.7157 |
| occluder | 32 | 10825 | 0.0437 | -0.7156 | -33.9696 |

## Table 7. L16 stress ablation

| Family | Method | AJ_RD_256 | AJ_256 | ΔAJ_RD vs offline | ΔAJ vs offline |
| --- | --- | --- | --- | --- | --- |
| translate | offline | 0.4708 | 75.8656 | — | — |
| translate | online_global | 0.5194 | 43.9771 | 0.0486 | -31.8885 |
| translate | b2_fullpost_p1 | 0.5311 | 75.2650 | 0.0603 | -0.6006 |
| translate | b2_w8_p2 | 0.5498 | 75.8454 | 0.0790 | -0.0202 |
| translate | b2_w16_p1 | 0.5487 | 75.8343 | 0.0779 | -0.0313 |
| translate | b2_w16_p2 | 0.5485 | 75.8344 | 0.0777 | -0.0312 |
| translate | b2_w32_p2 | 0.5448 | 75.7849 | 0.0740 | -0.0807 |
| occluder | offline | 0.6422 | 77.6880 | — | — |
| occluder | online_global | 0.6477 | 43.6800 | 0.0055 | -34.0080 |
| occluder | b2_fullpost_p1 | 0.6560 | 76.0911 | 0.0138 | -1.5969 |
| occluder | b2_w8_p2 | 0.6897 | 77.2316 | 0.0475 | -0.4564 |
| occluder | b2_w16_p1 | 0.6877 | 77.1701 | 0.0455 | -0.5179 |
| occluder | b2_w16_p2 | 0.6873 | 77.1682 | 0.0451 | -0.5198 |
| occluder | b2_w32_p2 | 0.6822 | 77.0227 | 0.0400 | -0.6653 |

## Table 8. Per-video statistical robustness on dev stress

| Family | L | mean ΔAJ_RD | 95% CI | positive videos | sign-test p | mean ΔAJ | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| translate | 8 | 0.0705 | [0.0488, 0.0931] | 10/10 | 0.001953 | -0.0767 | [-0.5745, 0.4227] |
| translate | 16 | 0.0682 | [0.0499, 0.0870] | 10/10 | 0.001953 | -0.0312 | [-0.4929, 0.4504] |
| translate | 32 | 0.0752 | [0.0533, 0.0995] | 10/10 | 0.001953 | 0.0782 | [-0.3913, 0.5595] |
| occluder | 8 | 0.0427 | [0.0300, 0.0599] | 10/10 | 0.001953 | -0.4282 | [-0.9170, -0.0012] |
| occluder | 16 | 0.0427 | [0.0313, 0.0597] | 10/10 | 0.001953 | -0.5198 | [-0.9738, -0.1154] |
| occluder | 32 | 0.0415 | [0.0276, 0.0607] | 10/10 | 0.001953 | -0.7156 | [-1.1753, -0.2895] |

## Table 9. Frozen RGB fresh20-29 validation

| Stress | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
| --- | --- | --- | --- | --- | --- |
| translate | offline | 0.5049 | 76.4320 | 91.6794 | 86.3829 |
| translate | online | 0.5166 | 43.3072 | 57.0030 | 67.3190 |
| translate | b2_w16_p2 | 0.5583 | 76.2507 | 92.9006 | 86.3832 |
| occluder | offline | 0.6640 | 79.4009 | 92.3044 | 88.5818 |
| occluder | online | 0.6241 | 43.5482 | 58.1152 | 72.1135 |
| occluder | b2_w16_p2 | 0.6879 | 78.7506 | 93.1315 | 88.3889 |

## Table 10. Frozen RGB fresh20-29 gains

| Stress | Comparison | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
| --- | --- | --- | --- | --- |
| translate | online_vs_offline | 0.0117 | -33.1248 | -34.6764 |
| translate | b2_vs_offline | 0.0534 | -0.1813 | 1.2212 |
| translate | b2_vs_online | 0.0417 | 32.9435 | 35.8976 |
| occluder | online_vs_offline | -0.0399 | -35.8527 | -34.1892 |
| occluder | b2_vs_offline | 0.0239 | -0.6503 | 0.8271 |
| occluder | b2_vs_online | 0.0638 | 35.2024 | 35.0163 |

## Table 11. Frozen RGB fresh20-29 per-video robustness

| Stress | mean ΔAJ_RD | 95% CI | positive videos | mean ΔAJ | 95% CI |
| --- | --- | --- | --- | --- | --- |
| translate | 0.0433 | [0.0155, 0.0699] | 8/10 | -0.1813 | [-1.0149, 0.8457] |
| occluder | 0.0224 | [0.0027, 0.0445] | 7/10 | -0.6504 | [-1.4108, 0.2437] |

## Table 12. Frozen RGB fresh20-29 stress-induced-only AJ_RD_256

| Stress | stress-induced event rate | offline | online | B2-W16-P2 | B2-offline | B2-online |
| --- | --- | --- | --- | --- | --- | --- |
| translate L16 | 0.2784 | 0.8253 | 0.7798 | 0.8334 | 0.0081 | 0.0536 |
| occluder L16 | 0.6715 | 0.7794 | 0.7140 | 0.7790 | -0.0004 | 0.0650 |
