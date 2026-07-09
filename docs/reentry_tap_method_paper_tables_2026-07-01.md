# ReEntry-TAP Method-Paper Tables and Figures — 2026-07-01

This document collects the main tables for the method-paper framing: **improving re-entry recovery in Tracking Any Point with Selective Local Override / B2-W16-P2**.

## Table 1. RGB fresh20-49 natural result and ablation

| Method | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD | ΔAJ | ΔOA |
| --- | --- | --- | --- | --- | --- | --- |
| offline | 0.3816 | 79.5944 | 91.4636 | — | — | — |
| online_global | 0.4121 | 44.6933 | 55.7585 | 0.0305 | -34.9011 | -35.7051 |
| b2_fullpost_p1 | 0.4219 | 78.9590 | 92.7706 | 0.0403 | -0.6354 | 1.3070 |
| b2_w8_p2 | 0.4462 | 79.0753 | 92.8853 | 0.0646 | -0.5191 | 1.4217 |
| b2_w16_p1 | 0.4456 | 79.0634 | 92.8891 | 0.0640 | -0.5310 | 1.4255 |
| b2_w16_p2 | 0.4454 | 79.0664 | 92.8804 | 0.0638 | -0.5280 | 1.4168 |
| b2_w32_p2 | 0.4421 | 79.0451 | 92.8736 | 0.0605 | -0.5493 | 1.4100 |

Key message: windowed local override is the core mechanism. Online/global improves AJ_RD only modestly but collapses AJ, while W8/W16 local override gives the strongest AJ_RD/AJ tradeoff.

## Table 2. RGB fresh20-49 natural video-level robustness

| Metric | Mean Δ | 95% CI | Positive videos | Sign-test p | Drop >1 | Drop >2 |
| --- | --- | --- | --- | --- | --- | --- |
| AJ_RD_256 | 0.061327 | [0.041946, 0.080477] | 24/30 | 0.00054611 | 0 | 0 |
| AJ_256 | -0.527959 | [-0.971579, -0.066773] | 7/30 | 0.00522288 | 10 | 3 |
| OA_256 | 1.416824 | [0.796169, 2.067319] | 23/30 | 0.00522288 | 1 | 0 |
| delta_avg_256 | -0.046879 | [-0.139175, 0.039170] | 13/30 | 0.58466471 | 0 | 0 |

Key message: the main AJ_RD gain is video-consistent: +0.0613 mean video-level gain, 24/30 positive videos, and CI fully above zero.

## Table 3. Frozen RGB fresh20-49 L16 stress validation

| Stress | Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 |
| --- | --- | --- | --- | --- | --- |
| translate | offline | 0.4788 | 75.1940 | 90.7805 | 85.4834 |
| translate | online | 0.4981 | 43.0978 | 56.7847 | 67.1146 |
| translate | b2_w16_p2 | 0.5310 | 74.8457 | 92.2166 | 85.5166 |
| occluder | offline | 0.6311 | 77.6280 | 90.8918 | 87.4234 |
| occluder | online | 0.6146 | 43.1490 | 58.0983 | 72.0255 |
| occluder | b2_w16_p2 | 0.6588 | 76.7991 | 92.1682 | 87.2293 |

## Table 4. Frozen RGB fresh20-49 L16 stress gains

| Stress | Comparison | ΔAJ_RD_256 | ΔAJ_256 | ΔOA_256 |
| --- | --- | --- | --- | --- |
| translate | online_vs_offline | 0.0193 | -32.0962 | -33.9958 |
| translate | b2_vs_offline | 0.0522 | -0.3483 | 1.4361 |
| translate | b2_vs_online | 0.0329 | 31.7479 | 35.4319 |
| occluder | online_vs_offline | -0.0165 | -34.4790 | -32.7935 |
| occluder | b2_vs_offline | 0.0277 | -0.8289 | 1.2764 |
| occluder | b2_vs_online | 0.0442 | 33.6501 | 34.0699 |

## Table 5. Frozen RGB fresh20-49 stress-induced-only AJ_RD_256

| Stress | Stress-induced rate | offline | online | B2 | B2-offline | B2-online |
| --- | --- | --- | --- | --- | --- | --- |
| translate_exit_reenter_L16 | 0.236414 | 0.7618 | 0.7478 | 0.7802 | 0.0184 | 0.0324 |
| moving_occluder_L16 | 0.623460 | 0.7415 | 0.7118 | 0.7541 | 0.0126 | 0.0423 |

Key message: on the 30-video frozen split, B2 remains positive on stress-induced-only re-entry events, not just mixed natural re-entry events.

## Table 6. Frozen RGB fresh20-49 video-level robustness

| Stress | Metric | Mean Δ | 95% CI | Positive videos | Sign-test p |
| --- | --- | --- | --- | --- | --- |
| translate | AJ_RD_256 | 0.045137 | [0.031663, 0.058554] | 27/30 | 0.00000800 |
| translate | AJ_256 | -0.348297 | [-0.766001, 0.091747] | 8/30 | 0.01612500 |
| occluder | AJ_RD_256 | 0.026343 | [0.016307, 0.036927] | 26/30 | 0.00005900 |
| occluder | AJ_256 | -0.828894 | [-1.241856, -0.410184] | 4/30 | 0.00005900 |

## Table 7. Oracle upper-bound analysis

| Setting | Method | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD vs B2 |
| --- | --- | --- | --- | --- | --- |
| rgb_fresh20_49_natural | offline | 0.3816 | 79.5944 | 91.4636 | — |
| rgb_fresh20_49_natural | online_global | 0.4121 | 44.6933 | 55.7585 | — |
| rgb_fresh20_49_natural | b2_w16_p2 | 0.4454 | 79.0664 | 92.8804 | — |
| rgb_fresh20_49_natural | oracle_b2 | 0.4597 | 79.5633 | 92.0193 | 0.0143 |
| rgb_fresh20_49_natural | oracle_online | 0.4677 | 78.2940 | 90.3600 | 0.0223 |
| fresh20_49_translate_L16 | offline | 0.4788 | 75.1940 | 90.7805 | — |
| fresh20_49_translate_L16 | online_global | 0.4981 | 43.0978 | 56.7847 | — |
| fresh20_49_translate_L16 | b2_w16_p2 | 0.5310 | 74.8457 | 92.2166 | — |
| fresh20_49_translate_L16 | oracle_b2 | 0.5437 | 75.3044 | 91.4607 | 0.0127 |
| fresh20_49_translate_L16 | oracle_online | 0.5545 | 74.1730 | 89.7453 | 0.0235 |
| fresh20_49_occluder_L16 | offline | 0.6311 | 77.6280 | 90.8918 | — |
| fresh20_49_occluder_L16 | online_global | 0.6146 | 43.1490 | 58.0983 | — |
| fresh20_49_occluder_L16 | b2_w16_p2 | 0.6588 | 76.7991 | 92.1682 | — |
| fresh20_49_occluder_L16 | oracle_b2 | 0.6732 | 77.6702 | 92.0931 | 0.0144 |
| fresh20_49_occluder_L16 | oracle_online | 0.6820 | 76.0199 | 89.8640 | 0.0232 |

Key message: oracle_b2 consistently adds roughly +0.013 to +0.014 AJ_RD over B2-W16-P2, suggesting modest but real headroom for learned reliability gates.

## Generated figures

- `docs/figures/reentry_tap_method/frozen_fresh20_49_b2_gain_ajrd.png`
- `docs/figures/reentry_tap_method/natural_ablation_ajrd.png`
- `docs/figures/reentry_tap_method/natural_tradeoff_aj_vs_ajrd.png`


## Appendix Table. TrackOn2 first-input plug-in supplement

This table is **not** part of the main ReEntry-TAP strided-original evaluation. It uses a parity-valid DAVIS first-query/input-resolution bridge and should be reported as supplementary external plug-in evidence.

| Method | query-weighted AJ_RD_256 | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 |
|---|---:|---:|---:|---:|
| TrackOn2 first-input | 0.5444 | 0.497728 | 67.0406 | 93.0615 |
| B2-W16-P2, TrackOn2 base + CoTracker3 override | 0.5509 | 0.508408 | 67.1260 | 93.2224 |

Aggregate gain over TrackOn2:

```text
query-weighted AJ_RD_256 +0.0065
video-mean AJ_256 +0.0854
video-mean OA_256 +0.1609
```

Video-level paired AJ_RD difference is positive but not statistically strong:

```text
mean ΔAJ_RD_256 = +0.010680
95% CI = [-0.010289, 0.038233]
positive videos = 12/25 re-entry-valid videos
```

Safe use: supplementary plug-in generality evidence only.
