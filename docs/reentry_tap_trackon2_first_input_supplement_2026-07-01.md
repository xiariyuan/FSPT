# TrackOn2 First-input Plug-in Supplement — 2026-07-01

This supplement uses the parity-valid DAVIS first-query/input-resolution bridge. It is **not** mixed into the main ReEntry-TAP strided-original tables.

## Aggregate results

| Method | query-weighted AJ_RD_256 | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 | video-mean delta_avg_256 | re-entry queries |
| --- | --- | --- | --- | --- | --- | --- |
| cotracker3_offline_first | 0.4525 | 0.409808 | 62.1661 | 89.0107 | 77.2244 | 259 |
| trackon2_first | 0.5444 | 0.497728 | 67.0406 | 93.0615 | 79.8418 | 259 |
| b2_w16_trackon_base | 0.5513 | 0.508632 | 67.1750 | 93.2513 | 79.8543 | 259 |
| b2_w16_p2_trackon_base | 0.5509 | 0.508408 | 67.1260 | 93.2224 | 79.8202 | 259 |
| b2_w16_cotracker_base | 0.5492 | 0.509392 | 64.4149 | 92.6774 | 77.8309 | 259 |
| b2_w16_p2_cotracker_base | 0.5495 | 0.509472 | 64.4295 | 92.7140 | 77.8145 | 259 |

## Key paired comparisons

### b2_w16_p2_trackon_base_vs_trackon2_first
| Metric | mean Δ | 95% CI | positive videos | sign-test p |
| --- | --- | --- | --- | --- |
| AJ_RD_256 | 0.010680 | [-0.010289, 0.038233] | 12/25 | 0.50344467 |
| AJ_256 | 0.085407 | [-0.394355, 0.544089] | 13/30 | 1.00000000 |
| OA_256 | 0.160953 | [-0.206127, 0.582687] | 13/30 | 1.00000000 |
| delta_avg_256 | -0.021600 | [-0.380479, 0.270292] | 15/30 | 0.30745625 |

### b2_w16_trackon_base_vs_trackon2_first
| Metric | mean Δ | 95% CI | positive videos | sign-test p |
| --- | --- | --- | --- | --- |
| AJ_RD_256 | 0.010904 | [-0.010436, 0.038757] | 12/25 | 0.50344467 |
| AJ_256 | 0.134400 | [-0.381118, 0.628563] | 13/30 | 1.00000000 |
| OA_256 | 0.189790 | [-0.202248, 0.622775] | 13/30 | 1.00000000 |
| delta_avg_256 | 0.012477 | [-0.370895, 0.343311] | 14/30 | 0.69003797 |

### trackon2_first_vs_cotracker3_offline_first
| Metric | mean Δ | 95% CI | positive videos | sign-test p |
| --- | --- | --- | --- | --- |
| AJ_RD_256 | 0.087920 | [0.044968, 0.133900] | 18/25 | 0.02265584 |
| AJ_256 | 4.874590 | [3.006533, 6.828148] | 23/30 | 0.00522288 |
| OA_256 | 4.050810 | [2.617869, 5.639178] | 27/30 | 0.00000843 |
| delta_avg_256 | 2.617420 | [0.958995, 4.429041] | 20/30 | 0.09873715 |

### b2_w16_p2_cotracker_base_vs_cotracker3_offline_first
| Metric | mean Δ | 95% CI | positive videos | sign-test p |
| --- | --- | --- | --- | --- |
| AJ_RD_256 | 0.099664 | [0.057908, 0.143708] | 20/25 | 0.00154388 |
| AJ_256 | 2.263490 | [1.177040, 3.440378] | 20/30 | 0.06142835 |
| OA_256 | 3.703323 | [2.381877, 5.318692] | 26/30 | 0.00001524 |
| delta_avg_256 | 0.590090 | [-0.080528, 1.342415] | 15/30 | 0.85055402 |

## Safe paper use

```text
Under a parity-valid first-query/input-resolution DAVIS protocol, B2-W16-P2 gives a small positive aggregate gain on top of a reproduced TrackOn2 baseline. This supports plug-in generality, but should be reported as supplementary evidence rather than a main ReEntry-TAP stress baseline.
```