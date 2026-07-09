# B2 Per-Video Stability Audit — 2026-06-28

## Decision

B2 is not a single-video artifact. It is broadly stable across the 30-video DAVIS/TAP-Vid set and nearly matches global B1 re-entry behavior while recovering standard AJ on most videos.

## Compared methods

```text
fixed_offline
global_b1_vis4
b2_mainline
b2_persist2
b2_gt_oracle
```

## Aggregate video-weighted metrics

| method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 | video-mean delta_avg_256 | video-mean reentry miss |
|---|---:|---:|---:|---:|---:|
| fixed_offline | 0.497408 | 70.050983 | 92.154353 | 82.320567 | 0.376951 |
| global_b1_vis4 | 0.580244 | 47.404633 | 72.676183 | 76.491333 | 0.122005 |
| b2_mainline | 0.580100 | 68.970223 | 91.185060 | 82.460387 | 0.122005 |
| b2_persist2 | 0.579452 | 69.024543 | 91.216180 | 82.452757 | 0.133463 |
| b2_gt_oracle | 0.580628 | 70.182103 | 92.643887 | 82.349380 | 0.122005 |

Note: this table is video-weighted, while the main paper-facing global score is query/event-weighted. The relative trend is consistent with the global result.

## Stability counts

```text
n_videos = 30
B2 improves AJ_RD vs fixed = 19 / 30
B2 improves AJ_RD vs fixed by >= 0.01 = 18 / 30
B2 AJ drop vs fixed <= 1.5 = 20 / 30
B2 AJ drop vs fixed > 3 = 5 / 30
B2 AJ_RD within 0.005 of global B1 = 25 / 30
B2 AJ_RD loses to B1 by > 0.01 = 0 / 30
B2 AJ beats B1 by >= 10 = 26 / 30
```

## Interpretation

B2 is doing what the method claims: it keeps the global B1 re-entry behavior but restores most standard AJ.

The important finding is that B2 never loses to global B1 by more than 0.01 AJ_RD on any video, while beating B1 standard AJ by at least 10 points on 26 / 30 videos.

## Main gains vs fixed_offline

Largest AJ_RD gains include:

```text
paragliding-launch: +0.3307 AJ_RD, +2.7215 AJ
mbike-trick: +0.2949 AJ_RD, -1.5886 AJ
lab-coat: +0.2061 AJ_RD, -0.4260 AJ
bike-packing: +0.1848 AJ_RD, +0.4697 AJ
loading: +0.1692 AJ_RD, +0.2601 AJ
parkour: +0.1687 AJ_RD, +2.0307 AJ
```

## Main standard-AJ failures vs fixed_offline

Worst AJ drops:

```text
drift-chicane: -11.3197 AJ, +0.0286 AJ_RD
pigs: -5.1670 AJ, +0.0483 AJ_RD
shooting: -4.2294 AJ, -0.0026 AJ_RD
soapbox: -3.3793 AJ, -0.0026 AJ_RD
drift-straight: -3.3061 AJ, -0.0153 AJ_RD
```

These videos should be inspected in trigger taxonomy / false-trigger analysis.

## Oracle gap

B2 is close to the GT-window oracle. The largest AJ_RD gap to oracle is small:

```text
soapbox: -0.0086
bmx-trees: -0.0050
bike-packing: -0.0043
breakdance: -0.0024
parkour: -0.0022
```

This shows that the predicted trigger almost closes the gap to the GT-window localized oracle.

## Next step

Run occlusion-length bucket audit to confirm that B2 gains come from long occlusion / re-entry cases while preserving short/ordinary tracking behavior.
