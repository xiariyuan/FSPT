# B1 Candidate Action Oracle — 2026-06-28

## Decision

Baseline action: `vis4_gated288`.
Baseline `true_AJ_RD_256`: `0.6279`.
Candidate-action oracle `true_AJ_RD_256`: `0.6531`.
Gain over baseline: `+0.0252`.

This shows the current action set still has measurable routing headroom. The oracle is diagnostic only because it selects the best action after seeing per-query outcomes.

## Single-action ranking

| rank | action | mean AJ_RD_256 | delta vs baseline | n |
|---:|---|---:|---:|---:|
| 1 | vis4_gated384 | 0.627927 | 0.000071 | 1385 |
| 2 | vis4_gated320 | 0.627883 | 0.000027 | 1385 |
| 3 | vis4_gated288 | 0.627857 | 0.000000 | 1385 |
| 4 | trimTrack24_visG256 | 0.626913 | -0.000943 | 1385 |
| 5 | vis4_gated256 | 0.626912 | -0.000944 | 1385 |
| 6 | trimTrack48_visG256 | 0.626868 | -0.000988 | 1385 |
| 7 | hybridTrack64_visG256 | 0.626813 | -0.001044 | 1385 |
| 8 | hybridTrack128_visG256 | 0.626786 | -0.001070 | 1385 |
| 9 | all4_gated320 | 0.626606 | -0.001250 | 1385 |
| 10 | all4_gated288 | 0.626580 | -0.001277 | 1385 |
| 11 | all4_gated256 | 0.626529 | -0.001328 | 1385 |
| 12 | all4_gated192 | 0.626422 | -0.001435 | 1385 |
| 13 | b1_visible_median4_union | 0.625636 | -0.002221 | 1385 |
| 14 | b1_all_median4_union | 0.624368 | -0.003488 | 1385 |
| 15 | old3_all_median_gated144 | 0.618874 | -0.008983 | 1385 |
| 16 | old3_all_median_union | 0.617126 | -0.010731 | 1385 |
| 17 | b1_all_median4_half | 0.606443 | -0.021414 | 1385 |

## Oracle action usage

| action | queries | share |
|---|---:|---:|
| vis4_gated384 | 755 | 54.5% |
| b1_all_median4_half | 233 | 16.8% |
| old3_all_median_union | 221 | 16.0% |
| trimTrack48_visG256 | 139 | 10.0% |
| trimTrack24_visG256 | 14 | 1.0% |
| hybridTrack64_visG256 | 11 | 0.8% |
| hybridTrack128_visG256 | 6 | 0.4% |
| old3_all_median_gated144 | 5 | 0.4% |
| b1_all_median4_union | 1 | 0.1% |

## Delta threshold counts

| threshold | queries | share |
|---:|---:|---:|
| 0.0 | 630 | 45.5% |
| 0.001 | 627 | 45.3% |
| 0.002 | 618 | 44.6% |
| 0.005 | 582 | 42.0% |
| 0.01 | 521 | 37.6% |
| 0.02 | 401 | 29.0% |
| 0.05 | 217 | 15.7% |

## Top video gains

| video | n | baseline | action oracle | delta |
|---|---:|---:|---:|---:|
| pigs | 30 | 0.581670 | 0.683783 | 0.102113 |
| libby | 40 | 0.460347 | 0.518650 | 0.058303 |
| shooting | 14 | 0.392314 | 0.446807 | 0.054493 |
| motocross-jump | 6 | 0.185450 | 0.226150 | 0.040700 |
| mbike-trick | 46 | 0.429967 | 0.467865 | 0.037898 |
| paragliding-launch | 19 | 0.616447 | 0.653268 | 0.036821 |
| parkour | 97 | 0.197635 | 0.234238 | 0.036603 |
| bike-packing | 62 | 0.520121 | 0.556419 | 0.036298 |
| camel | 4 | 0.864950 | 0.896750 | 0.031800 |
| bmx-trees | 205 | 0.583382 | 0.613361 | 0.029980 |

## Next step

Train a baseline-preserving safe router. It should default to `vis4_gated288` and use another action only when predicted delta is above a margin. Do not repeat the direct learned visibility replacement experiment as the main route.
