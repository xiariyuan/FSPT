# RGB-Stacking Cross-Dataset Smoke — 2026-06-28

## Decision

The TAP-Vid RGB-Stacking dataset is present locally and the official evaluation path is usable.

GT-as-prediction sanity on the first 5 videos passed:

```text
mean AJ = 100.0000
mean OA = 100.0000
mean delta_avg = 100.0000
```

## Dataset schema

Local file:

```text
/gemini/code/datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl
```

Schema observed:

```text
type: list
length: 50
item keys: points, occluded, video
video: (250, 256, 256, 3) uint8
points: (30, 250, 2) float64, normalized [x,y]-style coordinates
occluded: (30, 250) bool
```

## Smoke rows

| index | n_queries | AJ | OA | delta_avg | occ_rate | points_min | points_max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1148 | 100.0000 | 100.0000 | 100.0000 | 0.236933 | -0.267642 | 0.973959 |
| 1 | 1382 | 100.0000 | 100.0000 | 100.0000 | 0.081467 | -0.040364 | 0.890624 |
| 2 | 936 | 100.0000 | 100.0000 | 100.0000 | 0.380800 | -0.156912 | 0.987631 |
| 3 | 1061 | 100.0000 | 100.0000 | 100.0000 | 0.293067 | -0.233945 | 0.980469 |
| 4 | 1225 | 100.0000 | 100.0000 | 100.0000 | 0.183733 | -0.081197 | 1.027182 |

## Interpretation

The dataset and official TAP-Vid metric path are ready for cross-dataset validation. This does not yet validate the B1 fusion method on RGB-Stacking because no teacher prediction caches have been exported for this dataset yet.

## Next required step

Build RGB-Stacking teacher cache exporters for the same teacher pool:

```text
cotracker3_online
cotracker3_offline
trackon2
tapnext
```

Recommended staged validation:

1. export 1-video smoke caches;
2. evaluate single-teacher AJ_RD / standard metrics;
3. export 5-video teacher caches;
4. run B1 fusion rules on RGB-Stacking;
5. only then scale to all 50 videos.
