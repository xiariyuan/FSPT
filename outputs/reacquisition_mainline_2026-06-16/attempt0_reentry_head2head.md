# Re-Entry Head-to-Head Summary

## Cache Alignment

- Compared videos: `30`
- Compared queries: `650`
- Status: `aligned`

## Overall Re-Entry Metrics

| Model | n | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|---:|
| trackon2 | 259 | 1.60 | 6.91 | 75.7% | 85.7% | 31.47 |
| cotracker3_online | 259 | 1.73 | 8.75 | 73.0% | 81.5% | 48.00 |
| cotracker3_offline | 259 | 1.63 | 7.62 | 76.1% | 83.4% | 40.49 |

## Long-Occlusion Re-Entry (occ >= 20)

| Model | n | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|---:|
| trackon2 | 24 | 3.97 | 21.85 | 50.0% | 62.5% | 118.17 |
| cotracker3_online | 24 | 2.99 | 24.47 | 62.5% | 70.8% | 121.88 |
| cotracker3_offline | 24 | 3.19 | 21.54 | 54.2% | 66.7% | 90.48 |

## Pairwise

### trackon2 vs cotracker3_online

- Common re-entry queries: `259`
- Median signed diff `trackon2 - cotracker3_online`: `-0.09px` (negative means `trackon2` better)
- Better fraction: `trackon2` `56.0%`, `cotracker3_online` `44.0%`, `tie` `0.0%`
- Per-video medians: `trackon2` better on `17/25`, `cotracker3_online` better on `8/25`, `tie` `0`

### trackon2 vs cotracker3_offline

- Common re-entry queries: `259`
- Median signed diff `trackon2 - cotracker3_offline`: `-0.08px` (negative means `trackon2` better)
- Better fraction: `trackon2` `55.2%`, `cotracker3_offline` `44.8%`, `tie` `0.0%`
- Per-video medians: `trackon2` better on `11/25`, `cotracker3_offline` better on `14/25`, `tie` `0`

### cotracker3_online vs cotracker3_offline

- Common re-entry queries: `259`
- Median signed diff `cotracker3_online - cotracker3_offline`: `0.05px` (negative means `cotracker3_online` better)
- Better fraction: `cotracker3_online` `45.2%`, `cotracker3_offline` `54.8%`, `tie` `0.0%`
- Per-video medians: `cotracker3_online` better on `12/25`, `cotracker3_offline` better on `13/25`, `tie` `0`

## Occlusion Buckets

### <20

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 235 | 1.46 | 78.3% | 88.1% |
| cotracker3_online | 235 | 1.63 | 74.0% | 82.6% |
| cotracker3_offline | 235 | 1.55 | 78.3% | 85.1% |

### 20-49

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 24 | 3.97 | 50.0% | 62.5% |
| cotracker3_online | 24 | 2.99 | 62.5% | 70.8% |
| cotracker3_offline | 24 | 3.19 | 54.2% | 66.7% |

### 50-99

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 0 | - | - | - |
| cotracker3_online | 0 | - | - | - |
| cotracker3_offline | 0 | - | - | - |

### 100+

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 0 | - | - | - |
| cotracker3_online | 0 | - | - | - |
| cotracker3_offline | 0 | - | - | - |
