# Re-Entry Head-to-Head Summary

## Cache Alignment

- Compared videos: `30`
- Compared queries: `5882`
- Status: `aligned`

## Overall Re-Entry Metrics

| Model | n | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|---:|
| trackon2 | 1385 | 3.91 | 18.47 | 51.0% | 73.6% | 109.77 |
| cotracker3_online | 1385 | 3.86 | 14.73 | 51.3% | 77.0% | 74.95 |
| cotracker3_offline | 1385 | 3.71 | 12.21 | 53.8% | 80.2% | 44.97 |

## Long-Occlusion Re-Entry (occ >= 20)

| Model | n | median px | mean px | <4px | <8px | p95 px |
|---|---:|---:|---:|---:|---:|---:|
| trackon2 | 146 | 5.70 | 32.65 | 37.7% | 56.2% | 198.04 |
| cotracker3_online | 146 | 6.39 | 31.15 | 31.5% | 58.2% | 226.24 |
| cotracker3_offline | 146 | 5.58 | 22.10 | 31.5% | 69.9% | 164.99 |

## Pairwise

### trackon2 vs cotracker3_online

- Common re-entry queries: `1385`
- Median signed diff `trackon2 - cotracker3_online`: `0.26px` (negative means `trackon2` better)
- Better fraction: `trackon2` `43.0%`, `cotracker3_online` `57.0%`, `tie` `0.0%`
- Per-video medians: `trackon2` better on `12/25`, `cotracker3_online` better on `13/25`, `tie` `0`

### trackon2 vs cotracker3_offline

- Common re-entry queries: `1385`
- Median signed diff `trackon2 - cotracker3_offline`: `0.31px` (negative means `trackon2` better)
- Better fraction: `trackon2` `41.4%`, `cotracker3_offline` `58.6%`, `tie` `0.0%`
- Per-video medians: `trackon2` better on `10/25`, `cotracker3_offline` better on `15/25`, `tie` `0`

### cotracker3_online vs cotracker3_offline

- Common re-entry queries: `1385`
- Median signed diff `cotracker3_online - cotracker3_offline`: `0.12px` (negative means `cotracker3_online` better)
- Better fraction: `cotracker3_online` `44.8%`, `cotracker3_offline` `55.2%`, `tie` `0.0%`
- Per-video medians: `cotracker3_online` better on `8/25`, `cotracker3_offline` better on `17/25`, `tie` `0`

## Occlusion Buckets

### <20

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 1239 | 3.80 | 52.5% | 75.7% |
| cotracker3_online | 1239 | 3.69 | 53.7% | 79.2% |
| cotracker3_offline | 1239 | 3.55 | 56.4% | 81.4% |

### 20-49

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 144 | 5.43 | 38.2% | 56.9% |
| cotracker3_online | 144 | 6.35 | 31.9% | 59.0% |
| cotracker3_offline | 144 | 5.44 | 31.9% | 70.8% |

### 50-99

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 2 | 223.05 | 0.0% | 0.0% |
| cotracker3_online | 2 | 210.26 | 0.0% | 0.0% |
| cotracker3_offline | 2 | 203.11 | 0.0% | 0.0% |

### 100+

| Model | n | median px | <4px | <8px |
|---|---:|---:|---:|---:|
| trackon2 | 0 | - | - | - |
| cotracker3_online | 0 | - | - | - |
| cotracker3_offline | 0 | - | - | - |
