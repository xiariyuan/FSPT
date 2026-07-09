# Official-Style RGB-Stacking Full50 Strided/256 Audit — 2026-07-04

## 1. Purpose

This document audits whether the existing TAPVid RGB-Stacking full50 result can be described as an official-style local evaluation.

Key conclusion:

```text
Yes, with precise wording: TAPVid RGB-Stacking full50 official-style strided/256 local evaluation.
```

This is still not an official leaderboard/server submission.

---

## 2. Cache/protocol audit

Audited caches:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/offline_rgb_stacking_full50.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/online_rgb_stacking_full50.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v1_thr010/reentry_v1_rgb_full50_thr010.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v22Q/reentry_v22Q_rgb_full50.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_dinoscore/reentry_v24_dinoscore_rgb_full50.pt
```

Although the merged cache metadata says:

```text
protocol = first_input_bridge/merged_existing_caches
```

actual query audit shows:

```text
Dataset: tapvid_rgb_stacking
Split: rgb_stacking_full50_0_49
Videos: 50 / 50
Video IDs: rgb_stacking_000000 -- rgb_stacking_000049
Queries: 60,829
Input/original size: 256 x 256
Query frames: 0, 5, 10, 15, ...
All query frames are multiples of 5: yes
Query frame visible rate: 1.0
Query anchor error: 0.0
```

Therefore, the actual query sampling matches the official TAP-Vid `strided` query protocol behavior.

---

## 3. Rescoring method

The generic `attempt0_rescore_cache.py` could not directly read the RGB full50 merged cache because its `schema_version` is a string:

```text
schema_version = reentry_rgb_full50_merged_v1
```

Therefore, a read-only custom rescoring script was used. It directly calls:

```text
datasets.metrics.compute_tapvid_metrics(..., query_mode="strided", resolution=(256, 256))
```

This wrapper delegates to the vendored official TAP-Vid metric core:

```text
datasets.tapvid_official_eval.compute_tapvid_metrics_official
```

Generated artifacts:

```text
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided_256/custom_rescore/
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided_256/ajrd/
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided_256/paired_rgb_full50_strided256_vs_all.json
```

---

## 4. Main results

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD |
|---|---:|---:|---:|---:|---:|
| CoTracker3 offline | 79.9345 | 91.6371 | 88.5714 | 91.5875 | 0.3617 |
| CoTracker3 online | 44.8432 | 55.8092 | 73.1881 | 83.3099 | 0.4028 |
| ReEntry V1 | 79.4302 | 93.0758 | 88.5714 | 91.5875 | 0.4414 |
| ReEntry V22Q | 79.5756 | 93.0481 | 88.5714 | 91.5875 | 0.4400 |
| ReEntry V24-DINOScore | 79.5974 | 93.0389 | 88.5714 | 91.5875 | 0.4394 |

Deltas vs CoTracker3 offline:

| Method | ΔAJ | ΔOA | Δδ_avg | ΔAJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 online | -35.0912 | -35.8280 | -15.3833 | +0.0411 |
| ReEntry V1 | -0.5043 | +1.4387 | +0.0000 | +0.0797 |
| ReEntry V22Q | -0.3589 | +1.4110 | +0.0000 | +0.0783 |
| ReEntry V24-DINOScore | -0.3371 | +1.4018 | +0.0000 | +0.0777 |

---

## 5. Paired-video statistics

```text
V1 - offline:
  AJ    mean -0.5043 pp, CI [-0.8055, -0.1928], W/L/T = 9/41/0
  OA    mean +1.4387 pp, CI [+0.9547, +1.9219], W/L/T = 39/11/0
  δ_avg mean +0.0000 pp, W/L/T = 0/0/50

V22Q - offline:
  AJ    mean -0.3589 pp, CI [-0.6370, -0.0716], W/L/T = 10/40/0
  OA    mean +1.4110 pp, CI [+0.9471, +1.8881], W/L/T = 40/10/0
  δ_avg mean +0.0000 pp, W/L/T = 0/0/50

V24 - offline:
  AJ    mean -0.3371 pp, CI [-0.6182, -0.0483], W/L/T = 11/39/0
  OA    mean +1.4018 pp, CI [+0.9419, +1.8734], W/L/T = 40/10/0
  δ_avg mean +0.0000 pp, W/L/T = 0/0/50
```

Micro-comparisons:

```text
V24 - V22Q:
  AJ mean +0.0217 pp, CI [+0.0096, +0.0381], W/L/T = 39/6/5
  OA mean -0.0092 pp, CI [-0.0282, +0.0057], W/L/T = 19/25/6

V24 - V1:
  AJ mean +0.1672 pp, CI [+0.1089, +0.2572], W/L/T = 49/1/0
  OA mean -0.0369 pp, CI [-0.1237, +0.0240], W/L/T = 20/30/0
```

---

## 6. Interpretation

This result can now be described more strongly than before:

```text
TAPVid RGB-Stacking full50 official-style strided/256 local evaluation.
```

Main conclusion:

```text
On the full 50-video RGB-Stacking official-style strided/256 evaluation, ReEntry substantially improves AJ_RD and OA while leaving δ_avg unchanged and incurring a small AJ trade-off.
```

Best positioning:

```text
V1 is best for AJ_RD/OA.
V22Q improves standard AJ relative to V1 with a small AJ_RD/OA cost.
V24 further improves standard AJ relative to V22Q but slightly reduces OA and AJ_RD.
```

---

## 7. Safe paper wording

Recommended:

```text
On TAPVid RGB-Stacking full50 under an official-style strided/256 local evaluation, ReEntry improves AJ_RD by +0.0797 and OA by +1.4387 with V1, while maintaining δ_avg and trading off -0.5043 AJ. V24 reduces the AJ trade-off to -0.3371 while preserving a +0.0777 AJ_RD gain and +1.4018 OA gain.
```

Avoid:

```text
official leaderboard submission
full TAP-Vid benchmark submission
all metrics improve
```

---

## 8. Impact on protocol status

Updated status:

```text
DAVIS first/input: official-style local evaluation complete.
DAVIS strided/original: official-style local evaluation complete.
RGB-Stacking full50 strided/256: official-style local evaluation complete after query audit and custom rescore.
```

Still missing for a full TAP-Vid benchmark claim:

```text
Kinetics full evaluation
Kubric full evaluation
official server/leaderboard submission, if such a server is available
```
