# Official-Style RGB-Stacking Full50 Strided Comparison — 2026-07-04

## 1. Scope

This document records the stricter official-style local evaluation for TAPVid RGB-Stacking full50.

Protocol:

```text
Dataset: TAPVid RGB-Stacking
Videos: 50
Queries: 60,829
Query mode: strided
Metric resolution: input / 256-style local cache
Metrics: AJ / OA / δ_avg / δ_4px
Diagnostic metric: AJ_RD
Evaluation type: official-protocol local evaluation, not official leaderboard submission
```

Primary artifact:

```text
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided_input/paired_rgb_full50_strided_input_official_metric.json
```

Related artifacts:

```text
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided/paired_rgb_full50_strided_vs_all.json
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided_256/paired_rgb_full50_strided256_vs_all.json
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided_input/rescore_input/
```

---

## 2. Main results

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

---

## 3. Paired-video checks

V1 minus offline:

```text
AJ    mean -0.5043 pp, CI [-0.8046, -0.2011], W/L/T = 9/41/0
OA    mean +1.4387 pp, CI [+0.9589, +1.9236], W/L/T = 39/11/0
δ_avg mean +0.0000 pp, W/L/T = 0/0/50
```

V22Q minus offline:

```text
AJ    mean -0.3589 pp, CI [-0.6391, -0.0670], W/L/T = 10/40/0
OA    mean +1.4110 pp, CI [+0.9452, +1.8875], W/L/T = 40/10/0
δ_avg mean +0.0000 pp, W/L/T = 0/0/50
```

V24 minus offline:

```text
AJ    mean -0.3371 pp, CI [-0.6065, -0.0531], W/L/T = 11/39/0
OA    mean +1.4018 pp, CI [+0.9438, +1.8769], W/L/T = 40/10/0
δ_avg mean +0.0000 pp, W/L/T = 0/0/50
```

V24 minus V22Q:

```text
AJ    mean +0.0217 pp, CI [+0.0100, +0.0380], W/L/T = 39/6/5
OA    mean -0.0092 pp, CI [-0.0283, +0.0055], W/L/T = 19/25/6
δ_avg mean +0.0000 pp, W/L/T = 0/0/50
```

---

## 4. Interpretation

This is a stricter official-style full50 result than the earlier local full50 wording.

It supports the following claim:

```text
On TAPVid RGB-Stacking full50 under strided official-style local evaluation, ReEntry substantially improves AJ_RD and OA over the CoTracker3 offline base, while introducing a small standard-AJ trade-off and leaving position-only δ_avg unchanged.
```

The result is strong for the paper because:

```text
1. It uses the complete RGB-Stacking full50 set.
2. It includes standard TAP-Vid metrics AJ/OA/δ_avg.
3. It includes paired-video confidence intervals.
4. It confirms the same trade-off pattern seen in the local full50 summary.
```

It should not be described as:

```text
official leaderboard submission
full TAP-Vid benchmark submission
universal standard-AJ improvement
```

---

## 5. Recommended paper wording

Safe wording:

```text
On TAPVid RGB-Stacking full50 under a strided official-style local evaluation, ReEntry improves AJ_RD from 0.3617 to 0.4414 with V1 and to 0.4394 with V24-DINOScore. It also improves OA by approximately +1.40 to +1.44 points, while incurring a small AJ trade-off of roughly -0.34 to -0.50 points.
```

More compact wording:

```text
RGB-Stacking full50 confirms the main trade-off: ReEntry improves re-entry recovery and occlusion accuracy, while slightly reducing standard AJ.
```
