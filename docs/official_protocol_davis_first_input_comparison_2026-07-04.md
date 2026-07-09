# Official-Style DAVIS First/Input Comparison — 2026-07-04

## 1. Scope

This document adds a query-first / first-input official-style local evaluation for TAPVid-DAVIS.

Protocol:

```text
Dataset: TAPVid-DAVIS
Videos: 30
Queries: 650
Query mode for official metric: first
Metric resolution mode: input
Input resolution: 256x256
Metrics: AJ / OA / δ_avg / δ_4px
Diagnostic metric: AJ_RD
Evaluation type: official-protocol local evaluation, not official leaderboard submission
```

Source caches:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v1_offline_to_baseline_thr010/reentry_v1_davis_offline_to_baseline_thr010.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v22Q_offline_to_baseline/reentry_v22Q_davis_offline_to_baseline.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v24_dinoscore_offline_to_baseline/reentry_v24_dinoscore_davis_offline_to_baseline.pt
```

Generated outputs:

```text
outputs/paper_discovery_2026-06-27/official_protocol_davis_first_input/rescore_input/
outputs/paper_discovery_2026-06-27/official_protocol_davis_first_input/ajrd/
outputs/paper_discovery_2026-06-27/official_protocol_davis_first_input/paired_first_input_vs_all.json
```

---

## 2. Main results

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 baseline | 64.8949 | 91.7983 | 77.3560 | 84.9284 | 0.3486 | 0.5246 |
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 84.5776 | 0.3142 | 0.4525 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 87.8233 | 0.3714 | 0.5444 |
| ReEntry V1 | 64.5758 | 91.7274 | 77.2244 | 84.5776 | 0.3588 | 0.5305 |
| ReEntry V22Q | 64.7260 | 91.7406 | 77.2244 | 84.5776 | 0.3556 | 0.5252 |
| ReEntry V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 84.5776 | 0.3549 | 0.5236 |

Deltas vs CoTracker3 offline:

| Method | ΔAJ | ΔOA | Δδ_avg | ΔAJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 baseline | +2.2383 | +3.6496 | +0.1316 | +0.0344 |
| TrackOn2 | +4.3841 | +3.9429 | +2.6174 | +0.0572 |
| ReEntry V1 | +1.9193 | +3.5787 | +0.0000 | +0.0446 |
| ReEntry V22Q | +2.0695 | +3.5919 | +0.0000 | +0.0414 |
| ReEntry V24-DINOScore | +2.1874 | +3.7363 | +0.0000 | +0.0407 |

---

## 3. Paired-video standard metric checks

Paired differences vs CoTracker3 offline:

```text
ReEntry V1 - offline:
  AJ    mean +1.9193 pp, CI [+0.8375, +3.0545], W/L/T = 22/7/1
  OA    mean +3.5787 pp, CI [+1.8485, +5.6697], W/L/T = 24/5/1
  δ_avg mean +0.0000 pp, W/L/T = 0/0/30

ReEntry V22Q - offline:
  AJ    mean +2.0695 pp, CI [+1.0763, +3.1667], W/L/T = 23/6/1
  OA    mean +3.5919 pp, CI [+2.0011, +5.5624], W/L/T = 25/4/1
  δ_avg mean +0.0000 pp, W/L/T = 0/0/30

ReEntry V24 - offline:
  AJ    mean +2.1874 pp, CI [+1.2582, +3.2487], W/L/T = 24/5/1
  OA    mean +3.7363 pp, CI [+2.1673, +5.7423], W/L/T = 27/2/1
  δ_avg mean +0.0000 pp, W/L/T = 0/0/30
```

V24 compared with baseline and V22Q:

```text
V24 - CoTracker3 baseline:
  AJ    mean -0.0510 pp, CI [-0.9409, +0.6909], W/L/T = 16/14/0
  OA    mean +0.0868 pp, CI [-0.3916, +0.5818], W/L/T = 14/14/2
  δ_avg mean -0.1316 pp, CI [-1.0177, +0.7479], W/L/T = 15/15/0

V24 - V22Q:
  AJ    mean +0.1179 pp, CI [+0.0204, +0.2386], W/L/T = 7/6/17
  OA    mean +0.1445 pp, CI [+0.0019, +0.3138], W/L/T = 6/6/18
  δ_avg mean +0.0000 pp, W/L/T = 0/0/30
```

---

## 4. Interpretation

This result is important because it complements the DAVIS strided+original audit.

Under DAVIS strided+original:

```text
Default V1/V22Q improved AJ_RD but hurt official-style AJ/OA.
```

Under DAVIS first/input:

```text
Default V1/V22Q/V24 improve AJ, OA, and AJ_RD over the CoTracker3 offline base.
V24 gives the strongest ReEntry standard metric result among V1/V22Q/V24.
TrackOn2 remains the strongest external baseline overall.
```

This means the method is protocol-sensitive:

```text
The ReEntry visibility correction is helpful under first-query evaluation and local bridge settings, but needs conservative thresholding or metric-aware selection under strided+original evaluation.
```

---

## 5. Safe paper wording

Recommended wording:

```text
On TAPVid-DAVIS under a first-query/input-resolution official-style evaluation, ReEntry improves over the CoTracker3 offline base by +2.19 AJ and +3.74 OA with V24-DINOScore, while also improving AJ_RD by +0.0407. Under the stricter strided+original protocol, the default method improves AJ_RD but requires a conservative threshold to preserve official-style AJ/OA.
```

Avoid:

```text
ReEntry universally improves all official protocols.
V24 beats TrackOn2 on DAVIS.
This is an official leaderboard submission.
```

---

## 6. How to use this in the paper

Use this table as the main DAVIS first-query official-style result.

Use `docs/official_protocol_audit_2026-07-04.md` for the strided+original cautionary table.

Together, they support a nuanced claim:

```text
ReEntry improves re-entry behavior and first-query DAVIS metrics, but official strided evaluation requires conservative or metric-aware selection.
```
