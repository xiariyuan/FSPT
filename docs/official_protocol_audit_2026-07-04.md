# Official Protocol Audit — TAPVid DAVIS Strided + Original — 2026-07-04

## 1. Purpose

This audit checks whether the current ReEntry results can be described as official leaderboard results.

Short answer:

```text
No. The current RGB-Stacking full50 / DAVIS bridge results are strong local standard evaluations, but not official leaderboard submissions.
```

This audit also runs the existing ReEntry methods on the closest available official-style protocol in the repository:

```text
TAPVid-DAVIS
query mode: strided
metric resolution: original
protocol label: strided+original
```

---

## 2. Official-style protocol found in repository

Relevant files:

```text
datasets/tapvid_official_eval.py
scripts/attempt0_eval_strided_original.py
scripts/attempt0_export_strided_original_cache.py
scripts/attempt0_rescore_cache.py
```

Important protocol properties:

```text
query_mode = strided
query_stride = 5
metric_resolution_mode = original
metrics = official TAPVid AJ / OA / average_pts_within_thresh
```

The metric implementation uses the vendored TAPVid official core:

```text
datasets/tapvid_official_eval.py::compute_tapvid_metrics_official
```

This is closer to official leaderboard protocol than the earlier local bridge results.

---

## 3. Difference from previous results

Previous strong results used local bridge protocols:

```text
TAPVid-DAVIS bridge:
  first-input bridge cache
  30 videos
  650 queries
  metric resolution = 256 in local eval

TAPVid RGB-Stacking full50:
  local merged cache over videos 0--49
  50 videos
  60,829 queries
  metric resolution = 256 in local eval
```

Those are valid local standard evaluations, but they are not official leaderboard submissions.

For a result to be called official leaderboard result, it must be:

```text
1. Generated under the official benchmark protocol.
2. Evaluated with official metrics and required resolution/query mode.
3. Formatted as an official submission if the benchmark requires hidden evaluation.
4. Submitted to the official leaderboard/evaluation server.
5. Returned by the official leaderboard/evaluation server.
```

Current status:

```text
We have official-style local evaluation artifacts, but no official server submission or returned leaderboard score.
```

---

## 4. Existing DAVIS official-style caches

Existing base/external caches:

```text
outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt
outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt
caches/trackon2_strided_original.pt
outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt
```

Generated ReEntry caches under official-style DAVIS protocol:

```text
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/v1_thr010/reentry_v1_davis_strided_original_thr010.pt
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/v22Q/reentry_v22Q_davis_strided_original.pt
```

Official-style original-resolution rescoring outputs:

```text
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/rescore_original/offline_strided_original_rescore.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/rescore_original/online_strided_original_rescore.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/rescore_original/trackon2_strided_original_rescore.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/rescore_original/tapnext_strided_original_rescore.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/rescore_original/v1_strided_original_rescore.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/rescore_original/v22Q_strided_original_rescore.json
```

Official-style AJ_RD outputs:

```text
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/ajrd/offline_ajrd.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/ajrd/online_ajrd.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/ajrd/trackon2_ajrd.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/ajrd/tapnext_ajrd.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/ajrd/v1_ajrd.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/ajrd/v22Q_ajrd.json
```

---

## 5. Official-style DAVIS standard metrics

These metrics are original-resolution local rescoring under:

```text
query_mode = strided
metric_resolution_mode = original
```

| Method | AJ | OA | δ_avg | δ_4px |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 68.0060 |
| CoTracker3 online | 36.9543 | 68.1493 | 45.0072 | 47.7100 |
| TrackOn2 strided_original cache | 28.3785 | 58.3304 | 33.2280 | 35.8975 |
| TAPNext strided_original cache | 25.9433 | 65.9795 | 33.4476 | 36.1557 |
| ReEntry V1 | 50.7303 | 91.2730 | 63.5892 | 68.0060 |
| ReEntry V22Q | 50.8342 | 91.3439 | 63.5892 | 68.0060 |

Deltas vs CoTracker3 offline:

| Method | ΔAJ | ΔOA | Δδ_avg | Δδ_4px |
|---|---:|---:|---:|---:|
| ReEntry V1 | -0.8081 | -0.8813 | +0.0000 | +0.0000 |
| ReEntry V22Q | -0.7043 | -0.8105 | +0.0000 | +0.0000 |

Per-video paired deltas:

```text
V1 - offline:
  AJ mean = -0.8081 pp, CI [-1.5417, -0.2582], W/L/T = 10/19/1
  OA mean = -0.8813 pp, CI [-1.9671, +0.0659], W/L/T = 14/15/1
  δ_avg mean = +0.0000 pp, W/L/T = 0/0/30

V22Q - offline:
  AJ mean = -0.7043 pp, CI [-1.4455, -0.1659], W/L/T = 11/18/1
  OA mean = -0.8105 pp, CI [-1.8538, +0.0780], W/L/T = 13/16/1
  δ_avg mean = +0.0000 pp, W/L/T = 0/0/30

V22Q - V1:
  AJ mean = +0.1038 pp, CI [+0.0306, +0.2052], W/L/T = 23/0/7
  OA mean = +0.0709 pp, CI [-0.0453, +0.2585], W/L/T = 14/8/8
```

Interpretation:

```text
Under official-style DAVIS strided+original standard metrics, the current ReEntry methods do not improve leaderboard-style AJ/OA over the CoTracker3 offline base. They keep position metrics unchanged, but visibility changes reduce AJ/OA.
```

---

## 6. Official-style DAVIS AJ_RD results

AJ_RD is our re-entry diagnostic metric, not the standard TAPVid leaderboard metric.

| Method | true AJ_RD | true AJ_RD_256 | first re-entry proxy | Re-entry queries |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 0.3870 | 0.5546 | 0.4101 | 1,385 |
| CoTracker3 online | 0.4101 | 0.5972 | 0.4732 | 1,385 |
| TrackOn2 | 0.3700 | 0.5383 | 0.3681 | 1,385 |
| TAPNext | 0.3624 | 0.5199 | 0.3684 | 1,385 |
| ReEntry V1 | 0.4144 | 0.6072 | 0.4777 | 1,385 |
| ReEntry V22Q | 0.4136 | 0.6047 | 0.4770 | 1,385 |

Deltas vs CoTracker3 offline:

| Method | Δtrue AJ_RD | Δtrue AJ_RD_256 | Δfirst re-entry proxy |
|---|---:|---:|---:|
| ReEntry V1 | +0.0274 | +0.0526 | +0.0676 |
| ReEntry V22Q | +0.0266 | +0.0501 | +0.0669 |

Interpretation:

```text
The method still improves the intended re-entry recovery diagnostic under official-style DAVIS strided+original. However, this does not translate into an official leaderboard gain because standard TAPVid AJ/OA penalize false visibility changes over all strided queries and frames.
```

---

## 7. Critical conclusion

The current method should not be submitted or advertised as a leaderboard-improving method yet.

Current status:

```text
Official-style local DAVIS strided+original:
  AJ_RD improves.
  Official-style AJ/OA do not improve.
  δ_avg is unchanged because ReEntry changes visibility only, not coordinates.
```

Therefore:

```text
The current ReEntry pipeline is a strong re-entry diagnostic / visibility-lag correction method, but it is not yet a leaderboard-optimized TAPVid method.
```

This is not a failure. It clarifies the paper positioning:

```text
ReEntry addresses a specific failure mode that standard metrics only partially reward. The method improves AJ_RD and visibility recovery but can trade off global AJ/OA under official strided evaluation.
```

---

## 8. What can be claimed safely

Safe claims:

```text
1. Under official-style DAVIS strided+original evaluation, ReEntry improves AJ_RD over CoTracker3 offline.
2. Under local bridge/full50 evaluation, ReEntry strongly improves AJ_RD and OA.
3. ReEntry is not currently a universal TAPVid leaderboard improvement.
4. V22Q is safer than V1 for standard AJ/OA, but still below offline on official-style DAVIS.
```

Avoid:

```text
1. Official leaderboard result.
2. Official leaderboard improvement.
3. All TAPVid metrics improve.
4. V24/V22Q should be submitted as-is for leaderboard dominance.
```

---

## 9. Recommended next step

The next step should not be another DINOScore run.

The correct next step is to create a leaderboard-safe variant whose objective explicitly preserves standard TAPVid AJ/OA while retaining part of AJ_RD gain.

Recommended experiment:

```text
V25 Official-Safe ReEntry
Goal:
  Maintain official-style AJ/OA >= CoTracker3 offline on DAVIS strided+original,
  while improving AJ_RD as much as possible.

Core idea:
  Apply ReEntry only when a candidate recovery is expected to improve standard Jaccard, not only re-entry recovery.
```

Candidate approaches:

```text
1. Stricter V1 threshold sweep under official-style DAVIS:
   threshold = 0.20, 0.30, 0.40, 0.50, 0.60
   Expected: fewer visibility flips, less AJ/OA damage, smaller AJ_RD gain.

2. Gate on high-confidence windows only:
   require v1_prob >= tau and event_gate_prob >= tau_gate.

3. Short false-positive suppression:
   keep only recovery segments that persist >= k frames and have high core score.

4. Official metric-aware decision rule:
   approximate per-frame standard Jaccard risk and drop only when expected standard-Jaccard harm is low.
```

First concrete action:

```text
Run V1 threshold sweep on DAVIS strided+original:
  thresholds = 0.20, 0.30, 0.40, 0.50, 0.60
Evaluate each with:
  attempt0_rescore_cache.py --query-mode strided --metric-resolution-mode original
  eval_aj_rd_from_cache.py

Select the best trade-off point where:
  AJ is closest to offline or non-negative vs offline,
  OA is closest to offline or non-negative vs offline,
  AJ_RD remains positive vs offline.
```

If no threshold satisfies official AJ/OA preservation:

```text
The paper should position ReEntry as a failure-mode diagnostic and re-entry-specialized method, not as a leaderboard method.
```

---

## 10. Follow-up: V25 official-safe threshold sweep

Artifact:

```text
docs/official_protocol_v25_threshold_sweep_2026-07-04.md
```

A stricter V1 threshold sweep was run under DAVIS strided+original:

```text
thresholds = 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99
```

Best official-safe point:

```text
threshold = 0.80
```

At threshold 0.80:

```text
AJ    = 51.5648  vs offline 51.5385  => +0.0263 pp
OA    = 92.2106  vs offline 92.1543  => +0.0563 pp
δ_avg = 63.5892  vs offline 63.5892  => +0.0000 pp
AJ_RD = 0.3900   vs offline 0.3870   => +0.0030
```

Paired-video check:

```text
AJ mean = +0.0263 pp, CI [-0.0517, +0.1253], W/L/T = 12/13/5
OA mean = +0.0563 pp, CI [-0.1060, +0.2473], W/L/T = 14/10/6
```

Interpretation:

```text
Threshold 0.80 is an official-style safe operating point, not a strong leaderboard improvement. It preserves/very slightly improves official-style AJ/OA while retaining a tiny AJ_RD gain.
```

Updated next direction:

```text
V26 should be official-metric-aware selective ReEntry: train or derive a frame-level rule that only fires when standard-Jaccard harm is unlikely.
```
