# Final Audit and Next Steps — 2026-07-04

## 1. Audit scope

This audit verifies the current paper evidence package after completing:

```text
1. TAPVid RGB-Stacking full50 standard/full evaluation.
2. TAPVid-DAVIS bridge evaluation.
3. RGB-Stacking fresh20-49 re-entry/stress evaluations.
4. V24-DINOScore full50 chunk generation and remapped fresh20-49 reuse.
```

---

## 2. Artifact existence check

Verified key artifacts exist:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/rgb_full50_base_compare.json
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v1_thr010/reentry_v1_rgb_full50_thr010.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v22Q/reentry_v22Q_rgb_full50.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_dinoscore/reentry_v24_dinoscore_rgb_full50.pt
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_dinoscore/manifest.json
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/paired_stats_rgb_full50_v24_vs_all.json
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/rgb_full50_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_chunks_final_nonoverlap/
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/paired_stats_davis_v24_vs_all.json
outputs/paper_discovery_2026-06-27/standard_benchmark_davis/v24_dinoscore_offline_to_baseline/manifest.json
```

---

## 3. RGB-Stacking full50 chunk integrity

Frame-keep rows:

```text
N = 609,090
```

Final V24 non-overlap chunk directory:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_chunks_final_nonoverlap
```

Verified:

```text
n_chunks = 30
scanned rows = 609,090
selected visible rows = 574,871
fires = 6,403
coverage = 0 -- 609,090
gaps = none
overlaps = none
bad chunks = none
```

Important: do not use the earlier working directory below for final apply because it contains timeout-overlap remnants:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_chunks_remap_clean
```

Use only:

```text
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/v24_chunks_final_nonoverlap
```

---

## 4. Remap validity check

The full50 V24 computation reused the already-computed fresh20-49 natural chunks for videos 20--49.

This reuse was verified as valid:

```text
full50 fresh20-49 row offset = 245,461
full50.meta_json[245461:] == fresh20_49.meta_json: True
full50.X[245461:] == fresh20_49.X: True
full50 labels/weights/utility slice == fresh20_49 labels/weights/utility: True
V22Q full50 records 20--49 == V22Q fresh20-49 records: True
```

Therefore the remapped fresh20-49 DINOScore chunks are valid for full50.

---

## 5. RGB-Stacking full50 results

Scope:

```text
Dataset: TAPVid RGB-Stacking
Split: rgb_stacking_000000 -- rgb_stacking_000049
Videos: 50
Queries: 60,829
Re-entry queries: 10,585
Base: CoTracker3 offline
Override: CoTracker3 online
```

Query-weighted results:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline Base | 0.3617 | 79.9345 | 91.6371 |
| online | 0.4028 | 44.8432 | 55.8092 |
| V1 | 0.4414 | 79.4302 | 93.0758 |
| V22Q | 0.4400 | 79.5756 | 93.0481 |
| V24-DINOScore | 0.4394 | 79.5974 | 93.0389 |

Query-weighted deltas vs offline Base:

| Method | ΔAJ_RD | ΔAJ | ΔOA |
|---|---:|---:|---:|
| V1 - Base | +0.0797 | -0.5043 | +1.4387 |
| V22Q - Base | +0.0783 | -0.3589 | +1.4110 |
| V24 - Base | +0.0777 | -0.3371 | +1.4018 |

Video-weighted paired means:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline | 0.318334 | 79.934460 | 91.637145 |
| online | 0.344838 | 44.843239 | 55.809194 |
| V1 | 0.394862 | 79.430179 | 93.075843 |
| V22Q | 0.393572 | 79.575607 | 93.048122 |
| V24 | 0.393184 | 79.597351 | 93.038938 |

Key paired comparisons:

```text
V24 - offline:
  ΔAJ_RD = +0.074850, CI [0.060516, 0.090088], wins/losses/ties = 48/0/2
  ΔAJ    = -0.337109, CI [-0.624740, -0.048121], wins/losses/ties = 11/39/0
  ΔOA    = +1.401793, CI [0.940280, 1.873843], wins/losses/ties = 40/10/0

V24 - V22Q:
  ΔAJ_RD = -0.000388, CI [-0.000726, -0.000108]
  ΔAJ    = +0.021744, CI [0.009674, 0.038240]
  ΔOA    = -0.009184, CI [-0.028577, 0.005473]
```

Conclusion:

```text
RGB-Stacking full50 gives the strongest full-standard evidence:
ReEntry substantially improves AJ_RD and OA over the offline base, while standard AJ decreases slightly.
```

---

## 6. DAVIS bridge results

Scope:

```text
Dataset/protocol: TAPVid-DAVIS first-input bridge
Videos: 30
Queries: 650
Re-entry queries: 259
Base for ReEntry: CoTracker3 offline
Override for ReEntry: CoTracker3 baseline
External comparison: TrackOn2 DINOv3
```

Video-weighted paired means:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| offline | 0.409808 | 62.166050 | 89.010664 |
| baseline | 0.479548 | 62.818065 | 90.263068 |
| TrackOn2 | 0.497728 | 67.040639 | 93.061465 |
| V1 | 0.480024 | 64.095014 | 92.347688 |
| V22Q | 0.476936 | 64.238349 | 92.338242 |
| V24 | 0.473188 | 64.354510 | 92.471189 |

Key paired comparisons:

```text
V24 - offline:
  ΔAJ_RD = +0.063380, CI [0.038576, 0.090013]
  ΔAJ    = +2.188460, CI [1.248721, 3.242748]
  ΔOA    = +3.460525, CI [1.937991, 5.361877]

V24 - baseline:
  ΔAJ_RD = -0.006360, CI [-0.020940, 0.008208]
  ΔAJ    = +1.536446, CI [0.447824, 2.572093]
  ΔOA    = +2.208121, CI [1.325593, 3.152698]

V24 - V22Q:
  ΔAJ_RD = -0.003748, CI [-0.009124, -0.000288]
  ΔAJ    = +0.116161, CI [0.021575, 0.237639]
  ΔOA    = +0.132948, CI [0.003486, 0.292907]
```

Conclusion:

```text
DAVIS bridge confirms that ReEntry is not only a fresh20-49 artifact.
However, TrackOn2 remains stronger overall on DAVIS, so the paper must not claim V24 is the best DAVIS method.
```

---

## 7. Fresh20-49 / stress evidence

Current remembered final comparison vs Base on fresh20-49 video-weighted:

```text
natural:
  AJ_RD +0.0698
  OA    +1.4608
  AJ    -0.3060

translate_L16:
  AJ_RD +0.0484
  OA    +1.4146
  AJ    -0.1953

occluder_L16:
  AJ_RD +0.0345
  OA    +1.3375
  AJ    -0.4004

Average:
  AJ_RD +0.0509
  OA    +1.4043
  AJ    -0.3006
```

Conclusion:

```text
Fresh/stress results are consistent with full50: strong AJ_RD/OA gains with small AJ trade-off.
```

---

## 8. Final method positioning

The correct final positioning is now very stable:

```text
V1 Learned ReEntry-VisCalibrator:
  Main AJ_RD-oriented method.
  Best AJ_RD among ReEntry variants on RGB full50 and DAVIS.

V22Q Interval/Gate:
  Stability/interval extension.
  Recovers some standard AJ relative to V1 while keeping most AJ_RD/OA gain.

V24-DINOScore 0.395:
  Optional AJ-oriented DINOv3 appearance micro-filter.
  Gives small AJ gain over V22Q but slightly lowers AJ_RD and often slightly lowers/neutralizes OA.
```

Do not replace V1/V22Q with V24 as the default method.

---

## 9. Safe paper claims

Strong claims supported:

```text
1. On TAPVid RGB-Stacking full50, ReEntry improves AJ_RD by about +0.078 and OA by about +1.40 over the CoTracker3 offline base.
2. On DAVIS bridge, ReEntry improves over CoTracker3 offline with +0.063 AJ_RD, +2.19 AJ, and +3.46 OA for V24.
3. On fresh20-49 and stress variants, the same AJ_RD/OA improvement pattern appears.
4. V1 is the AJ_RD-oriented main method; V24 is an optional AJ-oriented micro-filter.
```

Claims to avoid:

```text
1. All metrics improve.
2. V24 is universally best.
3. V24 beats TrackOn2 on DAVIS.
4. The method is a public official leaderboard result.
5. DINOScore should replace V1/V22Q by default.
```

---

## 10. Next steps

Recommended immediate next step:

```text
Create docs/final_paper_tables_2026-07-04.md
```

It should include exactly these paper-ready tables:

```text
Table 1. TAPVid RGB-Stacking full50 standard/full benchmark.
Table 2. TAPVid-DAVIS bridge benchmark.
Table 3. RGB-Stacking fresh20-49 + stress split results.
Table 4. V24 vs V22Q paired-video statistics.
Table 5. V24 DINOScore decision statistics.
Table 6. DINOv3 diagnostic evidence.
Table 7. Negative/diagnostic results.
```

Then write the paper result narrative:

```text
docs/paper_results_section_draft_2026-07-04.md
```

The narrative should be built around this structure:

```text
1. Standard/full validation on RGB-Stacking full50.
2. Cross-dataset validation on DAVIS bridge.
3. Re-entry-focused fresh/stress validation.
4. Ablation/positioning: V1 vs V22Q vs V24.
5. Limitations and honest trade-off: AJ_RD/OA up, AJ slightly down.
```
