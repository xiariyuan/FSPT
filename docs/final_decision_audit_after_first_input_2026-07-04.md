# Final Decision Audit after DAVIS First/Input Supplement — 2026-07-04

## 1. What was checked

This audit was created after adding the DAVIS first/input official-style local evaluation.

Checked artifacts:

```text
docs/strict_official_protocol_completion_plan_2026-07-04.md
docs/official_protocol_davis_first_input_comparison_2026-07-04.md
docs/evaluation_protocol_section_2026-07-04.md
docs/official_protocol_audit_2026-07-04.md
docs/official_protocol_v25_threshold_sweep_2026-07-04.md
docs/standard_benchmark_rgb_full50_comparison_2026-07-04.md
docs/standard_benchmark_davis_comparison_2026-07-04.md
docs/paper_results_summary_2026-07-04.md
docs/paper_tables_plan_2026-07-04.md
```

All required docs exist.

Checked result roots:

```text
outputs/paper_discovery_2026-06-27/official_protocol_davis_first_input/
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/
outputs/paper_discovery_2026-06-27/standard_benchmark_rgb_full50/
```

Main result JSONs exist for:

```text
DAVIS first/input: baseline, offline, trackon2, V1, V22Q, V24
DAVIS strided/original: offline, V1, V22Q, V25 threshold=0.80
RGB full50: offline, V1, V22Q, V24 documented and paired stats documented
```

---

## 2. DAVIS first/input official-style local evaluation

Protocol:

```text
Dataset: TAPVid-DAVIS
Videos: 30
Queries: 650
Query mode: first
Metric resolution: input / 256
Evaluation type: official-style local evaluation
```

Main table:

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 baseline | 64.8949 | 91.7983 | 77.3560 | 84.9284 | 0.3486 | 0.5246 |
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 84.5776 | 0.3142 | 0.4525 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 87.8233 | 0.3714 | 0.5444 |
| ReEntry V1 | 64.5758 | 91.7274 | 77.2244 | 84.5776 | 0.3588 | 0.5305 |
| ReEntry V22Q | 64.7260 | 91.7406 | 77.2244 | 84.5776 | 0.3556 | 0.5252 |
| ReEntry V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 84.5776 | 0.3549 | 0.5236 |

Deltas vs CoTracker3 offline:

```text
V1:
  ΔAJ    = +1.9193
  ΔOA    = +3.5787
  Δδ_avg = +0.0000
  ΔAJ_RD = +0.0446

V22Q:
  ΔAJ    = +2.0695
  ΔOA    = +3.5919
  Δδ_avg = +0.0000
  ΔAJ_RD = +0.0414

V24:
  ΔAJ    = +2.1874
  ΔOA    = +3.7363
  Δδ_avg = +0.0000
  ΔAJ_RD = +0.0407
```

Paired-video check vs offline:

```text
V1 - offline:
  AJ mean +1.9193 pp, CI [+0.8375, +3.0545], W/L/T 22/7/1
  OA mean +3.5787 pp, CI [+1.8485, +5.6697], W/L/T 24/5/1

V22Q - offline:
  AJ mean +2.0695 pp, CI [+1.0763, +3.1667], W/L/T 23/6/1
  OA mean +3.5919 pp, CI [+2.0011, +5.5624], W/L/T 25/4/1

V24 - offline:
  AJ mean +2.1874 pp, CI [+1.2582, +3.2487], W/L/T 24/5/1
  OA mean +3.7363 pp, CI [+2.1673, +5.7423], W/L/T 27/2/1
```

Decision:

```text
This is the strongest DAVIS official-style result for the current ReEntry pipeline.
Under first/input evaluation, ReEntry improves both standard metrics and AJ_RD over the offline base.
V24 is the best ReEntry variant in this specific first/input setting, but TrackOn2 remains stronger overall.
```

---

## 3. DAVIS strided/original official-style local evaluation

Protocol:

```text
Dataset: TAPVid-DAVIS
Videos: 30
Query mode: strided
Metric resolution: original
Evaluation type: official-style local evaluation
```

Main table:

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 68.0060 | 0.3870 | 0.5546 |
| ReEntry V1 default | 50.7303 | 91.2730 | 63.5892 | 68.0060 | 0.4144 | 0.6072 |
| ReEntry V22Q default | 50.8342 | 91.3439 | 63.5892 | 68.0060 | 0.4136 | 0.6047 |
| V25 threshold=0.80 | 51.5648 | 92.2106 | 63.5892 | 68.0060 | 0.3900 | 0.5601 |

Decision:

```text
Default V1/V22Q improve AJ_RD but hurt official-style AJ/OA under strided/original.
V25 threshold=0.80 is an official-safe operating point: small positive AJ/OA and tiny positive AJ_RD.
This should be reported as a conservative safety point, not a strong leaderboard improvement.
```

---

## 4. RGB-Stacking full50 local standard evaluation

Scope:

```text
Dataset: TAPVid RGB-Stacking
Videos: 50, rgb_stacking_000000 -- rgb_stacking_000049
Queries: 60,829
Evaluation type: full50 local standard evaluation
```

Main table:

| Method | AJ_RD | AJ | OA |
|---|---:|---:|---:|
| CoTracker3 offline | 0.3617 | 79.9345 | 91.6371 |
| ReEntry V1 | 0.4414 | 79.4302 | 93.0758 |
| ReEntry V22Q | 0.4400 | 79.5756 | 93.0481 |
| ReEntry V24-DINOScore | 0.4394 | 79.5974 | 93.0389 |

Decision:

```text
This is the strongest full-data local standard result.
It supports the main ReEntry claim: large AJ_RD and OA improvement with a small AJ trade-off.
It should not be called strict official query-protocol evaluation until RGB-Stacking is explicitly audited/rerun under first or strided official query mode.
```

---

## 5. Consistency and risk audit

No core numerical conflict was found among the newly generated DAVIS first/input JSONs, DAVIS strided/original JSONs, V25 sweep JSONs, and RGB full50 docs.

High-risk claims are correctly avoided or placed in avoid sections:

```text
Avoid claiming:
  Do not claim: official leaderboard submission
  strict full TAP-Vid benchmark result
  Do not claim: universal improvement across all official protocols
  Do not claim: V24 beats TrackOn2 on DAVIS
  Do not claim: V24 is universally best
  DINOScore should replace V1/V22Q by default
```

Safe claims:

```text
1. ReEntry improves DAVIS first/input official-style AJ/OA/AJ_RD over CoTracker3 offline.
2. ReEntry improves RGB-Stacking full50 AJ_RD/OA over CoTracker3 offline with a small AJ trade-off.
3. Under DAVIS strided/original, default ReEntry improves AJ_RD but needs conservative thresholding to preserve AJ/OA.
4. V25 threshold=0.80 is an official-style safe operating point, not a strong leaderboard result.
```

---

## 6. Final recommendation

The project now has enough evidence to move from experiment accumulation to paper assembly.

Recommended next action:

```text
Create final paper tables and Results section now.
```

Priority order:

```text
1. Finalize paper-ready tables.
2. Write Results section draft.
3. Then optionally run RGB-Stacking official query-protocol audit/rerun.
4. Do not start Kinetics/Kubric before the paper tables are locked.
```

Reason:

```text
DAVIS first/input now supplies the missing official-style positive result.
DAVIS strided/original supplies the rigorous caution and V25 safety result.
RGB full50 supplies the strongest full local standard evidence.
Together, these are sufficient for a coherent and honest paper narrative.
```

---

## 7. Next document to create

Create:

```text
docs/final_paper_tables_2026-07-04.md
```

Include tables:

```text
Table 1. Evaluation protocol and claim level
Table 2. TAPVid-DAVIS first/input official-style local evaluation
Table 3. TAPVid-DAVIS strided/original official-style local evaluation
Table 4. TAPVid RGB-Stacking full50 local standard evaluation
Table 5. RGB fresh20-49 and stress diagnostic evaluations
Table 6. V25 official-safe threshold sweep
Table 7. Method positioning / ablation summary
```

Then create:

```text
docs/paper_results_section_draft_2026-07-04.md
```

Main result narrative:

```text
ReEntry is not a universal leaderboard booster. It is a re-entry failure-mode method that improves recovery after occlusion/disappearance. It gives strong gains in first-query DAVIS and RGB-Stacking full50, while stricter strided/original evaluation requires conservative selection to preserve global metrics.
```
