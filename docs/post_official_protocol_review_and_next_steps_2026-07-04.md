# Post Official-Protocol Review and Next Steps — 2026-07-04

## 1. Review purpose

This document performs a decision-level review after completing the official-style protocol补位.

Question:

```text
Have we now done enough strict official-style evaluation to anchor the paper, and what should be done next?
```

Short answer:

```text
Yes, the paper can now be anchored by official-style local evaluations on TAPVid-DAVIS and TAPVid RGB-Stacking. The next step should be final paper tables + Results draft, not more exploratory tuning.
```

---

## 2. Artifact existence review

Verified existing documents:

```text
docs/strict_official_protocol_completion_plan_2026-07-04.md
docs/official_protocol_davis_first_input_comparison_2026-07-04.md
docs/official_protocol_audit_2026-07-04.md
docs/official_protocol_v25_threshold_sweep_2026-07-04.md
docs/official_protocol_rgb_full50_strided_comparison_2026-07-04.md
docs/evaluation_protocol_section_2026-07-04.md
docs/strict_official_protocol_status_2026-07-04.md
```

Verified key JSON outputs:

```text
outputs/paper_discovery_2026-06-27/official_protocol_davis_first_input/paired_first_input_vs_all.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/rescore_original/offline_strided_original_rescore.json
outputs/paper_discovery_2026-06-27/official_protocol_davis_strided_original/v25_threshold_sweep/v1_thr080/reentry_v1_davis_strided_original_thr080_rescore_original.json
outputs/paper_discovery_2026-06-27/official_protocol_rgb_full50_strided/paired_rgb_full50_strided_vs_all.json
```

Status:

```text
All key official-style artifacts are present.
```

---

## 3. Recomputed high-level official-style numbers

### 3.1 TAPVid-DAVIS first/input

| Method | AJ | OA | delta_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 0.3142 |
| ReEntry V1 | 64.5758 | 91.7274 | 77.2244 | 0.3588 |
| ReEntry V22Q | 64.7260 | 91.7406 | 77.2244 | 0.3556 |
| ReEntry V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 0.3549 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 0.3714 |

Interpretation:

```text
This is the cleanest positive official-style DAVIS result.
ReEntry improves AJ/OA/AJ_RD vs CoTracker3 offline.
TrackOn2 remains the strongest external baseline overall.
```

Best ReEntry row for standard metrics:

```text
V24-DINOScore:
  dAJ    = +2.1874
  dOA    = +3.7363
  dAJ_RD = +0.0407
```

---

### 3.2 TAPVid-DAVIS strided/original

| Method | AJ | OA | delta_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 0.3870 |
| ReEntry V1 default | 50.7303 | 91.2730 | 63.5892 | 0.4144 |
| ReEntry V22Q default | 50.8342 | 91.3439 | 63.5892 | 0.4136 |
| V25-safe threshold=0.80 | 51.5648 | 92.2106 | 63.5892 | 0.3900 |

Interpretation:

```text
Default ReEntry improves AJ_RD but hurts standard AJ/OA under stricter strided/original DAVIS.
V25 threshold=0.80 is an official-safe operating point, but the gain is tiny.
```

Safe wording:

```text
Under strided/original DAVIS, ReEntry requires conservative thresholding to preserve official-style AJ/OA.
```

---

### 3.3 TAPVid RGB-Stacking full50 strided/input

| Method | AJ | OA | delta_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 79.9345 | 91.6371 | 88.5714 | 0.3617 |
| ReEntry V1 | 79.4302 | 93.0758 | 88.5714 | 0.4414 |
| ReEntry V22Q | 79.5756 | 93.0481 | 88.5714 | 0.4400 |
| ReEntry V24-DINOScore | 79.5974 | 93.0389 | 88.5714 | 0.4394 |

Interpretation:

```text
On the full 50-video RGB-Stacking subset under an audited strided query pattern, ReEntry improves OA and AJ_RD substantially, with a small consistent AJ trade-off.
```

Best overall paper row depends on emphasis:

```text
V1: best AJ_RD and OA.
V24: best AJ among ReEntry variants, with tiny OA/AJ_RD cost.
V22Q: middle stable variant.
```

---

## 4. Consistency and risk review

### 4.1 Can we say “strict official protocol”?

Safe phrase:

```text
official-style local evaluation using the official TAP-Vid metric formulation and audited query protocols
```

Potentially safe but should be used carefully:

```text
strict official-protocol local evaluation
```

Only use this for:

```text
DAVIS first/input
DAVIS strided/original
RGB-Stacking full50 strided/input after query-frame audit
```

Do not use for:

```text
fresh20-49
translate_L16
occluder_L16
bridge-only diagnostic claims
```

### 4.2 Can we say “official leaderboard result”?

No.

Reason:

```text
No official server/leaderboard submission and returned score has been confirmed.
```

### 4.3 Is the method universally better?

No.

Correct nuance:

```text
ReEntry is strongly beneficial for re-entry recovery and visibility correction.
It improves DAVIS first/input and RGB full50 OA/AJ_RD.
But it has AJ trade-offs and requires conservative operation under DAVIS strided/original.
```

### 4.4 Should V24 be the main method?

Not universally.

Recommended positioning:

```text
V1 = AJ_RD-oriented main recovery method.
V22Q = stable interval/post-processing variant.
V24-DINOScore = optional appearance micro-filter, best AJ among ReEntry variants in several settings.
V25-safe = conservative official-safe threshold point for DAVIS strided/original.
```

---

## 5. Decision: what to do next

### Recommended next step: write final paper tables and Results draft

Do next:

```text
1. docs/final_paper_tables_2026-07-04.md
2. docs/paper_results_section_draft_2026-07-04.md
```

Reason:

```text
The protocol foundation is now strong enough. More experiments can continue later, but the current result package should be frozen into paper tables before more tuning creates confusion.
```

### Do not do next

Do not immediately start Kinetics/Kubric full runs unless the goal changes to broad benchmark competition.

Reason:

```text
They are high-cost and not necessary for the current ReEntry failure-mode paper.
```

Do not keep sweeping thresholds blindly.

Reason:

```text
V25 already identifies the trade-off curve. A real improvement now requires metric-aware selection, not more threshold search.
```

---

## 6. Final recommended paper structure for results

### Table 1: Official-style DAVIS first/input

Rows:

```text
CoTracker3 offline
CoTracker3 baseline
TrackOn2
ReEntry V1
ReEntry V22Q
ReEntry V24-DINOScore
```

Main message:

```text
ReEntry improves AJ/OA/AJ_RD vs CoTracker3 offline; TrackOn2 remains strongest external baseline.
```

### Table 2: Official-style DAVIS strided/original

Rows:

```text
CoTracker3 offline
ReEntry V1 default
ReEntry V22Q default
V25-safe threshold=0.80
```

Main message:

```text
Default ReEntry improves AJ_RD but hurts AJ/OA; conservative threshold preserves official-style AJ/OA with tiny AJ_RD gain.
```

### Table 3: Official-style RGB-Stacking full50 strided/input

Rows:

```text
CoTracker3 offline
ReEntry V1
ReEntry V22Q
ReEntry V24-DINOScore
```

Main message:

```text
Full 50-video RGB-Stacking result: strong AJ_RD/OA gains with small AJ trade-off.
```

### Table 4: Diagnostic fresh/stress evaluations

Rows:

```text
natural fresh20-49
translate_L16
occluder_L16
```

Main message:

```text
ReEntry behavior is robust under synthetic re-entry stress variants.
```

### Table 5: Ablation / variant positioning

Rows:

```text
V1
V22Q
V24
V25-safe
```

Main message:

```text
Different variants occupy different points on the recovery-vs-official-metric trade-off curve.
```

---

## 7. Final one-sentence project status

```text
The project now has protocol-safe official-style local evaluations on TAPVid-DAVIS first/input, TAPVid-DAVIS strided/original, and TAPVid RGB-Stacking full50 strided/input. The next step should be final paper tables and Results writing, while reserving V26 metric-aware selection for future leaderboard-oriented improvement.
```
