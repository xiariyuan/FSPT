# Final Protocol Review and Next Steps — 2026-07-04

## 1. Final reviewed status

After checking all generated artifacts and result JSONs, the project now has three protocol-safe result blocks:

```text
1. TAPVid-DAVIS first/input official-style local evaluation
2. TAPVid-DAVIS strided/original official-style local evaluation
3. TAPVid RGB-Stacking full50 strided official-style local evaluation
```

It also has diagnostic re-entry/stress evaluations:

```text
1. RGB-Stacking fresh20-49 natural
2. translate_L16 stress
3. occluder_L16 stress
```

Important correction:

```text
RGB-Stacking full50 is not merely a local full50 summary anymore. Existing artifacts confirm a stricter strided official-style local evaluation with AJ/OA/δ_avg/δ_4px and paired-video statistics.
```

Still not true:

```text
official leaderboard submission
full TAP-Vid benchmark submission
Kinetics/Kubric full benchmark result
```

---

## 2. Key results after review

### 2.1 DAVIS first/input official-style

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 0.3142 |
| ReEntry V1 | 64.5758 | 91.7274 | 77.2244 | 0.3588 |
| ReEntry V22Q | 64.7260 | 91.7406 | 77.2244 | 0.3556 |
| ReEntry V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 0.3549 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 0.3714 |

Conclusion:

```text
On DAVIS first/input, ReEntry improves AJ, OA, and AJ_RD over CoTracker3 offline. V24 is the strongest ReEntry variant here, but TrackOn2 remains stronger overall.
```

### 2.2 DAVIS strided/original official-style

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 0.3870 |
| ReEntry V1 default | 50.7303 | 91.2730 | 63.5892 | 0.4144 |
| ReEntry V22Q default | 50.8342 | 91.3439 | 63.5892 | 0.4136 |
| V25-safe threshold=0.80 | 51.5648 | 92.2106 | 63.5892 | 0.3900 |

Conclusion:

```text
Default ReEntry improves AJ_RD but hurts strided/original AJ/OA. V25 threshold=0.80 is an official-safe operating point with tiny positive AJ/OA and AJ_RD gains.
```

### 2.3 RGB-Stacking full50 strided official-style

| Method | AJ | OA | δ_avg | δ_4px | AJ_RD |
|---|---:|---:|---:|---:|---:|
| CoTracker3 offline | 79.9345 | 91.6371 | 88.5714 | 91.5875 | 0.3617 |
| ReEntry V1 | 79.4302 | 93.0758 | 88.5714 | 91.5875 | 0.4414 |
| ReEntry V22Q | 79.5756 | 93.0481 | 88.5714 | 91.5875 | 0.4400 |
| ReEntry V24-DINOScore | 79.5974 | 93.0389 | 88.5714 | 91.5875 | 0.4394 |

Conclusion:

```text
RGB full50 confirms the main trade-off: ReEntry substantially improves AJ_RD and OA, keeps δ_avg unchanged, and incurs a small AJ reduction.
```

---

## 3. What the paper can now claim

Safe main claim:

```text
ReEntry improves re-entry recovery across standard TAP-Vid official-style local evaluations and diagnostic stress settings. It improves DAVIS first-query AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter strided evaluations reveal a trade-off between re-entry recovery, occlusion accuracy, and standard AJ.
```

More detailed claim:

```text
On DAVIS first/input, ReEntry V24 improves over CoTracker3 offline by +2.19 AJ, +3.74 OA, and +0.0407 AJ_RD. On RGB-Stacking full50 strided, ReEntry improves AJ_RD by +0.0777 to +0.0797 and OA by about +1.40 to +1.44 points, at the cost of a small AJ reduction. On DAVIS strided/original, default ReEntry improves AJ_RD but requires conservative thresholding to preserve standard AJ/OA.
```

Avoid:

```text
official leaderboard submission
full TAP-Vid benchmark result
universal improvement across all protocols
V24 beats TrackOn2 on DAVIS
```

---

## 4. Remaining gaps

### Gap A — No official server submission

No verified TAP-Vid online submission server / leaderboard return score exists in our workflow.

Label all results as:

```text
official-style local evaluation
```

### Gap B — No Kinetics/Kubric full benchmark

Kinetics/Kubric full prediction caches were not found earlier. Running them would require generating tracker predictions first and is likely expensive.

### Gap C — No V26 official-metric-aware selector yet

V25 threshold sweep showed that thresholding alone can preserve AJ/OA only by nearly eliminating AJ_RD gains. A stronger leaderboard-style method would require a new metric-aware selector.

---

## 5. Recommended next step

The next step should be paper consolidation, not more blind experiments.

### Step 1 — Create final paper tables

Create:

```text
docs/final_paper_tables_2026-07-04.md
```

Include:

```text
Table 1: DAVIS first/input official-style
Table 2: DAVIS strided/original official-style
Table 3: RGB-Stacking full50 strided official-style
Table 4: RGB fresh/stress diagnostic
Table 5: V25 threshold sweep
Table 6: Method positioning / ablation summary
```

### Step 2 — Write Results section draft

Create:

```text
docs/paper_results_section_draft_2026-07-04.md
```

Structure:

```text
1. Evaluation protocol
2. DAVIS first-query result
3. RGB-Stacking full50 result
4. Strided/original protocol stress test
5. Diagnostic re-entry/stress analysis
6. Ablation: V1 vs V22Q vs V24 vs V25
7. Limitations and leaderboard-safe discussion
```

### Step 3 — Only after paper tables are fixed, start V26

V26 goal:

```text
official-metric-aware selective ReEntry
```

Success criterion:

```text
AJ/OA >= offline and AJ_RD improvement meaningfully larger than V25 threshold=0.80.
```

A practical target:

```text
AJ/OA non-negative vs offline
AJ_RD gain >= +0.01 on DAVIS strided/original
```

---

## 6. Final decision

Recommended project decision:

```text
Freeze current results for paper drafting.
Do not chase more experiments until final tables and Results text are written.
Then open V26 as a separate improvement branch if leaderboard-style optimization remains the target.
```

Rationale:

```text
The current evidence is already sufficient for a careful re-entry-focused paper. Additional experiments may improve the method, but they should not block the paper consolidation step.
```
