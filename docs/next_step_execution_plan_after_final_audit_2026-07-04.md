# Next Step Execution Plan after Final Audit — 2026-07-04

## 1. Final audit verdict

The final writing package is internally consistent and ready for paper assembly.

Validated files:

```text
docs/final_paper_tables_2026-07-04.md
docs/paper_results_section_draft_2026-07-04.md
docs/paper_experimental_setup_section_draft_2026-07-04.md
docs/paper_limitations_and_future_work_draft_2026-07-04.md
docs/paper_combined_draft_core_sections_2026-07-04.md
docs/evaluation_protocol_section_2026-07-04.md
docs/final_paper_writing_package_index_2026-07-04.md
docs/final_decision_audit_after_first_input_2026-07-04.md
```

Key values were checked against JSON-backed results:

```text
DAVIS first/input V24:
  AJ    = 64.8439
  OA    = 91.8851
  AJ_RD = 0.3549

DAVIS strided/original V25 threshold=0.80:
  AJ    = 51.5648
  OA    = 92.2106
  AJ_RD = 0.3900
```

Required table coverage exists:

```text
Table 1. Evaluation protocol and claim level
Table 2. TAPVid-DAVIS first/input official-style local evaluation
Table 3. TAPVid-DAVIS strided/original official-style local evaluation
Table 4. TAPVid RGB-Stacking full50 local standard evaluation
Table 5. RGB fresh20-49 and stress diagnostic evaluations
Table 6. V25 official-safe threshold sweep
Table 7. Method positioning / ablation summary
```

Risky phrases such as `official leaderboard submission`, `universal improvement`, and `V24 beats TrackOn2` are present only in avoid / claim-boundary contexts.

---

## 2. Decision: do not run more large experiments now

Current evidence is sufficient for paper assembly.

Do not start Kinetics/Kubric now.
Do not start another broad DINOScore sweep now.
Do not continue threshold-only tuning now.

Reason:

```text
The paper now has:
1. A positive official-style DAVIS first/input result.
2. A rigorous DAVIS strided/original cautionary result plus V25 safety point.
3. A strong RGB-Stacking full50 local standard result.
4. Re-entry/stress diagnostics that support the failure-mode claim.
```

The highest return is no longer another experiment. The highest return is converting the existing result package into a complete paper draft.

---

## 3. Immediate next task: assemble full paper draft v0

Create:

```text
docs/paper_full_draft_v0_2026-07-04.md
```

Use these sources:

```text
Abstract / contributions:
  docs/paper_combined_draft_core_sections_2026-07-04.md

Experimental setup:
  docs/paper_experimental_setup_section_draft_2026-07-04.md

Results:
  docs/paper_results_section_draft_2026-07-04.md

Tables:
  docs/final_paper_tables_2026-07-04.md

Limitations:
  docs/paper_limitations_and_future_work_draft_2026-07-04.md
```

Target structure:

```text
Title
Abstract
1. Introduction
2. Related Work
3. Method
4. Experimental Setup
5. Results
6. Limitations and Future Work
7. Conclusion
Appendix A. Additional tables / paired stats
```

---

## 4. Missing writing component: Method section

The main missing paper component is not another result table. It is the Method section.

Need to write:

```text
3.1 Problem setup
3.2 Re-entry failure mode
3.3 ReEntry visibility correction pipeline
3.4 V1 Learned ReEntry-VisCalibrator
3.5 V22Q Interval/Gate post-processing
3.6 V24-DINOScore appearance micro-filter
3.7 V25 official-safe thresholding
3.8 Why current ReEntry changes visibility but not coordinates
```

Important method positioning:

```text
V1 = AJ_RD-oriented main recovery method.
V22Q = stability / interval extension.
V24 = optional AJ-oriented appearance micro-filter.
V25 = conservative official-style safety point.
```

Avoid:

```text
V24 is the final/default universal method.
V25 is a strong leaderboard method.
ReEntry fixes localization errors.
```

---

## 5. Figures to prepare next

Recommended figures:

```text
Figure 1. Re-entry failure-mode diagram
  Show point visible -> occluded/disappears -> reappears -> base tracker visibility lag -> ReEntry correction.

Figure 2. ReEntry pipeline
  Base tracker predictions -> candidate re-entry windows -> V1 confidence -> V22Q interval/gate -> optional V24 appearance filter -> corrected visibility.

Figure 3. Main result bar chart
  DAVIS first/input: offline vs V1/V22Q/V24 on AJ/OA/AJ_RD.

Figure 4. RGB full50 trade-off
  AJ_RD/OA gain with AJ trade-off.

Figure 5. V25 threshold sweep
  threshold vs AJ/OA/AJ_RD, showing 0.80 as official-safe point.
```

---

## 6. Paper claim to keep fixed

Use this as the central claim:

```text
ReEntry improves re-entry recovery across TAP-Vid local evaluations and diagnostic settings. It improves DAVIS first/input official-style AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter DAVIS strided/original evaluation requires conservative selection to preserve standard metrics.
```

---

## 7. Optional experiments after full draft v0

Only after full draft v0 exists, consider:

```text
1. RGB-Stacking full50 strict official first/strided query-protocol audit or rerun.
2. V26 metric-aware selective ReEntry.
3. Kinetics/Kubric only if paper target requires broader benchmark coverage.
```

These are optional next-stage improvements, not blockers for the current paper draft.

---

## 8. Immediate execution order

```text
Step 1. Create paper_full_draft_v0_2026-07-04.md.
Step 2. Write Method section in full.
Step 3. Insert final tables and table captions.
Step 4. Insert Results section from existing draft.
Step 5. Add Limitations and Future Work.
Step 6. Create figure plan / captions.
Step 7. Run final consistency audit on the complete draft.
```

This is the next step.
