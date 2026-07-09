# Final Paper Writing Package Index — 2026-07-04

This index lists the final writing artifacts created after the DAVIS first/input supplement and final decision audit.

## Core writing files

```text
docs/final_paper_tables_2026-07-04.md
docs/paper_results_section_draft_2026-07-04.md
docs/paper_experimental_setup_section_draft_2026-07-04.md
docs/paper_limitations_and_future_work_draft_2026-07-04.md
docs/evaluation_protocol_section_2026-07-04.md
```

## Supporting audit files

```text
docs/final_decision_audit_after_first_input_2026-07-04.md
docs/strict_official_protocol_completion_plan_2026-07-04.md
docs/official_protocol_davis_first_input_comparison_2026-07-04.md
docs/official_protocol_audit_2026-07-04.md
docs/official_protocol_v25_threshold_sweep_2026-07-04.md
docs/standard_benchmark_rgb_full50_comparison_2026-07-04.md
docs/standard_benchmark_rgb_full50_paired_stats_2026-07-04.md
docs/standard_benchmark_davis_comparison_2026-07-04.md
docs/standard_benchmark_davis_paired_stats_2026-07-04.md
```

## Recommended paper structure using these files

```text
1. Introduction
2. Related Work
3. Method
4. Experimental Setup
   Source: docs/paper_experimental_setup_section_draft_2026-07-04.md
5. Results
   Source: docs/paper_results_section_draft_2026-07-04.md
6. Limitations and Future Work
   Source: docs/paper_limitations_and_future_work_draft_2026-07-04.md
7. Conclusion
```

## Tables to copy into the paper

Source file:

```text
docs/final_paper_tables_2026-07-04.md
```

Tables:

```text
Table 1. Evaluation protocol and claim level
Table 2. TAPVid-DAVIS first/input official-style local evaluation
Table 3. TAPVid-DAVIS strided/original official-style local evaluation
Table 4. TAPVid RGB-Stacking full50 local standard evaluation
Table 5. RGB fresh20-49 and stress diagnostic evaluations
Table 6. V25 official-safe threshold sweep
Table 7. Method positioning / ablation summary
```

## Main claim to use

```text
ReEntry improves re-entry recovery across TAP-Vid local evaluations and diagnostic settings. It improves DAVIS first/input official-style AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter DAVIS strided/original evaluation requires conservative selection to preserve standard metrics.
```

## Claim boundary

Use:

```text
official-style local evaluation
official TAP-Vid metric implementation / equivalent port
TAPVid-DAVIS first/input and strided/original local protocols
TAPVid RGB-Stacking full50 local standard evaluation
```

Do not use:

```text
Do not claim: official leaderboard submission
Do not claim: full TAP-Vid official benchmark result
Do not claim: universal improvement across all protocols
Do not claim: V24 beats TrackOn2 on DAVIS
Do not claim: V24 is universally best
```

## Immediate next action

Do not start another large experiment before the paper draft is assembled.

Recommended next step:

```text
Create a combined paper draft skeleton that imports or references:
  - Experimental Setup
  - Results
  - Limitations
  - Final tables
```

Optional later experiment:

```text
RGB-Stacking full50 strict official first/strided query-protocol audit or rerun.
```
