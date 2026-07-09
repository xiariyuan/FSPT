# Next Step after Final Writing Audit — 2026-07-04

## 1. Current status

Final consistency audit status:

```text
PASS
```

Audit file:

```text
docs/final_writing_package_consistency_audit_2026-07-04.md
```

Summary:

```text
No numeric mismatch found.
No missing core writing artifact found.
No unsafe unbounded claim found.
Risky phrases are now explicitly bounded by Do not claim / Avoid contexts.
```

The paper writing package is ready for assembly.

---

## 2. What should happen next

The next step should be paper assembly, not more experiments.

Create a full paper draft skeleton and integrate the existing finalized writing pieces:

```text
1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experimental Setup
6. Results
7. Limitations and Future Work
8. Conclusion
```

Use these source files:

```text
docs/paper_combined_draft_core_sections_2026-07-04.md
docs/final_paper_tables_2026-07-04.md
docs/paper_experimental_setup_section_draft_2026-07-04.md
docs/paper_results_section_draft_2026-07-04.md
docs/paper_limitations_and_future_work_draft_2026-07-04.md
```

---

## 3. Immediate concrete task

Create:

```text
docs/paper_full_draft_skeleton_2026-07-04.md
```

The skeleton should include:

```text
Title
Abstract
Contributions
Introduction outline
Related Work outline
Method outline
Experimental Setup full draft
Results full draft
Limitations and Future Work full draft
Conclusion
Tables checklist
Claim boundary checklist
```

---

## 4. Do not do yet

Do not start these before the full paper skeleton is assembled:

```text
Kinetics full run
Kubric full run
Large RGB-Stacking official query-protocol rerun
V26 model development
```

Reason:

```text
The current paper evidence is already coherent. More experiments may change the narrative and delay paper assembly.
```

---

## 5. Optional later supplements

After the full paper skeleton is assembled, optional supplements are:

```text
1. RGB-Stacking full50 official first/strided query-protocol audit or rerun.
2. V26 metric-aware selective ReEntry.
3. Kinetics/Kubric if compute budget allows.
```

Priority order:

```text
Paper skeleton > polished Introduction/Method > RGB strict-protocol supplement > V26 > Kinetics/Kubric
```

---

## 6. Current final claim

Use this as the paper's central result claim:

```text
ReEntry improves re-entry recovery across TAP-Vid local evaluations and diagnostic settings. It improves DAVIS first/input official-style AJ/OA/AJ_RD and RGB-Stacking full50 AJ_RD/OA, while stricter DAVIS strided/original evaluation requires conservative selection to preserve standard metrics.
```

---

## 7. Claim boundary

Safe:

```text
official-style local evaluation
TAPVid-DAVIS first/input and strided/original local protocols
TAPVid RGB-Stacking full50 local standard evaluation
re-entry diagnostic metric AJ_RD
```

Do not claim:

```text
official leaderboard submission
full TAP-Vid official benchmark result
universal improvement across all official protocols
V24 beats TrackOn2 on DAVIS
V24 is universally best
```
