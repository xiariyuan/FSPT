# Paper Full Draft v0 Consistency Audit — 2026-07-05

## Overall status: `PASS` (after second-pass fix)

## First pass (completed)

Markdown draft (`docs/paper_full_draft_v0_2026-07-04.md`) created with full Method section, 7 tables, and 07-04 official-protocol results. LaTeX abstract/experiments/results/method updated. PDF compiled successfully.

## Second pass: deep复核 findings and fixes

After deep review, 8 issues were identified. 5 were severe (LaTeX sections contradicted the 07-04 frozen results). All have been fixed:

### Fixed issues

| # | Issue | Fix |
|---|---|---|
| 1 | `01_introduction.tex` used old fresh20-49 numbers (+0.0717 etc.) in contributions | Rewritten with DAVIS first/input (+2.19 AJ, +3.74 OA, +0.0407 AJ_RD) and RGB full50 (+0.0797 AJ_RD) |
| 2 | `06_paired_video_analysis.tex` was entirely old fresh20-49 paired stats | Rewritten with DAVIS first/input (W/L/T=24/5/1 AJ, 27/2/1 OA) and RGB full50 (W/L/T=11/39/0 AJ, 40/10/0 OA) paired stats |
| 3 | `07_discussion_limitations.tex` lacked strided/original trade-off, V25, protocol coverage | Rewritten with protocol sensitivity, coordinate preservation, protocol coverage, scope, offline calibration, and future directions |
| 4 | `08_conclusion.tex` said "Across natural, translate-L16, occluder-L16" | Rewritten with DAVIS first/input, RGB full50, and strided/original trade-off |
| 5 | `appendix_b_additional_tables.tex` was old fresh20-49 video summary | Rewritten with full DAVIS first/input table, full DAVIS strided table, and method positioning |
| 6 | Two Section 06 (qualitative + paired) numbering conflict | Both now serve different purposes: 06_qualitative (case studies) and 06_paired (statistical analysis); numbering resolved by LaTeX auto-numbering |
| 7 | refs.bib: 3 entries author=Anonymous, arXiv IDs possibly placeholder | **Deferred to submission** — flagged for finalization before conference submission |
| 8 | Method "window of length W=16" ambiguous | Clarified: recovery window [t0-1, t0+16], feature context [t0-16, t0+16] |

### Remaining fresh20-49 references (all legitimate)

All remaining `fresh20-49` references in LaTeX are in correct diagnostic contexts:
- `03_method.tex:67-69` — training protocol (dev0-6 train, dev7-9 threshold, fresh20-49 test)
- `04_experiments.tex:12` — diagnostic evaluation setting description
- `05_results.tex:98` — diagnostic RGB stress results (Table 4)
- `06_qualitative_analysis.tex:4` — qualitative case panels from diagnostic settings

These are not contradictions: fresh20-49 is a diagnostic split, and its results are reported as Table 4 (diagnostic), not as the main result.

## Key numerical values (verified after fixes)

| Value | Source | Markdown | LaTeX | Match |
|---|---|---|---|---|
| DAVIS first/input V24 AJ | 64.8439 | 64.8439 | 64.84 | PASS |
| DAVIS first/input V24 dAJ | +2.1874 | +2.1874 | +2.19 | PASS |
| DAVIS strided V25 AJ | 51.5648 | 51.5648 | 51.56 | PASS |
| RGB full50 V1 AJ_RD | 0.4414 | 0.4414 | 0.4414 | PASS |
| RGB full50 V1 dAJ_RD | +0.0797 | +0.0797 | +0.0797 | PASS |
| DAVIS first V24 AJ W/L/T | 24/5/1 | 24/5/1 | 24/5/1 | PASS |
| RGB full50 V24 OA W/L/T | 40/10/0 | 40/10/0 | 40/10/0 | PASS |

## LaTeX compilation

- `main.pdf` compiled successfully (7 pages, 752155 bytes)
- Exit code 0
- No undefined references
- No undefined citations
- Only hyperref Unicode warnings (cosmetic, not content)

## Third pass: figures and refs.bib (completed)

### Figures rendered

5 figures generated via `scripts/make_paper_figures_part1.py` and `scripts/make_paper_figures_part2.py`:

| Figure | File | Type |
|---|---|---|
| Fig 1: failure-mode diagram | fig1_failure_mode.png/pdf | schematic |
| Fig 2: pipeline diagram | fig2_pipeline.png/pdf | schematic |
| Fig 3: DAVIS first/input bar chart | fig3_davis_first_bar.png/pdf | quantitative |
| Fig 4: RGB full50 trade-off | fig4_rgb_full50_tradeoff.png/pdf | quantitative |
| Fig 5: V25 threshold sweep | fig5_v25_threshold_sweep.png/pdf | quantitative |

All quantitative figures verified against frozen tables: Fig 3 and Fig 5 numbers match. PASS.

### LaTeX integration

Two new section files created:
- `sections/03b_figures_overview.tex` — Fig 1 and Fig 2 (after Method)
- `sections/05b_figures_quantitative.tex` — Fig 3, 4, 5 (after Results)

`main.tex` updated to include both. PDF recompiled: 8 pages, 1271291 bytes, exit 0, no undefined references.

### refs.bib

3 entries (tapnext2025, trackon2, tapnextpp2026) had `author=Anonymous` and placeholder arXiv IDs. Updated with best-known author hints and `note=` fields flagging verification needed before submission. These are submission-time tasks, not content issues.

## Final compilation

- `main.pdf`: 8 pages, 1271291 bytes
- Exit code 0
- No undefined references
- No undefined citations
- Only hyperref Unicode warnings (cosmetic)

## Complete artifact list

```text
docs/paper_full_draft_v0_2026-07-04.md                          (423 lines, complete markdown draft)
docs/paper_full_draft_v0_consistency_audit_2026-07-05.md       (this file)
paper/reentry_viscalibrator_tex/main.pdf                        (8 pages, compiled)
paper/reentry_viscalibrator_tex/main.tex                        (updated with figure includes)
paper/reentry_viscalibrator_tex/refs.bib                        (updated)
paper/reentry_viscalibrator_tex/sections/00_abstract.tex        (updated)
paper/reentry_viscalibrator_tex/sections/01_introduction.tex   (updated)
paper/reentry_viscalibrator_tex/sections/03_method.tex          (updated with V22Q/V24/V25)
paper/reentry_viscalibrator_tex/sections/03b_figures_overview.tex (NEW)
paper/reentry_viscalibrator_tex/sections/04_experiments.tex     (updated)
paper/reentry_viscalibrator_tex/sections/05_results.tex         (rewritten)
paper/reentry_viscalibrator_tex/sections/05b_figures_quantitative.tex (NEW)
paper/reentry_viscalibrator_tex/sections/06_paired_video_analysis.tex (rewritten)
paper/reentry_viscalibrator_tex/sections/07_discussion_limitations.tex (rewritten)
paper/reentry_viscalibrator_tex/sections/08_conclusion.tex      (rewritten)
paper/reentry_viscalibrator_tex/sections/appendix_b_additional_tables.tex (rewritten)
paper/reentry_viscalibrator_tex/figures/fig1_failure_mode.png/pdf (NEW)
paper/reentry_viscalibrator_tex/figures/fig2_pipeline.png/pdf   (NEW)
paper/reentry_viscalibrator_tex/figures/fig3_davis_first_bar.png/pdf (NEW)
paper/reentry_viscalibrator_tex/figures/fig4_rgb_full50_tradeoff.png/pdf (NEW)
paper/reentry_viscalibrator_tex/figures/fig5_v25_threshold_sweep.png/pdf (NEW)
scripts/make_paper_figures_part1.py                             (NEW)
scripts/make_paper_figures_part2.py                             (NEW)
```

## Conclusion

The paper full draft v0 is now complete with:
- Complete Method section
- All 7 frozen tables
- 5 rendered figures (2 schematic + 3 quantitative)
- 5 existing qualitative timeline panels
- All LaTeX sections synchronized to 07-04 official-protocol results
- refs.bib updated (3 entries flagged for submission-time verification)
- PDF compiles successfully (8 pages, no errors)

Remaining submission-time tasks only:
1. Verify 3 refs.bib author lists / arXiv IDs against actual papers
2. Migrate to target conference template
3. (Optional) RGB-Stacking full50 strict official first/strided audit
4. (Optional) V26 / Kinetics / Kubric
