# ReEntry-VisCalibrator TeX Static Check — 2026-07-03

## Summary

The LaTeX paper draft has been statically checked. The current runtime does not have a TeX compiler, so no PDF was generated here. However, the source tree is internally consistent according to static checks.

## TeX engine availability

Checked commands:

```bash
which pdflatex
which latexmk
which xelatex
which lualatex
which tectonic
which bibtex
```

Result:

```text
No LaTeX engine path was available in the current environment.
```

## Source folder

```text
paper/reentry_viscalibrator_tex/
```

Important files:

```text
main.tex
macros.tex
refs.bib
BUILD.md
README.md
sections/*.tex
figures/*.png
```

## Static checks passed

```text
all input files exist: PASS
all referenced image paths exist: PASS
all citation keys exist in refs.bib: PASS
no duplicate labels: PASS
simple begin/end environment balance: PASS
```

## Current figure references

The qualitative section references:

```text
figures/qual_timeline_natural_det_over_recovery.png
figures/qual_timeline_occluder_success.png
figures/qual_timeline_occluder_learned_preserves_base.png
figures/qual_timeline_natural_failure_short_occ.png
```

All exist.

Generated but not currently included in the main qualitative figure:

```text
figures/qual_timeline_natural_base_fail_learned_success.png
```

This can be used in the appendix or swapped into the main figure.

## Recommended external compile command

```bash
cd paper/reentry_viscalibrator_tex
latexmk -pdf -interaction=nonstopmode main.tex
```

or:

```bash
cd paper/reentry_viscalibrator_tex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Known remaining paper-edit risks

1. `refs.bib` contains placeholder metadata for some recent works and should be finalized.
2. The qualitative timeline panels may be small in two-column format; inspect the compiled PDF and enlarge if needed.
3. The main result table uses `resizebox`; if moving to a conference template, check readability.
4. The paper currently uses `article` class, not a target conference class.
5. The draft's claims are deliberately careful: Ours-Learned is the best overall version, but the deterministic variant captures most AJ_RD recovery.
