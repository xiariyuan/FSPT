# LaTeX Paper Draft

This directory contains the current LaTeX draft of the PRT paper.

## Files

- `main.tex`: main IEEE-style manuscript draft
- `refs.bib`: BibTeX references used by the current draft
- `figures/`: copied paper figures from `outputs/paper_assets/`

## Current Status

- The paper structure is complete and written in LaTeX.
- Main sections are filled: abstract, introduction, related work, problem formulation, method, experiments, discussion, conclusion.
- Main quantitative tables and figure references are integrated.
- BibTeX entries are present for the core related work.
- The current positioning is an event-level recovery evaluation plus a lightweight support-memory selector, not a full new tracker architecture.

## Known Limitations

1. This environment does not currently provide `pdflatex`, so local compilation was not verified here.
2. Some bibliography entries should still be checked against final official metadata before submission.
3. The manuscript has not been compile-verified in this environment.
4. The manuscript still needs final polishing for style, table compression, captions, and venue-specific formatting details.

## Expected Build Commands

If a LaTeX environment is available, compile with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Recommended Next Steps

1. Compile and fix any LaTeX syntax or package issues.
2. Tighten the related-work citations and author metadata.
3. Add final teaser / pipeline figure if desired.
4. Replace `Anonymous Authors` with the actual author block when ready.
