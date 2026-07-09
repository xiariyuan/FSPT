# ReEntry-VisCalibrator TeX Compile Success — 2026-07-03

## Summary

The LaTeX environment was installed and the paper draft compiled successfully.

## Environment installation

The default Virtaicloud apt mirror returned repeated HTTP 500 errors during package download. The apt source was switched to official Ubuntu Jammy repositories:

```text
archive.ubuntu.com
security.ubuntu.com
```

Installed packages:

```text
latexmk
texlive-latex-base
texlive-latex-recommended
texlive-latex-extra
texlive-fonts-recommended
texlive-bibtex-extra
```

## Build command

```bash
cd paper/reentry_viscalibrator_tex
latexmk -pdf -interaction=nonstopmode main.tex
```

## Output

```text
paper/reentry_viscalibrator_tex/main.pdf
```

Build result:

```text
main.pdf exists
size: about 667 KB
pages: 7
```

## Final log status

Final checks:

```text
undefined references: none
undefined citations: none
overfull hbox warnings: none
latexmk exit: success
```

## Fixes made before final compile

1. Added `microtype` to improve typesetting.
2. Wrapped narrow tables in `resizebox`.
3. Replaced long script path text with breakable `\path{...}` formatting.
4. Fixed an accidental duplicate `resizebox` in the paired-video table.
5. Recompiled until citations and references resolved.

## Remaining non-blocking notes

1. `refs.bib` still includes placeholder metadata for some recent works.
2. The draft uses `article` class, not a target conference template.
3. Timeline figures compile, but visual readability should be inspected manually.
4. If moved to a conference style file, tables and figure sizes should be rechecked.
