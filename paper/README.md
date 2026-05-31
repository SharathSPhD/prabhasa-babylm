# Paper

The PSALM manuscript: `psalm.tex` (IEEE single-column journal style, arXiv
preprint). The body is continuous journal prose — **no bulleted or numbered
lists** — with figures, flowcharts, and tables.

## Build

```bash
make paper          # latexmk -pdf psalm.tex
# or
cd paper && latexmk -pdf psalm.tex
```

## Rules

- Sections are filled incrementally **from findings** at each phase closure (the
  closure-contract ARTIFACTS layer), never from the plan.
- Every citation must resolve to a real, verifiable work. `references.bib` is a
  seed of verified entries; the full honest literature review is assembled in
  Phase 6 with each citation verified at cite time.
- The fabricated `arXiv:2605.12548` reference must never appear (ADR 0008).
- Tables report mean ± 95% CI; comparisons are Holm–Bonferroni corrected.
- Figures live in `paper/figures/`.
