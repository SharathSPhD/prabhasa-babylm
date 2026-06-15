# Paper

The Prabhāsa manuscript is provided in two versions: `psalm.tex` (the long-form
MDPI journal article, for venues without a page limit) and `psalm_short.tex`
(the 8-page BabyLM submission with expanded appendices). Both use continuous
journal prose (no bulleted or numbered lists) with figures, flowcharts, and
tables.

## Build

```bash
make paper          # latexmk -pdf psalm.tex
# or
cd paper && latexmk -pdf psalm.tex
```

## Rules

- Sections are filled incrementally **from findings** at each phase closure (the
  closure-contract ARTIFACTS layer), never from the plan.
- Every citation must resolve to a real, verifiable work; `references_new.bib`
  holds the verified entries used by both paper versions, each checked at cite
  time. Fabricated or unverifiable references must never appear.
- Tables report mean ± 95% CI; comparisons are Holm–Bonferroni corrected.
- Figures live in `paper/figures/`.
