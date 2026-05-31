# 3. H1-heavy scope and pre-registered thresholds

Date: 2026-05-31
Status: Accepted

## Context

The program spans corpus generation, pretraining, post-training, and benchmarking
across three hypotheses. Spreading effort evenly would dilute the load-bearing
claim. H1 (the grammar prior) is the foundation; H2 and H3 only matter if H1
holds. Outcome-only contracts let a run "succeed" technically while proving
nothing, so thresholds must be fixed in advance.

## Decision

Weight the program toward H1. Pre-register go/no-go thresholds in
`configs/*.yaml` (`go_no_go` block), enforced and reported via
`psalm.analysis.comparison_tests`:

- **H1 primary:** ≥20% token savings vs matched k-Shuffle Dyck on compositional
  benchmarks, OR ≥3-point compositional-accuracy gain.
- Significance: ≥3–5 seeds, mean ± 95% CI, permutation tests, Holm–Bonferroni
  across arms, α = 0.05.

Iterate to *prove* the hypothesis (mandatory intervention loop), not to report
the first iteration. Changing a threshold requires a superseding ADR.

## Consequences

Clear, falsifiable success criteria fixed before data is seen. The intervention
loop and Tarka memo prevent both premature success and premature failure. The
cost is discipline: a missed threshold triggers documented diagnosis and re-runs.
