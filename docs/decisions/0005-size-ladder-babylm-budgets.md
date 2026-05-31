# 5. Size ladder at BabyLM fixed budgets on a single DGX Spark

Date: 2026-05-31
Status: Accepted

## Context

All work is bounded by one DGX Spark GB10 (aarch64, 128GB, ~273 GB/s). A single
"headline" model size is the wrong frame: Chinchilla-optimal 350M is ~4 days per
run, too slow for an ablation battery, while BabyLM-style fixed token budgets fit
even 1B in hours (but only as an underfit reference). The scientific question is
"given the same data budget, which structural prior wins?" — sharpest where
models are comparably underfit.

## Decision

Use a three-tier ladder at BabyLM fixed budgets, not Chinchilla-optimal:

- **Tier 1 proxy:** 60M @ 10M-word (~25 min/run) — ablation workhorse.
- **Tier 2 battery:** 100–150M @ 100M-word (~1 hr/run) — all arms × ≥3–5 seeds;
  the publishable, peer-benchmarkable unit.
- **Tier 3 confirm:** 350M @ 100M-word (~2 hr/run) — single confirmation of the
  best arm via μTransfer.
- **1B:** only if Stage-1 go/no-go is strongly positive, at a fixed 1–2B-token
  budget (~35 hr), framed explicitly as an underfit scaling reference.

## Consequences

Fast iteration with many arms/seeds within a single machine. Claims are valid
within the BabyLM framing; world-knowledge benchmarks are honest reference points
only. The fixed budget is the scientific design, not a limitation.
