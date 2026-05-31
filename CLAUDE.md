# CLAUDE.md — PSALM operating guide for AI agents

This file is always-on guidance for any agent working in this repository. It
encodes what PSALM is, the rules that bind every phase, and the conventions that
keep the program reproducible and honest. Read it before acting.

## What PSALM is

**Pāṇinian Structured pretraining for Small LAnguage Models.** One small
multilingual model, three readings:

- **Pre-pretraining** on Pāṇinian synthetic Sanskrit (structure + gold kāraka
  parses) — the H1 independent variable.
- **Pretraining** on real Sanskrit (DCS/GRETIL) + BabyLM English.
- **Post-training** with a Navya-Nyāya reasoning scaffold (H2) and an epistemic
  constraint kernel (H3).

Evaluated for English structural generalization (SCAN/COGS/CFQ/BLiMP/GLUE/EWoK),
Sanskrit competence (morphology/sandhi/kāraka), and cross-lingual transfer.

The program is **H1-heavy**. H1 is the load-bearing claim; H2 and H3 build on a
validated H1 base.

## The hypotheses and their pre-registered thresholds

- **H1 (Grammar Prior):** Pāṇinian pre-pretraining yields ≥20% token savings vs
  a matched k-Shuffle Dyck control on compositional benchmarks, OR a ≥3-point
  compositional-accuracy gain. (`go_no_go.token_savings_vs_dyck`, `...gain_points`)
- **H2 (Nyāya Scaffold):** A 6-phase reasoning scaffold lowers fallacious-
  inference rates. The novel claim is the **H1×H2 synergy test** — PSALM-base +
  Nyāya vs a matched Generic-1B (TinyLlama) + Nyāya, measuring *sample
  efficiency*. DeepSeek-R1-8B + Nyāya runs separately as the ceiling.
- **H3 (Epistemic Constraint):** GBNF schema + Z3 vyāpti verifier + hetvābhāsa
  filter enforce validity by construction; measure the fluency cost.

Changing a pre-registered threshold requires an ADR in `docs/decisions/`.

## The closure contract is binding (this is the most important rule)

A phase is **not** done when CI is green. A phase closes only when it satisfies
the six-layer Ralph-loop contract, encoded in
`src/psalm/domain/contracts/closure.py` and checkable with
`psalm contract check <report.json>`:

1. **TECHNICAL** — tests pass, ruff + mypy clean, coverage ≥ 80%.
2. **EMPIRICAL** — all arms run, go/no-go metric computed + logged, a *finding*
   declared (positive / marginal / null) with an interpretation paragraph.
3. **INTEGRITY** — a Tarka memo (the strongest objection to your own finding),
   resolved; comparison fairness verified.
4. **ARTIFACTS** — code + results pushed, paper section updated *from the
   finding*, demos run clean.
5. **MEMORY** — experiment ledger updated, docs reflect current state.
6. **SIGN-OFF** — human sign-off on the *interpretation* before merge-to-main.

Hard rules:

- **Never declare "failed" on attempt 1.** If a metric misses threshold, run the
  mandatory intervention loop: diagnose why, hypothesize a fix, re-run, log.
- **Never declare a NULL finding without ≥2 documented interventions.**
- A well-documented null is a valid closure state. An unexplained failure is not.
- The ledger entry must carry attempt #, what changed, result, interpretation.

## How we work

- **TDD.** Write the failing test first, then the code. Domain and application
  layers must be unit-tested; coverage gate is 80%.
- **Config-driven.** No magic numbers in code. Everything that varies between
  runs lives in `configs/*.yaml` → validated by `psalm.config.settings`. The
  resolved config hash is recorded with every run.
- **Hexagonal architecture.** `domain/` is pure (no torch, no I/O).
  `application/` orchestrates use cases. `infrastructure/` wraps external
  systems (ML frameworks, generators, HF, Z3, storage). Dependencies point
  inward only.
- **Statistical honesty.** ≥3–5 seeds; report mean ± 95% CI; compare arms with
  paired bootstrap / permutation tests; correct families with Holm–Bonferroni
  (`psalm.analysis.comparison_tests`). Pre-register thresholds.
- **Citation integrity.** No fabricated citations, no plagiarism. The
  "Cubical Type Theoretic Navya-Nyāya" arXiv:2605.12548 reference is fabricated —
  **do not cite it.** Verify every citation resolves to a real work.
- **Honest framing.** PSALM targets sample efficiency, compositional
  generalization, and epistemic discipline — not frontier parity. Report
  world-knowledge benchmarks (MMLU/ARC) only as honest reference points.

## Hardware & environment

- Single **NVIDIA DGX Spark (GB10 Grace-Blackwell, aarch64, 128GB, ~273 GB/s)**.
  No cloud unless a strong go/no-go warrants one bounded 1B run.
- `uv` for envs. NGC PyTorch container (Blackwell/CUDA-13/arm64) — verify Unsloth
  + flash-attn build on sm_121 early (flagged risk).
- Size ladder (BabyLM fixed budgets): 60M proxy (~25 min) → 100–150M battery
  (~1 hr, the publishable unit) → 350M confirm (~2 hr) → optional 1B underfit.

## Tooling

- **triz-engine** MCP — contradiction resolution (TRIZ matrix / IFR).
- **attractor-flow** MCP — divergent→convergent ideation + orchestration regime
  monitoring (detect STUCK/CYCLING intervention loops).
- **Serena** MCP — semantic code memory. **HF Hub** — publish under `qbz506/`.
- Memory layers: this file, `.serena/memories/`, `docs/decisions/` (ADRs),
  `docs/memory/` (knowledge base), the experiment ledger, and the SQLite/vector
  knowledge store (`psalm.infrastructure.storage.knowledge_store`).

## Orchestration

One git worktree per phase, phases sequential; parallel sub-agents within a
phase on independent tracks. Automated adversarial review (code/docs/paper/demo)
before closure. Merge to main + push at each milestone after human sign-off.

## Commands

```bash
make gate          # TECHNICAL closure gate: ruff + mypy + format + tests/coverage
uv run psalm --help
uv run psalm config show configs/phase2/arm_B_paninian_en.yaml
uv run psalm contract check <report.json>   # exits nonzero unless phase closed
```
