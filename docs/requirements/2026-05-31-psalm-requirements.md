# PSALM requirements (from brainstorming, 2026-05-31)

This document records the requirements as converged during the brainstorming
interview. It is the source the spec, PRD, and plan derive from.

## Goal

Build an independent research program and a working end-to-end small language
model that tests whether a Pāṇinian grammar data engine (H1), a Navya-Nyāya
reasoning scaffold (H2), and an epistemic constraint kernel (H3) yield sample-
efficient, compositionally strong, epistemically disciplined SLMs — all on a
single DGX Spark, with public artifacts and a publishable paper.

## Functional requirements

- Generate Pāṇinian synthetic Sanskrit with gold kāraka parses (Saṃsādhanī
  wrapper) and measure diversity/coverage.
- Build a sandhi-aware SentencePiece tokenizer.
- Assemble license-clean real Sanskrit (DCS/GRETIL/HF) + BabyLM English corpora.
- Implement a matched k-Shuffle Dyck control generator.
- Pretrain 60M / 100–150M / 350M decoders from scratch at BabyLM fixed budgets.
- Run the H1 arm battery (A–G) with ≥3–5 seeds and the eval suites.
- Post-train the Navya-Nyāya scaffold (SFT + GRPO + process reward model).
- Implement the epistemic kernel (GBNF + Z3 vyāpti + hetvābhāsa filter).
- Publish dataset(s) and model(s) to HF; produce paper, site, Colab, HF Spaces.

## Non-functional requirements

- Single DGX Spark GB10 (aarch64, 128GB); no cloud unless a strong go/no-go
  warrants one bounded 1B run.
- Hexagonal, config-driven, TDD; ≥80% coverage; ruff + mypy strict.
- Statistical rigor: pre-registered thresholds, CIs, permutation tests,
  Holm–Bonferroni.
- Citation integrity (no fabricated refs; arXiv:2605.12548 excluded); no
  plagiarism; honest framing (no frontier-parity claims).
- Reproducibility: every published result reproducible from a committed config +
  ledger entry.

## Process requirements

- Each phase closes only via the six-layer Ralph-loop contract; iterate to prove
  hypotheses (mandatory intervention loop), never accept the first iteration as
  the answer.
- One git worktree per phase, phases sequential, parallel sub-agents within a
  phase; automated adversarial review before closure; merge to main + push at
  each milestone after human sign-off on the interpretation.
- Use triz-engine for contradictions, attractor-flow for ideation/steering.

## Decisions locked (see ADRs)

- One model, three readings (ADR 0002); H1-heavy + pre-registered thresholds
  (0003); Ralph-loop contract (0004); BabyLM size ladder (0005); two Nyāya bases
  + synergy test (0006); NGC Blackwell/aarch64 stack (0007); citation integrity
  (0008).
- HF namespace `qbz506`; venue arXiv + BabyLM/CoNLL; memory backend vector/SQLite.

## Open questions / risks

- Saṃsādhanī generator diversity ceiling (measure before committing compute).
- Unsloth/flash-attn build on GB10 sm_121 (de-risk in Phase 0/1).
- Z3 vyāpti coverage (formally expressible rules only — honest PoC scope).
