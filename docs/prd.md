# PSALM product requirements document

PSALM is a research/product blend: it must both establish novel scientific claims
and ship a working, public, end-to-end small language model. This PRD defines the
deliverables and their acceptance criteria. The experimental design is in
`docs/spec.md`.

## Audiences

- ML researchers (BabyLM / CoNLL / arXiv): want clean, reproducible, peer-
  benchmarkable claims with honest statistics and ablations.
- Computational-linguistics / Indology community: want genuine Sanskrit
  competence and faithful use of Pāṇinian and Navya-Nyāya frameworks.
- Practitioners and the curious public: want an accessible story and runnable
  demos.

## Deliverables and acceptance criteria

Dataset (HF `qbz506/psalm-*`). The Pāṇinian synthetic corpus (with gold kāraka
parses), the assembled real-Sanskrit + English pretraining mix manifest, and the
k-Shuffle Dyck control set. Acceptance: license-clean, documented dataset card,
diversity/coverage statistics reported, loads via `datasets`.

Model(s) (HF `qbz506/psalm-*`). The PSALM base SLM (best Tier-2 arm, plus the
350M confirmation), and the H2 scaffolded variants. Acceptance: model card with
training config, eval results with CIs, and honest limitations; loads via
`transformers`; reproducible from a config + ledger entry.

Code repository (this repo). Acceptance: technical gate green on CI; hexagonal,
config-driven, TDD; every published result reproducible from a committed config.

Paper (arXiv preprint, IEEE single-column journal style). Acceptance: true
journal prose (no bullets/numbering in the body), rigorous honest literature
review with only verifiable citations, figures/flowcharts/tables, ablations, CIs,
and an explicit limitations section. BabyLM/CoNLL-ready.

Site (Astro, GitHub Pages). Acceptance: presents the depth of the paper but
reframed for technical and non-technical audiences, with the actual results and
figures; builds and deploys via GitHub Pages.

Demos. Run-from-GitHub Google Colab notebooks (clone-and-run) and Hugging Face
Space demos. Acceptance: a fresh Colab/Space runs end-to-end from a link without
manual fixes.

## Quality bars (program-wide)

Every published claim carries uncertainty (≥3–5 seeds, 95% CI) and a fair, matched
comparison. Every phase passes the six-layer closure contract. No fabricated
citations; no plagiarism. Honest framing: PSALM targets sample efficiency,
compositional generalization, and epistemic discipline — not frontier parity.

## Milestones

Phase 0 foundation → Phase 1 data engine + tokenizer → Phase 2 H1 controlled
battery → Phase 3 H1 scale confirmation → Phase 4 H2 Nyāya scaffold → Phase 5 H3
epistemic kernel → Phase 6 dissemination. Each milestone merges to main after
human sign-off on the interpretation.

## Risks

GB10/Blackwell/aarch64 + CUDA-13 stack maturity (Unsloth, flash-attn); Saṃsādhanī
generator diversity ceiling; Z3 vyāpti coverage limited to formally expressible
rules. Each is de-risked early and tracked in `docs/decisions/` and the ledger.
