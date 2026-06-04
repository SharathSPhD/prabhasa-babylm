# PSALM

**Pāṇinian Structured pretraining for Small LAnguage Models**

PSALM tests whether the most formally complete grammar ever written for a human
language — Pāṇini's Aṣṭādhyāyī — can serve as an *unbounded generative data
engine* that gives small language models a stronger structural inductive bias
than the artificial formal languages used in prior work, and whether a
Navya-Nyāya epistemic layer can make their reasoning formally disciplined.

This is an independent research program **and** a working end-to-end small
language model, built and trained on a single NVIDIA DGX Spark (GB10).

## The idea in one model, read three ways

PSALM trains a single small **multilingual** model:

- **Pre-pretraining** on Pāṇinian-generated synthetic Sanskrit (structure, with
  free gold kāraka / dependency parse annotations).
- **Pretraining** on real Sanskrit (DCS / GRETIL) for linguistic competence
  **and** BabyLM English for a competitive, peer-benchmarkable target.
- **Post-training** with a Navya-Nyāya 6-phase reasoning scaffold and an
  epistemic constraint kernel.

The same model is then evaluated three ways:

1. **English structural generalization** — SCAN, COGS/ReCOGS, CFQ, BLiMP, GLUE,
   EWoK (competitive BabyLM context).
2. **Sanskrit competence** — morphology, sandhi, kāraka role tasks (the grammar
   produced real linguistic competence, not discarded scaffolding).
3. **Cross-lingual transfer gap** — validates the premise that a structural
   prior transfers across languages.

## Hypotheses

- **H1 — Grammar Prior.** Pāṇinian pre-pretraining yields ≥20% token savings
  versus a matched k-Shuffle Dyck control on compositional benchmarks, *or* a
  ≥3-point compositional-accuracy gain.
- **H2 — Nyāya Scaffold.** A 6-phase Navya-Nyāya reasoning scaffold measurably
  lowers fallacious-inference rates. The novel **H1×H2 synergy test** asks
  whether a grammar-structured base is more *sample-efficient* to scaffold than
  a matched generic base.
- **H3 — Epistemic Constraint.** A GBNF schema + Z3 vyāpti verifier +
  hetvābhāsa filter enforce epistemic validity *by construction* at inference.

## Honest framing

PSALM does **not** claim frontier parity. Grammar generates *form*, not *facts*;
world-knowledge benchmarks (MMLU, ARC) are reported only as honest reference
points where the approach is expected to be non-competitive. The targets are
**sample efficiency, compositional generalization, and epistemic discipline** —
dimensions where structured data has theoretical purchase.

## Deliverables

A public dataset and model on the Hugging Face Hub (`qbz506/psalm-*`), this code
repository, an arXiv preprint written in IEEE single-column journal style, an
Astro GitHub Pages site that reframes the work for technical and non-technical
audiences, run-from-GitHub Google Colab notebooks, and Hugging Face Space demos.

## Quickstart

```bash
# Install with uv (recommended)
uv sync --extra dev

# Lint, type-check, test (the TECHNICAL closure gate)
uv run ruff check
uv run mypy
uv run pytest

# The PSALM CLI
uv run psalm --help
```

For GPU work on the DGX Spark, build the container:

```bash
docker build -t psalm:dev .
```

## Run the demos in Colab

Each notebook opens directly in Google Colab from this repository and loads the
published model from the Hub — nothing to train, nothing to download by hand.

| Notebook | What it does | Open |
|---|---|---|
| Minimal-pair scoring | Score BLiMP-style pairs by pseudo-log-likelihood | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/01_minimal_pairs.ipynb) |
| Śabdabodha pipeline | Sentence → typed semantic graph → Paribhāṣā string | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/02_shabdabodha_pipeline.ipynb) |
| Reproduce an eval | Re-run the official zero-shot suite on a checkpoint | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/03_reproduce_eval.ipynb) |

## Repository layout

```
src/psalm/
  domain/          # Pure logic: experiments, contracts, rewards, validators
  application/     # Use cases: data, training, evaluation orchestration
  infrastructure/  # External: generators, tokenizer, ml, storage, verification, ledger
  config/          # pydantic-settings + YAML config loading
  cli/             # typer command-line interface
  analysis/        # statistical validation + visualization
  benchmarks/      # eval-suite runners (Vyapti Probe, compositional, etc.)
configs/           # YAML run configurations (config-driven, no magic numbers)
docs/              # spec, prd, contracts, decisions (ADRs), memory, experiments
paper/             # arXiv / IEEE single-column manuscript
site/              # Astro GitHub Pages site
notebooks/         # Colab demos that clone-and-run from GitHub
scripts/           # thin entrypoints over the library
```

## How the program runs

PSALM executes as a phased, contract-bound program. Each phase runs in its own
git worktree, may fan out into parallel sub-agents, and **closes only when it
satisfies the Ralph-loop closure contract** (technical + empirical + integrity +
artifacts + memory + human interpretation sign-off). See
[`docs/contracts/closure-contract.md`](docs/contracts/closure-contract.md).

## License

MIT — see [LICENSE](LICENSE).
