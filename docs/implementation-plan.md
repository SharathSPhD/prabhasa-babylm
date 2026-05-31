# PSALM implementation plan

Sequencing for the program. Each phase runs in its own git worktree, may fan out
into parallel sub-agents on independent tracks, and closes only via the six-layer
Ralph-loop contract (`docs/contracts/closure-contract.md`). Merge to main + push
at each milestone after human sign-off on the interpretation.

## Phase 0 — Foundation (this phase)

Repo bootstrap, hexagonal scaffold, uv env, NGC Blackwell/aarch64 Dockerfile, CI
technical gate, triz-engine + attractor-flow plugins, layered memory, executable
closure contract, and the full governance doc set. Exit: technical gate green;
docs + contracts in place; human sign-off on the program design.

## Phase 1 — Data engine + tokenizer

Tracks (parallel sub-agents): (a) Saṃsādhanī generator wrapper streaming
`(sentence, kāraka parse, derivation)`; (b) diversity/coverage measurement;
(c) sandhi-aware SentencePiece tokenizer; (d) license-clean corpus assembly
(DCS/GRETIL/BabyLM/HF Sanskrit); (e) matched k-Shuffle Dyck control generator;
(f) HF dataset publication + card. De-risk the GB10 software stack here.
Go/no-go: generator diversity sufficient to support the pre-pretraining budget.

## Phase 2 — H1 controlled experiment

60M proxy sanity pass, then the 100–150M Tier-2 battery across arms A–G × seeds.
Eval suites: compositional (SCAN/COGS/CFQ), syntactic (BLiMP), general
(GLUE/EWoK), Sanskrit (morphology/sandhi/kāraka), transfer gap. Compute the H1
go/no-go (B vs C); on a miss, run the mandatory intervention loop; write the Tarka
memo; draft the paper's H1 section from the finding.

## Phase 3 — H1 scale confirmation

Promote the best arm to 350M via μTransfer; produce the token-efficiency curve;
optional bounded 1B underfit check if the go/no-go is strongly positive. Report
MMLU/ARC honestly as reference points.

## Phase 4 — H2 Nyāya scaffold

DeepSeek-R1-8B + scaffold (ceiling) and the PSALM-base vs Generic-1B sample-
efficiency synergy test. Data 55→500 examples (gold/silver/bronze). SFT (Unsloth
QLoRA) + GRPO with a Nyāya process reward model. Metric: examples-to-target-
quality and fallacious-inference rate.

## Phase 5 — H3 epistemic constraint kernel

GBNF schema enforcement, Z3 vyāpti verification, hetvābhāsa filter. Measure
constraint ON/OFF failure-rate reduction and the fluency tradeoff. Honest PoC
scope for Z3 coverage.

## Phase 6 — Dissemination

Finalize the arXiv/IEEE journal-style paper (honest full lit review,
figures/flowcharts/tables, no bullets), the Astro GitHub Pages site (technical +
lay), run-from-GitHub Colab notebooks, HF Space demos, and dataset/model cards.

## Cross-cutting

Statistical validation and the experiment ledger are updated every phase. The
paper and site grow incrementally from findings, not from the plan. Adversarial
reviews (code/docs/paper/demo) run before each closure.
