# Phase 2 (H1 controlled experiment) — status

Status date: 2026-05-31. Host: DGX Spark, GB10 Grace-Blackwell, `aarch64`.

## What is built and green

The complete H1 experiment harness is implemented, unit-tested, and validated
end-to-end at micro scale:

- **Experiment matrix** (`domain/experiments/matrix.py`): config-driven arms A–G
  at any model size, with executable fairness invariants (`verify_fairness`)
  protecting the decisive B-vs-C pair and the matched low-budget E/F/G triplet.
- **Corpus assembly** (`application/data/assembly.py`, ADR-0011): per-arm
  structural streams from the Phase-1 Vidyut/Dyck generators; arm D shares arm
  B's input and differs only in the auxiliary derivation target; token-budget
  enforcement.
- **Model** (`domain/model/config.py`): μP-aware decoder config with verified
  param-count math and a 60/100/150/350M preset ladder.
- **Trainer** (`infrastructure/ml/trainer.py`): packed-LM AdamW loop with cosine
  LR, grad accumulation/clipping, bf16/fp16 autocast, optional arm-D aux loss.
  CPU smoke test drives loss below the uniform-prior bound.
- **Evaluation** (`domain/eval/`, `infrastructure/ml/eval_lm.py`): minimal-pair
  accuracy, exact-match, token-savings; torch LM scorer and greedy decoder.
- **Go/no-go** (`domain/eval/go_no_go.py`): encodes the phase-2 contract
  (≥20% token savings OR ≥3-pt significant compositional gain) with permutation
  test + bootstrap CI.
- **Orchestrator** (`application/experiments/orchestrator.py`): refuses unfair
  matrices, runs arms×seeds, logs every run to the SQLite ledger, computes the
  B-vs-C decision.
- **End-to-end proxy** (`scripts/run_h1_proxy.py`): runs all 7 arms × 2 seeds on
  CPU with the real Vidyut and Dyck generators, producing a ledger and an
  `H1Decision`. As designed for an untrained micro-model on a copy task it
  returns accuracy 0.0 / null — proving the wiring is correct without
  fabricating a signal.

## What the empirical finding still requires (compute + data)

The scientific go/no-go cannot be honestly declared until the real battery runs.
That is a GPU + provisioning step, not a coding step:

- **Provision corpora:** BabyLM English (research-only) for NL continuation; the
  Pāṇinian stream is already available offline via Vidyut, the Dyck control is
  generated.
- **Provision eval suites:** SCAN, COGS/ReCOGS, CFQ (compositional), BLiMP
  (syntactic), ARC-Easy/HellaSwag (honest reference).
- **Run on the GB10:** 60M proxy sanity pass, then the 100–150M battery across
  arms A–G × ≥3 seeds, inside the NGC PyTorch Blackwell container (ADR-0007).
  Estimated order hours per the research-3 throughput table.

## Honest gate position

This is the Phase-2 analogue of the Phase-1 external-provisioning gate: the
engine is complete and technically green, but the *finding* depends on executing
the battery. No compositional-accuracy numbers are reported until that run
produces them. The Tarka memo and paper H1 section will be written FROM the
finding, not before it.
