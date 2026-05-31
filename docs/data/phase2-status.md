# Phase 2 (H1 controlled experiment) — status

Status date: 2026-05-31. Host: DGX Spark, GB10 Grace-Blackwell, `aarch64`.

## Re-scope: real Saṃsādhanī + corpora integrated (ADR-0012)

The `panini-data-toolkit` (MIT) is now a PSALM dependency and the original
Phase-1 blocker — the proprietary, unprovisioned Saṃsādhanī generator — is
**resolved on this machine**:

- **Real Pāṇinian sentence generator is live** (`SamsadhaniiGenerator`, container
  at `localhost:8090`). It produces full kāraka-composed Sanskrit sentences with
  **gold (surface, role) annotation for free**. Measured over 150 sentences:
  150/150 distinct, bigram entropy 0.99, trigram 1.00, 100% gold parse
  (`docs/data/phase2-samsadhani-diversity.json`). This replaces the form-level
  Vidyut stream as the primary Pāṇinian source and **dissolves the ADR-0011
  clause-level limitation** (now a clause-level structural prior, not form-level).
- **Arm D auxiliary target upgraded** to the gold per-word kāraka role sequence
  (real sentence-level parse supervision), via `aux_targets`.
- **Real corpora provisioned** (`PaniniToolkitCorpusSource`): DCS (Apache-2.0),
  IndicCorp v2, Itihāsa, Sāmayik — all available under
  `PANINI_DATA_DIR=~/projects/slm-1/data`. These feed the NL continuation stream
  and tokenizer training for the Sanskrit side.
- Contradiction (richer data vs. stable locked design) resolved with TRIZ
  Principle 35 (Parameter Changes): only the unit *granularity* changed; arm
  isolation, token budgeting, tokenizer, and seeds are unchanged.

Known residual limitation (honest): the kāraka-frame enumerator currently emits
two frame templates (intransitive / transitive), so arm D's auxiliary spans two
role sequences. Extending to oblique kārakas (karaṇa/sampradāna/apādāna/
adhikaraṇa) is a cheap tracked follow-up and does not touch the decisive B-vs-C
comparison.

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

- **Provision corpora:** Sanskrit NL continuation is now satisfied by the real
  DCS/Itihāsa corpora via the toolkit (done). BabyLM English (research-only)
  still to be fetched for the English continuation/peer-benchmark side.
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
