# Closure promise — BabyLM leaderboard, paper, dissemination (2026-06)

- Date: 2026-06-04
- Owner: central orchestrator (`integration/data-engine-v2`)
- Binds to: the six-layer Ralph closure contract (`closure-contract.md`) and ADR-0036/0037/0038.

This promise enumerates the objectives the program must satisfy to be "done to perfection",
the gate that proves each, and which steps are GPU-deferred behind the live H1 battery. The
orchestrator does not declare completion until every gate below is green (evidence, not
assertion).

## Objectives and gates

1. **Top-of-leaderboard build (OBJ-1).** Implement every leaderboard lever (ADR-0038) as
   tested code (adaptive/decaying + frequency-informed masking, Muon optimizer, progressive
   sequence length) behind a submission trainer profile; train a submission model at max
   budget once the GPU is free; report the full official Text-Average suite + (Super)GLUE.
   - Gate: levers unit-tested on CPU (parity where claimed); submission model official
     numbers recorded in the ledger and beating the arm-A baseline; target band the
     consolidation-report leaderboard numbers (BLiMP ~80, GLUE ~75, EWoK ~55 at 10M).

2. **Reconciliation (OBJ-2).** `docs/psalm-consolidation-report.md` reconciled against the
   current PRD/spec/ADRs/CLAUDE; BabyLM focus retained.
   - Gate: a reconciliation note maps every report gap to its current status; no stale
     "missing" claims remain unaddressed.

3. **Paper (OBJ-3a).** A journal-rigor manuscript: continuous prose (no bullet/numbered
   lists in the body), real tables, flowcharts and diagrams, real results, real citations
   (verifiable BibTeX, nothing fabricated), no plagiarism.
   - Gate: `paper/` compiles (or renders) with figures/tables populated from real result
     files; bibliography entries are real and checkable; prose passes a no-list lint.

4. **Dissemination site (OBJ-3b).** An Astro GitHub Pages site giving a multi-stakeholder
   (technical + general) account, deployed via CI.
   - Gate: `site/` builds; deploy workflow present; content covers both audiences and links
     to paper, models, datasets, and Colab.

5. **HF artifact collection (OBJ-3c).** Models, tokenizer, and dataset/corpus manifests
   published with proper cards under the `qbz506` namespace.
   - Gate: upload scripts run dry-clean; cards validate; collection links resolve.

6. **Colab demos (OBJ-4).** Runnable Colab notebooks with open-in-colab badges that load the
   published model and reproduce a zero-shot / fine-tune demo directly from the repo.
   - Gate: notebooks execute top-to-bottom against the published artifacts; badges in README
     open the notebooks from the repo.

## Sequencing (no-interruption order)

Non-GPU now (this session): closure promise + watcher/close-out automation; reconciliation;
levers code + CPU tests; paper draft with seed-0/arm-A real numbers; Astro site; Colab
notebooks; HF cards + upload scripts.

GPU-deferred (auto-triggered by `scripts/await_battery_and_closeout.sh` when the battery
finishes and the device frees): battery-wide official eval + (Super)GLUE on all checkpoints;
H1 paired-permutation analysis (Holm–Bonferroni); submission-model training; inject final
numbers into paper/site/HF; FF-merge worktrees; push at milestone; final sign-off.

## Battery status snapshot (update at each milestone)

- 12 runs = arms A–D × seeds 0–2 (seed-major). ~408 min train + ~20 min BLiMP per run.
- 2026-06-04 21:14: seed 0 complete (A reused; B/C/D trained). BLiMP-PLL A=0.6415, B=0.6354,
  C=0.6385, D=0.6377; best_loss D=0.90 (vs ~1.4 others). Run 5/12 (arm A seed 1) training.
- ETA full battery ≈ 2026-06-07 (~57h remaining at ~7.15h/run).

## Adversarial review log

Each milestone must record a self-challenge (what would falsify this? what is weakest?).
See `docs/memory/ORCHESTRATOR-STATE.md` for the live done/pending matrix.
