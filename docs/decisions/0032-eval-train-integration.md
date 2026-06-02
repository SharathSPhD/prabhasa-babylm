# ADR-0032 — Eval-train integration (ELC smoke loop + local PLL harness)

- Status: Accepted
- Date: 2026-06-02
- Depends on: ADR-0020, ADR-0023, ADR-0029
- Unit: eval-train (`workstream/eval-train`)

## Context

Wave-2 merged `ElcPsalmEncoder` and a mock BabyLM eval adapter. H1′ and the
BabyLM competition track need **real** PLL minimal-pair scores and a bounded
train→checkpoint→eval path on GB10 (SDPA, no flash-attn hard dep).

## Decision

1. **Default eval model** is untrained or checkpoint **ELC-PSALM** (`--mock` opt-in
   for random baseline wiring tests only).

2. **Local zero-shot harness** scores built-in minimal pairs via
   `minimal_pair_accuracy` (PLL good vs bad). Official pipeline install enables
   `invoke_official_zero_shot` but does not replace the local smoke path.

3. **Training** uses `train_elc_encoder` with `architecture: elc_psalm_s|m`,
   `TokenPacker.packed_batches`, and `hybrid_training_step`. Smoke mode shrinks
   widths for CPU CI; GPU uses torch SDPA per ADR-0023.

4. **HF export** is a minimal `ElcPsalmForMaskedLM` scaffold for future full-suite
   runs; tokenizer + Hub upload remain TODO.

## Consequences

- `RunMode` renamed: `local` / `official` / `mock` (was live/mock conflation).
- ML modules stay mypy/coverage-omitted; domain `architecture.py` helpers are tested.
- Full competition pretrain still blocked on joint tokenizer artifact paths in YAML.

## Alternatives considered

- **Vendoring evaluation-pipeline in git:** rejected — pin script + gitignored clone.
- **Decoder trainer reuse for ELC:** rejected — hybrid objective needs packed MLM batches.
