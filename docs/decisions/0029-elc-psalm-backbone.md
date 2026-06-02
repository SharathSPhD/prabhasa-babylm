# ADR-0029 — ELC-PSALM competition backbone (U7)

- Status: Accepted
- Date: 2026-06-02
- Depends on: ADR-0020, ADR-0023, ADR-0021
- Unit: U7 (`workstream/backbone`)

## Context

BabyLM competition entries need a HuggingFace-compatible encoder that scores strings
via pseudo-log-likelihood (PLL). Leaderboard winners ELC-BERT (2023) and GPT-BERT
(2024) motivate an **every-layer-counts** routing stack plus a **hybrid MLM+CLM**
objective. Wave-1 delivered the joint tokenizer (U6) and GB10 SDPA validation (U1).

## Decision

1. **Primary backbone:** `ElcPsalmEncoder` in `psalm.infrastructure.ml.elc_psalm`
   with learned lower-triangular per-layer routing (softmax over all prior hidden
   states) and pre-norm **torch SDPA** blocks (no flash-attn hard dependency).

2. **Sizes (config-driven, `elc_config.py`):**
   - **ELC-PSALM-S:** ~115M @ 20k vocab (Strict-Small band 90–140M).
   - **ELC-PSALM-M:** ~280M @ 28k vocab (Strict band 180–280M).

3. **Hybrid objective:** `HybridObjective` enum; default `HYBRID` with alternating
   MLM/CLM steps in `hybrid_training_step` (GPT-BERT pattern). Weights and
   `mlm_probability` live in `ElcPsalmConfig`, not in trainer magic numbers.

4. **BabyLM eval:** `ElcPsalmEvaluator.pseudo_log_likelihood(text)` implements
   `PseudoLogLikelihoodModel` via left-to-right single-token masking PLL.

5. **Integration deferred:** no changes to `__init__.py`, `trainer.py`, or HF export
   in this unit — wire in integration branch per interface-freeze.

## Consequences

- ML module excluded from mypy/coverage per existing `pyproject.toml` omit list.
- Domain `elc_config.py` is fully unit-tested for param bands and μP fields.
- HF `PreTrainedModel` wrapper and competition YAML presets are follow-up integration.

## Alternatives considered

- **Decoder-only BabyLlama path:** rejected as primary; spec §12 names ELC-PSALM encoder.
- **flash-attn in default stack:** rejected per ADR-0023 until GB10 container build is routine.
