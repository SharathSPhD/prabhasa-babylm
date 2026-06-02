# ADR-0027 — Pre-register Paribhāṣā ASCII vs IAST tokenizer fertility ablation

- Status: Accepted
- Date: 2026-06-02
- Related: ADR-0020 (dual track); ADR-0018 (Paribhāṣā generator); ADR-0021 (`comp:*` namespace)

## Context

Competition Paribhāṣā strings default to **transliterated ASCII** for tokenizer stability
(`babylm-res-2`, ADR-0020). IAST diacritics increase Unicode cardinality and may raise
subword fertility (tokens/word), consuming budget under Strict-Small (10M words) and
Strict (100M words) without adding semantic content.

Before locking the joint BabyLM tokenizer (16–24k Strict-Small; 24–32k Strict per
ADR-0020), we need a **pre-registered** comparison on the same synthetic Paribhāṣā
templates rendered in ASCII vs IAST.

## Decision

1. **Primary script for competition tokenizer training:** ASCII Paribhāṣā (`renderer.py`
   default when wired; ablation uses `paribhasha_ascii_to_iast` only for measurement).

2. **Pre-registered ablation (U6):** On a fixed seed and line count (`n_lines=80`,
   `seed=13` in `run_paribhasha_fertility_ablation`), report:
   - `tokens_per_word` (ASCII)
   - `tokens_per_word` (IAST)
   - `delta = IAST − ASCII`

3. **Go/no-go for switching default to IAST:** If `delta > 0.05` tokens/word on the
   joint tokenizer trained on within-budget demo corpus, keep ASCII for submission
   unless a follow-up ADR documents improved BLiMP/EWoK worth the fertility cost.

4. **Documentation home:** `docs/data/babylm-tokenizer-fertility.md` (refreshed when
   manifest shards or vocab targets change).

## Consequences

- Fertility numbers in the doc are **smoke samples** until the manifest-frozen corpus
  is assembled; re-run `scripts/build_babylm_joint_tokenizer.py` before submission.
- H1 matrix and `comp:*` arms are unchanged; this ADR only affects tokenizer script choice.

## Alternatives considered

- **Devanāgarī third arm:** deferred — out of scope for BabyLM English eval target.
- **Skip ablation, assume ASCII:** rejected — human direction requires pre-registration.
