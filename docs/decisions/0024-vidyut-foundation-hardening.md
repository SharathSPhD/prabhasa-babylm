# ADR-0024: Vidyut / Saṃsādhanī foundation hardening (workstream/vidyut)

## Status

Accepted (2026-06-02). Wave-1 unit **U2** foundation; does not change
`AnnotatedSentence`, `DEFAULT_KEYS`, or `matrix.py`.

## Context

Downstream units (U3 Dyck matching, U5 Śabdabodha) need a reproducible,
well-tested generator foundation. ADR-0012 made Saṃsādhanī the **primary**
sentence-level kāraka source; Vidyut remains the offline derivation/morphology
engine (ADR-0010/0011). The interface freeze pins `AnnotatedSentence` and
`DEFAULT_KEYS` through 2026-06.

## Decision

1. **Producer contract tests** — Generators must emit non-empty `text`, coherent
   `has_gold_parse` / `karaka_parse`, and deterministic streams under fixed
   `seed` (see `tests/unit/test_generator_contract.py`).

2. **Property tests** — `hypothesis` guards `enumerate_frames` determinism,
   signature uniqueness, and oblique role coverage
   (`tests/unit/test_karaka_frames_hypothesis.py`).

3. **Saṃsādhanī hardening** — Default `fail_closed=True`, `dedup=True`, reject
   empty gold parses, expose `samsadhani_diversity_metrics()` for offline QA.

4. **Diversity gate (domain only)** — `psalm.domain.data.diversity_gate` validates
   TTR / bigram / trigram floors. **Not** wired into `matrix.py` or training.

5. **Frozen fixture corpus** — `scripts/export_vidyut_fixtures.py` writes
   `vidyut-fixture-v1` JSONL (≥10⁴ rows from kāraka frames) plus
   `data/fixtures/vidyut-fixtures-ci.jsonl` and `docs/data/vidyut-fixture-stats.json`.

6. **Vidyut smoke** — `@pytest.mark.vidyut` tests document `vidyut` wheel version
   on aarch64/x86_64 when installed.

## Consequences

- Integration branch wires `JsonlSentenceSource` to cached paths; full corpus
  regenerated via script, not committed at 10⁴ scale.
- `pyproject.toml` dev extra adds `hypothesis`; optional `vidyut` wheel remains
  environment-specific (ADR-0007).
