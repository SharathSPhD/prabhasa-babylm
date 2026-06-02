# ADR-0026 — Paribhāṣā renderer dual channel (ASCII + IAST)

- Status: Accepted
- Date: 2026-06-02
- Depends on: ADR-0018 (L2 typed generator); interface freeze `paribhasha_aligned_v1`

## Context

Competition tokenizer work (babylm-res-5) recommends ASCII Paribhāṣā for stability,
while Indology / publication tracks need IAST diacritics. U5 (Śabdabodha) and U6
(manifest) must not fork two incompatible linearizations.

## Decision

1. `render_graph()` returns `RenderedParibhasha(ascii=..., iast=...)` with the
   **same tree structure** in both channels; only operator spellings differ
   (e.g. `PRAKARATA` vs `prakāratā`).
2. `paribhasha_aligned_v1.paribhasha_string` remains a **single** string per row;
   producers set it from `.ascii` by default; manifests may record `meta.encoding`
   when shipping IAST.
3. Round-trip validation uses the **ASCII** channel and a hand-written parser in
   `renderer.py` (no new runtime dependency on Lark).

## Consequences

- Tokenizer ablations compare ASCII vs IAST without changing graph types.
- U5 imports `ShabdabodhaGraph`, `render_graph`, `validate_graph` only.

## Alternatives considered

- **Lark grammar:** deferred — hand-written parser sufficient for v1 operator inventory.
- **IAST-only:** rejected — breaks BabyLM tokenizer stability guidance.
