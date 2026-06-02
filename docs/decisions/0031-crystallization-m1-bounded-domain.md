# 31. Crystallization Milestone 1 — bounded natural-kinds vocabulary proof

Date: 2026-06-02
Status: Accepted (Milestone 1 execution)

## Context

The crystallization charter (`docs/contracts/crystallization-track-charter-2026-06-01.md`)
requires a falsifiable, **bounded-domain** proof that the Sanskrit entity-vocabulary
layer (Attack-1) can be overcome before any Wikidata-wide build or Stage-2 training.
The Phase-2 spike established relation→kāraka coverage (~74% generable) but showed
the binding constraint is entity→WX-stem mapping, not relation typing.

## Decision

1. **Domain slice:** *Natural kinds + physical action* — animals, plants, elements,
   rivers, mountains, landforms, kinship/social roles; explicitly excluding modern
   technical entities (per charter §5).

2. **Knowledge source:** A **bundled curated seed lexicon**
   (`src/psalm/infrastructure/crystallization/data/natural_kinds_lexicon.json`) plus
   **deterministic in-domain triple enumeration** (taxonomy, geography, material,
   part-whole). No general Wikidata vocabulary pipeline in M1. Live SPARQL remains
   optional for future enrichment, not a gate dependency.

3. **Mapping stack:** Entity QID → WX stem + gender (lexicon); property ID → kāraka
   role via the shared `PROPERTY_MAP` (Attack-2, already validated); triple →
   `KarakaFrame` meaning structure with a lakāra/number/dhātu expansion grid for
   combinatorial yield.

4. **Realisation:** Saṃsādhanī via existing `SamsadhaniiGenerator._annotate` when the
   container is healthy; grammaticality audited as generator acceptance rate on a
   100-frame sample (charter §5 target 4).

5. **Verdict:** Measured PASS/FAIL against the four charter thresholds in
   `crystallization.pipeline` constants; honest shortfall is a valid M1 outcome.

## Consequences

- M1 code lives under `src/psalm/infrastructure/crystallization/` (CPU, light deps).
- `scripts/crystallization_spike.py` imports `PROPERTY_MAP` from the shared module
  so spike and M1 cannot drift.
- Expanding beyond the bounded domain or chasing missed targets by widening scope
  is **explicitly forbidden** by the charter until human sign-off on a new milestone.

## References

- Charter: `docs/contracts/crystallization-track-charter-2026-06-01.md`
- Spike: `docs/data/phase2-crystallization-spike-finding.md`
- Results: `docs/data/crystallization-m1-results.md`, `crystallization-m1-results.json`
