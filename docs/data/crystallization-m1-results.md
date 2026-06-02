# Crystallization Milestone 1 — bounded-domain results

**Domain:** natural_kinds_physical_action
**Overall verdict:** PARTIAL — see charter_verdicts

## Mapping (Attack-1 vocabulary layer)

- Lexicon entries: **126**
- Candidate entity–entity triples: **124448**
- Both endpoints mapped: **124448**
- **Net mapping rate:** 56.4%

## Crystallization yield

- Distinct kāraka frames enumerated: **120,000** (charter sentence target: 100,000)
- Unique annotated sentences (Saṃsādhanī): **0**
- No-repeat token count: **0**
- Generator: frames_only (configured=False)

## Charter targets (PASS/FAIL)

| Target | Required | Measured | PASS |
|--------|----------|----------|------|
| unique_crystallized_sentences | >= 100,000 | 0 | FAIL |
| net_mapping_rate | >= 25% | 56.4% | PASS |
| no_repeat_token_ceiling | >= 10× 179,876 | 0 | FAIL |
| spot_grammaticality | >= 80% on 100 | skipped (generator offline) | FAIL |

## Frame coverage (kāraka role patterns)

- `karwA`: 100.0%
- `karwA karma`: 40.5%
- `karwA aXikaraNam`: 40.0%
- `karwA karaNam`: 14.1%
- `karwA apAxAnam`: 2.7%
- `karwA sampraxAnam`: 2.7%

## Limitations

- Generation skipped (--skip-generation).

Machine-readable: `docs/data/crystallization-m1-results.json`
