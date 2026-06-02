# Dyck surface-stat match report

**PENDING U2:** `vidyut-fixture-stats.json` is published but Dyck targets still use live Saṃsādhanī sentence stats until fixture-vs-live diversity parity is reconciled (fixture corpus TTR ≪ live).

Generated: `2026-06-02T21:39:40.342524+00:00`

## Targets (DEFAULT_KEYS)

| Key | Value |
|-----|-------|
| type_token_ratio | 0.2625 |
| bigram_entropy | 0.9856 |
| trigram_entropy | 0.9986 |

Source: `docs/data/phase2-samsadhani-diversity.json#samsadhani_sentences (U2 vidyut-fixture-stats.json present; fixture-vs-live TTR reconciliation pending)`

## Best grid match (H1 fairness)

| Field | Value |
|-------|-------|
| bracket_types | 5 |
| max_depth | 8 |
| n_shuffles | 3 |
| min_len | 16 |
| max_len | 192 |
| Euclidean distance (DEFAULT_KEYS) | 0.263193 |

Matched corpus stats: `{"n_sequences": 200.0, "type_token_ratio": 0.0004757826624797792, "bigram_entropy": 0.9858388690450198, "trigram_entropy": 0.9738258680062745, "pattern_coverage": 0.0}`

## Hu et al. replication config (ADR-0025)

Pinned `hu_replication_config()`: k=35, max_depth=16,
n_shuffles=35, max_len=2048.

| Metric | Value |
|--------|-------|
| Distance to targets (DEFAULT_KEYS) | 0.262496 |
| L1 byte-length hist vs matched corpus | 0.000000 |

Hu replication stats: `{"n_sequences": 200.0, "type_token_ratio": 0.00035014005602240897, "bigram_entropy": 0.997722867050758, "trigram_entropy": 0.9927234386051836, "pattern_coverage": 0.0}`

## Artifacts

- Machine-readable: `docs/data/dyck-match-result.json`
