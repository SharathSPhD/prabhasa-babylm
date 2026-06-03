# Dyck surface-stat match report (real Vidyut-realized targets)

Generated: `2026-06-03T08:11:26.229221+00:00`

**Targets computed directly from realized sentences (no proxy, no cached blob).**
Source: `data/fixtures/paninian_v1.jsonl (n=10000 realized sentences)`

## Targets (DEFAULT_KEYS)

| Key | Value |
|-----|-------|
| type_token_ratio | 0.011589 |
| bigram_entropy | 0.911361 |
| trigram_entropy | 0.986538 |

## Best grid match (H1 fairness) — gate PASS

| Field | Value |
|-------|-------|
| bracket_types | 35 |
| max_depth | 16 |
| n_shuffles | 1 |
| min_len | 8 |
| max_len | 96 |
| DEFAULT_KEYS distance | 0.009414 |
| threshold | 0.05 |
| L1 byte-length hist vs real corpus (informational, not gated) | 1.692693 |

Matched corpus stats: `{"n_sequences": 400.0, "type_token_ratio": 0.0033460803059273425, "bigram_entropy": 0.9110056766767779, "trigram_entropy": 0.9820062573788739, "pattern_coverage": 0.0}`

Matched config written to `configs/data/dyck_matched_36d4c762e39a.yaml` (config_hash `36d4c762e39a`).

## Hu et al. replication config (ADR-0025)

Pinned `hu_replication_config()`: k=35, max_depth=16,
n_shuffles=35, max_len=2048 (reference; not the H1 match).

| Metric | Value |
|--------|-------|
| Distance to targets (DEFAULT_KEYS) | 0.087894 |

Hu replication stats: `{"n_sequences": 400.0, "type_token_ratio": 0.0001734304543877905, "bigram_entropy": 0.9984589563184965, "trigram_entropy": 0.9895420213623596, "pattern_coverage": 0.0}`

## Artifacts

- Machine-readable: `docs/data/dyck-match-result.json`
- Matched training config: `configs/data/dyck_matched_36d4c762e39a.yaml`
