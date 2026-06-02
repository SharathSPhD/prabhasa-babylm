# Śabdabodha pipeline coverage ledger

Pipeline: `src/psalm/infrastructure/generators/paribhasha/shabdabodha.py`  
ADR: [0019-shabdabodha-pipeline-full-vyutpattivada.md](../decisions/0019-shabdabodha-pipeline-full-vyutpattivada.md)  
Measured: 2026-06-02 (branch `workstream/shabdabodha`)

## Target (consolidation report §6 risk table)

| Stratum | Target | Notes |
|---|---|---|
| Saṃsādhanī / kāraka-frame fixtures (transitive + oblique) | ≥80% valid graphs | M1 export gate before training mix |
| Full Gadādhara scope | M3 per ADR-0019 | Out of scope for this wave; failures logged by `rule_id` |

## Measured coverage

| Corpus | *n* | Valid graphs | Fraction | Skips |
|---|---:|---:|---:|---|
| `data/fixtures/vidyut-fixtures-ci.jsonl` | 128 | 128 | **100.0%** | 0 |
| Kāraka frame enumerator (`enumerate_frames`, *n*=10⁴, seed=0) | 10,000 | 10,000 | **100.0%** | 0 |

The engine covers all kāraka roles in `karaka_frames.py`: `karwA`, `karma`, `karaNam`, `aXikaraNam`, `apAxAnam`, `sampraxAnam`. Constructions outside that inventory (unknown roles, missing dhātu, type violations) are **skipped** with a stable `rule_id`, not approximated.

## Vyutpattivāda rule ids (v1)

| Rule id | Stage |
|---|---|
| `VYU-001-dhatu-kriya` | Verbal root → `KRIYA` node |
| `VYU-002-karwA-samyogata` | Agent contact (`SAMYOGATA`) |
| `VYU-003-karma-visayata` | Object (`VISAYATA`) |
| `VYU-004-karaNam-prakarata` | Instrument (`PRAKARATA` on karma or kartā) |
| `VYU-005-adhikaraNa-prakarata` | Locus |
| `VYU-006-apadana-prakarata` | Source / separation |
| `VYU-007-sampradana-prakarata` | Goal |
| `VYU-008-avacchedaka-scope` | Viśeṣa scope on kriyā–kartā bundle |

`meta.rule_coverage` on each aligned row is the fraction of *applicable* rules successfully applied for that sentence.

## CI artifacts

- Aligned pairs: `data/fixtures/shabdabodha-aligned-ci.jsonl` (128 rows, schema-validated on export)
- Export stats: `docs/data/shabdabodha-export-stats.json`

Regenerate:

```bash
uv run python scripts/export_shabdabodha_pairs.py \
  --input data/fixtures/vidyut-fixtures-ci.jsonl \
  --out data/fixtures/shabdabodha-aligned-ci.jsonl
```

## Honest limits (not counted in numerator)

- Real DCS/GRETIL sentences without gold `karaka_parse` / `frame_signature`
- Non–kāraka-frame Saṃsādhanī constructions (causative, double-object, etc.)
- Inference-template graphs (U4 stratum 3 only)

Coverage on **represented** fixture strata exceeds the 80% program target; broader Vyutpattivāda scope remains incremental per ADR-0019 M3.
