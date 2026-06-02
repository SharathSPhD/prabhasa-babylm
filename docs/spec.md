# PSALM specification (reframe 2026-06)

Authoritative system and experimental specification. PRD: `docs/prd.md`. Interface
freeze for Wave 1: `docs/contracts/interface-freeze-2026-06.md`.

## 1. Hexagonal boundaries

```
domain/          Pure logic: experiments, contracts, data stats, Paribhāṣā types (future)
application/     Use cases: assembly, ports (AnnotatedSentence), orchestration
infrastructure/  torch, generators, HF, SQLite ledger, BabyLM runner, Z3
```

Dependencies point inward only. `domain/` has no torch/I/O. Generators implement
`SentenceGenerator` / `CorpusSource` from `psalm.application.data.ports`.

## 2. Two arm matrices (namespaced)

### 2.1 H1 research matrix (frozen — `matrix.py`)

Canonical IDs **`A`–`H`** in `src/psalm/domain/experiments/matrix.py`. **Do not
rename** (closed Phase-2 ledger, tag `h1-proxy-null-2026-06`).

| Arm | Pre-pretrain | NL budget | Role |
|---|---|---|---|
| A | none | 100M | Baseline |
| B | paninian | 100M | H1 treatment (closed null vs C) |
| C | dyck | 100M | Primary control |
| D | paninian_karaka_aux | 100M | Aux-loss ablation |
| E | paninian | 10M | Data-efficiency |
| F | dyck | 10M | Low-budget control |
| G | none | 10M | Low-budget baseline |
| H | paninian_scrambled | 100M | Structure vs tokens |

Decisive pair: **B vs C**. Config root: `configs/phase2/`. Ledger `arm_id`: bare letter.

### 2.2 Competition matrix (`comp:*`)

Namespace: **`comp:{track}:{arm}`** with `track ∈ {strict_small, strict}` (ADR-0021).

| comp arm | Pre-pretrain | Pretrain | Eval emphasis |
|---|---|---|---|
| A | none | English | BabyLM baseline |
| B | paninian | English | Grammar prior |
| C | dyck | English | Formal control |
| D | **paribhasha** | English | **H1′ treatment** |
| E | paninian | Sanskrit + English | Multi-target |
| F | paninian + paribhasha | Sanskrit + English | Full stack |

Config root: `configs/competition/{track}/`. Implementation:
`competition_matrix.py` on `integration/data-engine-v2` only.

**Collision warning:** `comp:*:D` is Paribhāṣā; H1 `D` is kāraka auxiliary loss.

### 2.3 H1′ research matrix (`h1p:*`)

Future pre-registered battery: **`h1p:{arm}`** e.g. `h1p:B` Paribhāṣā, `h1p:C` Dyck,
matched on manifest tokens and architecture. Config root: `configs/research/h1p/`.

## 3. Data ports

### 3.1 `AnnotatedSentence` (frozen)

Defined in `src/psalm/application/data/ports.py` — see interface-freeze doc for pinned
fields: `text`, `language`, `karaka_parse`, `derivation`, `meta`, `has_gold_parse`.

### 3.2 Generators (existing)

| Module | Source enum | Output |
|---|---|---|
| `dyck_source.py` | `DYCK` | Bracket strings |
| `samsadhani.py` / `vidyut_source.py` | `PANINIAN`, aux, scrambled | Sanskrit + gold |
| `scramble_source.py` | (arm H path) | Permuted tokens |
| `jsonl_source.py` | fixtures | Replay frozen JSONL |

### 3.3 `PrePretrainSource` extension process

Current enum in `src/psalm/domain/experiments/models.py`:

`none | paninian | dyck | paninian_karaka_aux | paninian_scrambled`

**Additive extension (Wave 1 workers):**

1. ADR approving value (e.g. `paribhasha`).
2. Register in `src/psalm/domain/experiments/source_extensions.py` (planned stub).
3. Wire generator in `application/data/assembly.py` on integration branch only.
4. **Never** edit `matrix.py` in unit worktrees.

### 3.4 Aligned pair schema `paribhasha_aligned_v1`

JSONL records for L2/L5 — schema at `docs/contracts/aligned-pair-schema.json`.

## 4. Paribhāṣā module spec (consolidation §5.1–5.3)

Path: `src/psalm/infrastructure/generators/paribhasha/`

```
types.py       PadarthaCategory, SansaType, Operator + validation rules
ontology.py    sapta-padārtha
relations.py   VISAYATA, PRAKARATA, VISESYATA, AVACCHEDAKA, …
inference.py   PAKSA, HETU, VYAPTI, UDAHARANA, UPANAYA, NIGAMANA
generator.py   strata 1–3 emission + optional synthetic aligned pairs
renderer.py    graph → canonical Paribhāṣā string (ASCII default)
```

**Type rules (examples, enforced in code):**

- `DRAVYA`: anuyogin/pratiyogin in `SAMAVAYA`
- `GUNA` / `KRIYA`: prakāratā holders per relation kind
- `ABHAVA`: requires anuyogin + pratiyogin
- `AVACCHEDAKA`: scopes any relation

**Strata:**

1. Atomic relations.
2. Nested depth 2–4.
3. Inference templates.
4. Aligned pairs (U5) → `paribhasha_aligned_v1`.

## 5. Śabdabodha pipeline spec (§5.4, full Vyutpattivāda)

```
AnnotatedSentence (Sanskrit)
  → kāraka parse (gold or Vidyut)
  → Vyutpattivāda engine (rule ids, coverage ledger)
  → shabdabodha_graph (typed JSON)
  → renderer → paribhasha_string
  → paribhasha_aligned_v1 export
```

**Coverage:** fixture strata ≥90% before training mix; full Gadādhara/Vyutpattivāda scope
per ADR-0019 with per-rule failure logging. U5 Wave 2.

## 6. Curriculum and BabyLM budget accounting

- All pretrain-visible text counts toward cap (`babylm-res-2`).
- `corpus_manifest.yaml`: per-source `word_count`, `dedup_hash`, `epochs`, `license`.
- Strict-Small illustrative mix (strategy doc): ~6.5M EN, ~2M Paribhāṣā, ~1M Pāṇinian,
  ~0.5M Dyck — finalize in U6 manifest, not hard-coded in domain.
- Temperature schedule: L0–L2 prefix → L4 dominant (annealing).

## 7. Evaluation suites

| Suite | Track | Role |
|---|---|---|
| Official BabyLM pipeline | Competition | BLiMP, GLUE, EWoK, compositional (U6) |
| COGS / SCAN | Research secondary | Historical H1 venue; not go/no-go |
| Sanskrit competence | Research | Morphology, sandhi, kāraka |
| EWoK + BLiMP (argument-structure subsets) | H1′ primary candidates | Paribhāṣā vs Dyck |
| Śabdabodha graph consistency | H1′ / L2 | Parse ↔ graph round-trip metrics |
| H2 Nyāya / H3 kernel | Phase 4–5 | Per phase YAML |

Pseudo-log-likelihood output required for BabyLM classifier models (ELC-PSALM).

## 8. Pre-registered go/no-go registry

| Claim | Metric home | Threshold status |
|---|---|---|
| H1 B vs C (COGS SE) | Closed | NULL @ proxy — ADR-0017 |
| H1′ Paribhāṣā vs Dyck | TBD in `configs/research/h1p/go_no_go.yaml` | **Human sign-off before battery** |
| H2 synergy | ADR-0006 | Open |
| H3 fluency cost | Phase 5 prereg | Open |
| C-S2 efficiency | Crystallization charter | N ≥ 3 at Milestone 2 |
| BabyLM submission | Competition checklist | Manifest + official eval pass |

Changing thresholds → superseding ADR.

## 9. Ledger and artifact schema

- SQLite: `src/psalm/infrastructure/ledger/sqlite_ledger.py`
- Human mirror: `docs/experiments/schema.sql`
- Row key: `run_id`, `arm_id` (namespaced for new work), `config_hash`, `seed`, `attempt`, `metrics` JSON
- Knowledge store: `knowledge_store.py` (vectors/SQLite)

Closure reports: `docs/contracts/phase-*.yaml` + `psalm contract check`.

## 10. Closure-contract mapping

| Phase | Contract file | Empirical focus |
|---|---|---|
| 0 | `phase-0.yaml` | Foundation / stack |
| 1 | `phase-1.yaml` | Data engine, tokenizer |
| 2 | `phase-2.yaml` | H1 battery — **closed** |
| 3+ | `phase-3.yaml` … | H1′, scale, multi-target |
| Crystallization | charter MD | C-S2 milestones CPU-first |

Each phase: TECHNICAL → EMPIRICAL → INTEGRITY (Tarka) → ARTIFACTS → MEMORY → SIGN-OFF.

## 11. Dyck matching (U3)

`match_dyck` in `src/psalm/domain/data/matching.py` — `DEFAULT_KEYS` frozen in
interface-freeze. U3: recompute targets vs **sentence-level** Saṃsādhanī stats;
document Hu et al. ACL'25 replication ADR; verify arm-H scramble fairness.

## 12. ELC-PSALM backbone (U7)

ELC-BERT every-layer-counts + GPT-BERT hybrid; competition primary. Depends on U6
tokenizer and U1 verified container. Sizes: 90–140M (Strict-Small), 180–280M (Strict).

## 13. Constraints

Single GB10; uv; config-driven (`configs/*.yaml`); coverage ≥80%; citation integrity
(ADR-0008). No fabricated arXiv:2605.12548.

## 14. Out of scope

350M COGS rescue; `matrix.py` edits outside integration branch; reusing `comp` letters
in `phase2` configs; cloud training unless ADR-0005 exception.
