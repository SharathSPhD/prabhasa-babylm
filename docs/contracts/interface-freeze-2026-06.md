# Interface freeze — 2026-06 (Wave 1)

**Status:** Proposed — human sign-off required before Wave-1 worktree charters.

Wave-1 units (U1, U2, U3, U4, U6) **must not break** the surfaces below. Changes
require an ADR + revision of this document.

## 1. `AnnotatedSentence` (application port)

**Source of truth:** `src/psalm/application/data/ports.py`

| Field | Type | Semantics |
|---|---|---|
| `text` | `str` | Surface training string |
| `language` | `str` | ISO 639 default `"sa"`; `"en"` for BabyLM streams |
| `karaka_parse` | `tuple[tuple[str, str], ...]` | `(token, karaka_role)` gold pairs; empty if N/A |
| `derivation` | `tuple[str, ...]` | Ordered sūtra ids / derivation steps; may be empty |
| `meta` | `dict[str, str]` | Provenance, `source`, `rule_id`, `fixture_id`, etc. |

**Frozen properties:**

- Dataclass remains `frozen=True`.
- `has_gold_parse` stays `bool(karaka_parse)`.
- No removal or rename of fields without ADR + major version bump (`AnnotatedSentence_v2`).

**Allowed:** new optional keys inside `meta` only.

## 2. `PrePretrainSource` enum + extension process

**Source of truth:** `src/psalm/domain/experiments/models.py`

**Frozen values (H1 closure):**

```
none | paninian | dyck | paninian_karaka_aux | paninian_scrambled
```

**Additive extension process (not optional for new priors):**

1. New ADR (e.g. ADR-0018 for `paribhasha`).
2. Document the value in `src/psalm/domain/experiments/source_extensions.py` (registry
   module — **no runtime side effects**, import-safe).
3. Implementation lands on branch `integration/data-engine-v2` only:
   - enum member in `models.py`
   - `assembly.py` generator branch
   - competition matrix entries — **not** retroactive edits to closed H1 arms
4. Parallel unit worktrees **must not** patch `matrix.py`.

**Planned registry stub:** `src/psalm/domain/experiments/source_extensions.py`

## 3. `match_dyck` / `DEFAULT_KEYS`

**Source of truth:** `src/psalm/domain/data/matching.py`

```python
DEFAULT_KEYS: tuple[str, ...] = (
    "type_token_ratio",
    "bigram_entropy",
    "trigram_entropy",
)
```

**Frozen:**

- Key names and order for H1′/competition matching unless ADR + recomputation of all
  stored Dyck configs.
- `stat_distance` remains Euclidean over the selected keys.
- `match_dyck(..., keys=DEFAULT_KEYS)` default signature.

**U3 may:** recompute `targets` from sentence-level Saṃsādhanī stats; append ADR for
Hu et al. replication; **must not** silently drop keys from published H1 configs.

## 4. `paribhasha_aligned_v1` JSONL schema

**Normative JSON Schema:** `docs/contracts/aligned-pair-schema.json`

Each JSONL line is one object:

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Sanskrit surface sentence |
| `karaka_parse` | array of `[token, role]` | yes | Gold or pipeline kāraka |
| `shabdabodha_graph` | object | yes | Typed semantic graph (see schema) |
| `paribhasha_string` | string | yes | Rendered Paribhāṣā line |
| `meta` | object | yes | `schema_version`, `source`, `rule_coverage`, `seed`, … |

**Frozen:** field names and `meta.schema_version = "paribhasha_aligned_v1"`.

**Producers:** U5 pipeline; U4 may emit synthetic rows for tests only (`meta.synthetic=true`).

## 5. Arm IDs in configs and ledger (cross-reference)

- H1: bare `A`–`H` only.
- Competition: `comp:{track}:{arm}`.
- H1′: `h1p:{arm}`.

See ADR-0021. Wave-1 code must not write bare `D` for Paribhāṣā.

## 6. What Wave-1 may change

| Area | Allowed |
|---|---|
| New generator packages under `infrastructure/generators/` | yes (U4) |
| Fixture JSONL under `data/fixtures/` | yes (U2) |
| `infra/dgx_spark/*` | yes (U1) |
| BabyLM eval wrapper | yes (U6) |
| `matrix.py` | **no** (integration branch only) |
| `AnnotatedSentence` fields | **no** |
| `DEFAULT_KEYS` | **no** without ADR |

## 7. Acceptance of freeze

Checklist for human sign-off (see `reframe-2026-06-foundation.md`):

- [ ] AnnotatedSentence pin approved
- [ ] Extension process approved
- [ ] DEFAULT_KEYS pin approved
- [ ] `paribhasha_aligned_v1` schema approved
- [ ] Arm namespaces approved (ADR-0021)
