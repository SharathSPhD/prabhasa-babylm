# ADR-0034 — Faithful Vyutpattivāda: lossless Paribhāṣā render, real semantics, dual-task

- Status: **Proposed** (core-rebuild reframe 2026-06; human sign-off at GATE 0).
- Date: 2026-06-03
- Supersedes: the skeleton scope of ADR-0018 (typed generator) and ADR-0019
  (Śabdabodha pipeline) where they permitted an isomorphic kāraka relabel.
- Depends on: ADR-0033 (Vidyut gold-kāraka sentences), ADR-0035 (information-parity gate).

## Context

Adversarial review established that the current L2 stack is scaffolding that cannot
test the scientific claim, regardless of scale:

- **Renderer drops the object role.** `renderer.py:147-165` suppresses edges under
  `AVACCHEDAKA`, removing `VISAYATA` (karma) from **88/128** transitive strings.
- **Vyutpattivāda is bijective relabel.** `shabdabodha.py` (8 rules) → **2 templates /
  128 strings**, H(structure | kāraka) ≈ 0. No information beyond arm B.
- **Default path trains on Sanskrit, not Paribhāṣā** (`assembly.py:34`).
- **~5-10% of Gadādhara modeled**: ākāṅkṣā, yogyatā, āsatti, tātparya, nested viṣayatā,
  mukhya-viśeṣyatā, padārtha assignment, viśeṣaṇa→viśeṣyatā all absent; all nominals
  forced to DRAVYA; AVACCHEDAKA always synthetic on kartā.

A typed semantic prior that adds nothing over the kāraka parse, and then renders away
the one relation that matters, is guaranteed to tie Dyck. The fix is faithfulness, not
scale.

## Decision

### D1 — Lossless linearization (blocking, P0)

Every graph edge — including `VISAYATA`, all obliques, and inner edges under
`AVACCHEDAKA` — must appear in `paribhasha_string`. Add round-trip property tests on
graphs that combine `AVACCHEDAKA` + `VISAYATA` + `SAMYOGATA`; a render that loses any
edge fails CI. `AVACCHEDAKA` must scope the correct prakāra–viśeṣya bundle (Gadādhara
limiter), not "first inner edge only."

### D2 — Train on Paribhāṣā, not Sanskrit (blocking, P0)

`serialize_line` gains a branch for `PrePretrainSource.SHABDABODHA_ALIGNED` /
`PARIBHASHA` that emits `paribhasha_string` (ASCII channel per ADR-0026; `paribhasha_iast`
available for ablation) — never `sentence.text`. The H1′ workaround source is deleted.

### D3 — Real Vyutpattivāda semantics (substantive)

The engine must derive distinctions the kāraka parse does not already encode:

1. **Padārtha assignment** from morphology/lexicon (dravya / guṇa / kriyā / sāmānya /
   viśeṣa / abhāva), not "all nominals → DRAVYA."
2. **Viśeṣaṇa → viśeṣyatā / prakāratā** from actual modifiers (adjectives, numerals,
   compounds) present in the sentence, not a synthetic `scope_{kartā}` boilerplate.
3. **Kāraka-specific topology**: adhikaraṇa as locus of the kriyā; apādāna as separation
   source; sampradāna as goal — distinct graph shapes per Śabdabodha commentary, not one
   `PRAKARATA(guna_tag, dravya_self)`.
4. **Comprehension gates** before emission: **ākāṅkṣā** (expectancy), **yogyatā**
   (semantic fitness), **āsatti** (proximity). Ill-formed combinations are skipped with
   a logged `rule_id` (no fabrication).
5. **Coverage ledger** per `rule_id` over construction classes (transitive, intransitive,
   ditransitive, locative, copular, …), not just the narrow Saṃsādhanī frame inventory.

### D4 — Dual-task aligned supervision

Train on `(Sanskrit sentence ↔ Śabdabodha graph / paribhasha_string)` pairs
(`paribhasha_aligned_v1`, schema `docs/contracts/aligned-pair-schema.json`), exposing
the graph for a dual-task head — not Paribhāṣā strings whose stems are isomorphic to
arm B. Standalone L2 corpus draws open vocabulary tied to the padārtha inventory, not
the ~17-label toy lexicon in `generator.py:32-40`.

### D5 — Information-parity precondition (hard gate, see ADR-0035)

Before any GPU training on the L2 stream, the workstream must show on 10⁴ frames:
`H(structure | kāraka) ≫ 0`, `VISAYATA` present in **100%** of transitive
`paribhasha_string`, and template count ≫ 2. If `H(structure | kāraka) ≈ 0`, the stream
is a relabel and the run is forbidden.

## Consequences

- `shabdabodha.py`, `renderer.py`, `generator.py`, `relations.py`, `ontology.py`,
  `types.py` are substantially rebuilt, not patched.
- ADR-0019 "full Vyutpattivāda" claim is staged behind an explicit M1→M3 rule inventory;
  results are not labeled "Vyutpattivāda" until M-level is published.
- Graph-consistency scoring (ADR-0030 secondary) must corrupt **all** sansa slots,
  including `VISAYATA` on karma.

## Alternatives considered

- **Patch the renderer only:** removes the object-role bug but leaves the relabel
  (H(structure|kāraka)≈0) — insufficient.
- **Keep toy generator for "structural warmup":** that is Dyck's job (L0); duplicating
  it as L2 confounds the contrast — rejected.

## Links

- Code: `src/psalm/infrastructure/generators/paribhasha/*`, `application/data/assembly.py`
- Schema: `docs/contracts/aligned-pair-schema.json`
- Evidence: `docs/experiments/h1prime-invalidation-2026-06.md`
- Measurement gate: ADR-0035; pre-registration: ADR-0030 (refresh).
