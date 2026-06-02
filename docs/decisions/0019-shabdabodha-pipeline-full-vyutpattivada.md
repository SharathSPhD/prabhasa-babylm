# ADR-0019 — Śabdabodha pipeline: full Vyutpattivāda coverage target

- Status: Accepted
- Date: 2026-06-02
- Depends on: ADR-0018; ADR-0012 (sentence-level kāraka unit); U2 frozen fixtures ≥10⁴
- Inputs: `slm-1/docs/psalm-consolidation-report.md` §5.4; Ghushe transcript (slm-1)

## Context

Gadādhara's Vyutpattivāda provides a deterministic map from parsed Sanskrit to a
Śabdabodha semantic graph and then to a linearized Paribhāṣā string. That pipeline
is the "free gold annotation" mechanism: every Saṃsādhanī/Vidyut sentence can yield
`(text, karaka_parse, shabdabodha_graph, paribhasha_string)` without human labels.

The repo implements sentence-level Pāṇinian generation (`samsadhani.py`,
`karaka_frames.py`) but has **no** Śabdabodha rule engine. A "simplified 80%"
pilot was considered during consolidation; human direction (2026-06) locks **full
Vyutpattivāda coverage** (broad Gadādhara / Vyutpattivāda scope) as the target,
implemented incrementally with explicit coverage accounting — not a permanent 80%
cap.

## Decision

1. **Pipeline stages (ordered, frozen I/O):**
   - Input: `AnnotatedSentence` from U2 fixtures or live Saṃsādhanī stream.
   - Step 1: kāraka parse (existing gold or Vidyut refresh).
   - Step 2: Vyutpattivāda rule engine → `shabdabodha_graph` (typed nodes/edges).
   - Step 3: `renderer` → `paribhasha_string` (canonical ASCII/IAST).
   - Output: `paribhasha_aligned_v1` record (JSON schema in contracts).

2. **Coverage policy:** maintain a versioned **coverage ledger** (% of fixture
   sentences with valid graphs, by construction class). Milestones:
   - M1: ≥10⁴ fixture sentences exported (U2).
   - M2: ≥90% coverage on **fixture strata** used in training mixes.
   - M3: full Vyutpattivāda rule set for Gadādhara/Vyutpattivāda scope as defined
     in the Śabdabodha spec appendix (U5 charter) — failures logged per rule id,
     not silently dropped.

3. **Wave placement:** U5 runs in **Wave 2** after U2 fixtures + U4 types/renderer.

4. **Dual-task training:** sentence ↔ graph alignment is a first-class objective
   alongside LM pretraining on Paribhāṣā strings (competition + research tracks).

## Consequences

- U5 is the longest pole; Wave 1 must not block on full coverage, but must not
  ship competition mixes claiming "full Vyutpattivāda" until M3 criteria met.
- Invalid graphs are **excluded with logged rule id**, not approximated — preserves
  scientific honesty.
- ISCLS-facing paper track: "first automated Vyutpattivāda annotation pipeline for LM
  training" remains valid if coverage ledger is published.

## Alternatives considered

- **Permanent 80% simplified rule set:** rejected by human direction.
- **Manual Śabdabodha annotation:** rejected — scale and closure discipline require
  deterministic automation.
- **Graph-only pretrain without Sanskrit sentences:** rejected — breaks aligned-pair
  schema and multi-target identity.

## Links

- `docs/contracts/aligned-pair-schema.json`
- Crystallization (orthogonal): `docs/contracts/crystallization-track-charter-2026-06-01.md`
