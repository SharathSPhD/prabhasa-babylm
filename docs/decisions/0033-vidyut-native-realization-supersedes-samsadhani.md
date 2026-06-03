# ADR-0033 — Vidyut-native realization supersedes the Saṃsādhanī dependency

- Status: **Proposed** (core-rebuild reframe 2026-06; human sign-off required at GATE 0).
- Date: 2026-06-03
- Supersedes: ADR-0011 (form-level Vidyut limitation) and the realization role of
  ADR-0012 (Saṃsādhanī sentence-level kāraka unit). Saṃsādhanī is **demoted to an
  optional offline cross-check**, not a runtime dependency.
- Depends on: ADR-0010 (open Pāṇinian generator + license-clean corpora), ADR-0022
  (GB10 native stack), ADR-0035 (GPU-only, no-mock).

## Context

The crystallization "M1" finding ("0 unique live sentences → realization is the
binding constraint") is a measurement artifact, not a linguistic ceiling:

- The published `0` came from `--skip-generation` / `generator_configured=false`
  (`docs/data/crystallization-m1-results.json`), **not** from Saṃsādhanī rejecting
  frames. A live probe accepts **~92%** (and 100% on the bounded lexicon); the ~8%
  rejects are **verb-transitivity mismatches** (akarmaka root paired with `karma` in
  `relation_map.ROLE_DHATUS`), fixable independent of engine choice.
- `VidyutGenerator` (`src/psalm/infrastructure/generators/vidyut_source.py`) emits
  **isolated tiṅanta with empty `karaka_parse`** — it cannot produce sentences or the
  gold structural signal. The only sentence-level gold-kāraka source is Saṃsādhanī, an
  external Docker service (fail-closed, unprovisionable on GB10/CI).
- `vidyut.prakriya` (verified, `vidyut==0.4.0` on host) supports **subanta** (nominal
  declension), **tiṅanta** (verb), and transliteration; sūtra-by-sūtra `prakriya.history`
  is free gold derivation.

Therefore the correct realization engine is **Vidyut, used natively at the sentence
level** — composing each pada and emitting gold kāraka by construction.

## Decision

### D1 — `VidyutFrameRealizer` is the primary realization port

Implement `VidyutFrameRealizer` against `application/data/ports.SentenceGenerator`:
`realize(frame: KarakaFrame) -> AnnotatedSentence | None` and `stream(n, seed)`.
For each frame:

1. Generate each **nominal** pada via `Pada.Subanta(Pratipadika.basic(stem), linga,
   vibhakti, vacana)` and the **verb** via `Pada.Tinanta(...)` through `Vyakarana.derive`.
2. Order padas deterministically (default kartā→karma→obliques→verb; `word_order`
   configurable for free-order robustness), seeded by `frame.signature + seed`.
3. Apply forward sandhi (D3) and emit `AnnotatedSentence(text, karaka_parse, derivation, meta)`.

### D2 — Role → vibhakti map (gold kāraka by construction)

| kāraka (WX) | Vibhakti |
|---|---|
| `karwA` | Prathama |
| `karma` | Dvitiya |
| `karaNam` | Trtiya |
| `sampraxAnam` | Caturthi |
| `apAxAnam` | Panchami |
| `aXikaraNam` | Saptami |

`number`→Vacana (eka/dvi/bahu), `gender`→Linga (puM/strI/napuM). The verb's
puruṣa/vacana agree with the kartā nominal; `prayoga` from frame (`Kartari`/`Karmani`).
`karaka_parse` is the set of `(inflected_surface, role)` pairs assembled from the frame
— **not** parser output — so `has_gold_parse` is true iff every content word has a role.

### D3 — Forward sandhi without mocks

`vidyut.sandhi` is a **splitter**, not a joiner. Phase-1 sandhi uses an **inverted
rules table** built from `vidyut-data` sandhi CSV: lookup `(prev_final, next_initial) →
fused`. On a miss, emit padas space-separated with `meta["sandhi"]="partial"` — never
fabricate a fused form. Document deviation from Saṃsādhanī prose surfaces. (Phase-2:
adopt a forward prakriya sandhi API if upstream adds one.)

### D4 — Grammaticality without Saṃsādhanī

Acceptance criteria, all native/deterministic:
1. **Paninian derivability** — every pada has non-empty `prakriya.history`.
2. **Agreement validator** (pure Python) — kartā linga/vacana vs verb prayoga/vacana.
3. Optional cross-checks: `vidyut.kosha` stem membership; DCS morphology sample;
   optional offline Saṃsādhanī probe (no longer on the critical path).
Reject only on empty `derive()` or unknown dhātu/lakāra — **target ~100% acceptance**
on frames Vidyut can derive.

### D5 — `ROLE_DHATUS` transitivity fix

Annotate dhātus with transitivity; never pair an akarmaka root with `karma`. This alone
recovers the ~8% live Saṃsādhanī rejects and is required for the Vidyut realizer too.

### D6 — Saṃsādhanī demotion

Saṃsādhanī becomes an optional, offline cross-check (`is_configured` may be false in
all default paths). No PSALM training, crystallization, or fixture-export path may
depend on the Docker container being up.

## Consequences

- Crystallization realization is rewired to `VidyutFrameRealizer` (remove the Docker
  gate from M1). `PrePretrainAssembler(paninian=VidyutFrameRealizer())`.
- Frozen fixture `data/fixtures/paninian_v1.jsonl` (≥10⁴ lines, gold kāraka, manifest
  hashes) becomes the reproducible L1 stream feeding the paribhāṣā and dyck workstreams.
- `vidyut_source.py` docstring corrected (it does not "replace Saṃsādhanī" as a sentence
  generator; the realizer does).

## Alternatives considered

- **Keep Saṃsādhanī, just fix `ROLE_DHATUS`:** recovers yield but retains an
  unprovisionable Docker dependency on GB10/CI — rejected as primary.
- **Vidyut verbs-only (status quo):** no gold kāraka, cannot feed arm B/D — rejected.

## Links

- Realizer port: `src/psalm/application/data/ports.py`
- Frames: `src/psalm/domain/data/karaka_frames.py`, `…/crystallization/frames.py`
- Sandhi data: `vidyut-data` sandhi rules; `src/psalm/domain/data/sandhi.py`
- GPU/no-mock: ADR-0035; GB10: ADR-0022.
