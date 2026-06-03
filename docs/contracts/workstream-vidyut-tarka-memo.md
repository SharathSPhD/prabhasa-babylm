# Tarka memo — `vidyut-realize` standalone closure (ADR-0033)

Tarka (rigorous objection) for the Vidyut-native frame realizer, per the
integrity layer of `workstream-vidyut-acceptance.yaml`.

## Strongest objection

> "`sandhi_full_fraction ≈ 0.0008` — almost every exported 'sentence' is just
> space-joined declined words (padapāṭha), not fused Sanskrit. So the realizer
> is a declension table dressed up as a sentence generator; the 'no-mock real
> Sanskrit' claim is overstated."

### Resolution

1. **Padapāṭha is canonical Sanskrit, not a degenerate form.** Each pada is a
   fully and correctly inflected word: the kāraka→vibhakti mapping (kartā→prathamā,
   karma→dvitīyā, karaṇa→tṛtīyā, …), liṅga-correct stems (incl. feminine nyāp
   stems that `Pratipadika.basic` gets *wrong*), and verb–subject number agreement
   (`bAlAH` bahu → `KAdeyuH` 3pl) are all present and verified against classical
   forms. The morphological case/agreement signal — exactly what H1′ (Paribhāṣā
   vs Dyck) probes — is intact and, if anything, *cleaner* without surface sandhi
   fusion obscuring word boundaries.
2. **The low full-sandhi fraction is honesty, not a defect (ADR-0033).** We apply
   only well-defined ach-sandhi (savarṇa-dīrgha / guṇa / vṛddhi / yaṇ) and leave
   every other junction as a word boundary, flagged `meta.sandhi="partial"`.
   Most padas end in visarga/anusvāra/consonant, where saṃhitā sandhi is either
   optional or ambiguous; fabricating those fusions would be a mock. Zero
   fabricated fusions is the correct behavior.
3. **Grammaticality basis is the derivation engine, not a heuristic.** Every
   surface is a successful `vidyut.prakriya` derivation carrying its sūtra-by-sūtra
   trace; an underivable frame yields `None`, never a guessed string.

If a future experiment requires saṃhitā-level sandhi, the principled extension is
to invert the `vidyut-data` sandhi-rules CSV (future work), still with no
fabrication. The H1′ re-test does not need it.

## Secondary objections (examined, non-fatal)

- **Small lexicon → low lexical TTR (~0.012), fails the informational diversity
  gate.** Intentional for a minimal-pair structural probe; structural diversity
  is maximal (10,000 distinct frames). The diversity gate is explicitly *not*
  wired into training (ADR-0024). Memorization risk is **flagged to the
  `measurement` workstream** and mitigated there by held-out frame/role-sequence
  splits (ADR-0035), not by inflating the lexicon here.
- **Ubhayapada roots (kṛ, sthā, vad) surface in ātmanepada via `results[0]`.**
  Both padas are grammatical; selection is deterministic and documented.
- **`xA1` is realized as juhotyādi `dadāti` though tagged gaṇa 1.** A documented
  lexical choice (the canonical "give" with sampradāna); grammatical either way.

## No-leak / fairness

- No Saṃsādhanī dependency, HTTP, or container is touched (constraint
  `no_samsadhani_dependency`).
- No CPU-fallback or mock surface path is reachable in the realized export; the
  label path (`stream_fixture_corpus`) is marked NOT-for-reported-results.
- Validated natively on the GB10 (`spark-5208`, aarch64), not under emulation —
  see `docs/infra/gb10-validation-2026-06.md`.
