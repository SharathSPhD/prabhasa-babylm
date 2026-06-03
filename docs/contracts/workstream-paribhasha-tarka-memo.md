# Tarka memo — paribhasha-core workstream (ADR-0034)

Pūrvapakṣa (objection) → Uttara (answer), addressing the adversarial review that
invalidated the prior L2 stack. Evidence is `docs/data/paribhasha-parity-ledger.json`
(10⁴ frames, seed 0) and `docs/data/shabdabodha-export-stats.json` (128 CI frames).

## P1. "The Paribhāṣā string is an isomorphic relabel of the kāraka parse (H≈0)."

Refuted, two ways:

- **Against the kāraka role grammar** (what arms B/D and the Dyck control encode):
  `H(structure | role-multiset) = 3.18 bits` on 10⁴ frames (1.55 bits on the 128 CI
  frames). The typed graph adds padārtha-conditioned topology the role labels do not fix.
- **Against the full lexicalized parse** `(stem, role)`: `H(structure | full parse) =
  0.44 bits > 0`. This residual is the **saṃkhyā (number)**, which is present in the
  surface and the Paribhāṣā string but *absent from `karaka_parse`*. A model given the
  gold parse still cannot predict the graph — so it is not a relabel even of the parse.

Unique structural templates = **499** (≫ 2); the prior stack had 2.

## P2. "The renderer drops the object role (VISAYATA) on transitive sentences."

Fixed (ADR-0034 D1). The renderer is now a canonical, sorted, **flat** typed-edge form
(`SANSA(category:src, category:dst[, qualifier])`); `parse(render(g)) == g` round-trips
on graphs combining AVACCHEDAKA + VISAYATA + SAMYOGATA. `transitive_visayata_fraction =
1.0` on both the 10⁴ and 128-frame sets (was 88/128 dropped). Regression tests:
`tests/unit/paribhasha/test_renderer_branches.py::test_avacchedaka_edge_is_lossless`,
`tests/unit/shabdabodha/test_fixtures_integration.py::test_transitive_visayata_present_in_every_success`.

## P3. "Training silently uses Sanskrit, not the Paribhāṣā string."

Fixed (ADR-0034 D2). `serialize_line` emits `meta['paribhasha_string']` for both
`PARIBHASHA` and `SHABDABODHA_ALIGNED`, and **raises** if it is missing rather than
falling back to `sentence.text`. The aligned source folds `paribhasha_string` into
`meta`. Tests: `test_assembly_aligned_lines_are_paribhasha_strings_not_sanskrit`,
`test_serialize_line_refuses_sanskrit_without_paribhasha_string`.

## P4. "Padārtha is fake — all nominals are DRAVYA."

Fixed (ADR-0034 D3.1). Padārtha is assigned from a stem lexicon; vidyā (knowledge) is a
guṇa. Categories realised on the corpus = {KRIYA, DRAVYA, GUNA} (≥ 3). A guṇa instrument
qualifies the kartā by prakāratā instead of being forced to DRAVYA.

## P5. "No comprehension gates — ill-formed frames are fabricated."

Implemented (ADR-0034 D3.4). ākāṅkṣā (kartā required; akarmaka dhātu rejects a karma) and
yogyatā (a guṇa cannot be kartā / karma-viṣaya / locus) gates **skip with a logged
`rule_id`**, never approximate. On 10⁴ frames the skip ledger is non-empty
(`VYU-Y01..Y04`, `VYU-G0x`) and coverage is 0.81 — a faithful drop, not 100% fabrication.

## Honest limitations (not blocking standalone closure; flagged downstream)

- **Lexicon scope.** Only `vidyā` is a guṇa in the current stem set, so guṇa-filler
  variety is narrow; expanding the open vocabulary tied to the padārtha inventory is M-level
  work (P-coverage-ledger is "published, no STOP"). The parity result does **not** rely on
  lexical richness — the saṃkhyā channel alone keeps `H(·|full parse) > 0`.
- **Oblique relations** use `SAMYOGATA` + a kāraka qualifier to obtain distinct typed
  shapes within the 6-sansa inventory; this is an abstraction over the full Gadādhara
  vibhāga/ādhāra distinctions, staged behind the M1→M3 rule inventory (ADR-0034 §Consequences).
- **āsatti / tātparya, nested viṣayatā, mukhya-viśeṣyatā** remain unimplemented (M2+).
- 10⁴-frame parity here is the standalone proxy; the **binding** information-parity gate
  (ADR-0035 D6) is re-run by the measurement/audit workstream on the frozen export before
  any GPU spend.
