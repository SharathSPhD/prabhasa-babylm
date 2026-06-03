# Charter — workstream: paribhasha-core (L2 faithful Vyutpattivāda)

- Branch: `workstream/paribhasha-core` (forked from `origin/main` @ core-rebuild reframe)
- Acceptance contract: [`docs/contracts/workstream-paribhasha-acceptance.yaml`](docs/contracts/workstream-paribhasha-acceptance.yaml)
- Governing ADRs: **ADR-0034** (faithful Vyutpattivāda + lossless render + dual-task), ADR-0026 (renderer channels), ADR-0035 (GPU-only/no-mock)
- Plan: `~/.cursor/plans/psalm_core_rebuild_gpu_5d971f30.plan.md` (Phase 2, paribhasha node)

## Mission

Rebuild L2 so the Paribhāṣā prior carries information **beyond** the kāraka parse — a
**faithful** Vyutpattivāda Śabdabodha pipeline with a **lossless** render.

## Scope IN
- **Lossless renderer**: every edge incl. `VISAYATA` and inner `AVACCHEDAKA`-scoped edges survives;
  round-trip property tests on AVACCHEDAKA+VISAYATA+SAMYOGATA graphs (fail CI on any dropped edge)
- `serialize_line` emits `paribhasha_string` (not `sentence.text`) for PARIBHASHA/SHABDABODHA_ALIGNED; delete the H1′ workaround source
- **real padārtha** assignment from morphology (not all-DRAVYA); **viśeṣaṇa→viśeṣyatā/prakāratā** from actual modifiers; kāraka-specific graph topology
- **ākāṅkṣā / yogyatā / āsatti** gates (skip + log `rule_id`, no fabrication)
- open vocabulary tied to padārtha inventory; coverage ledger per `rule_id`
- dual-task aligned export `paribhasha_aligned_v1`

## Scope OUT
Saṃsādhanī, Dyck, GB10 torch build, `matrix.py`, H1′ battery execution.

## Hard constraints
- **GPU-only** for reported runs; **no mocks** (no toy 17-label lexicon, no synthetic AVACCHEDAKA boilerplate).
- **Information parity is a STOP gate**: if H(structure | kāraka) ≈ 0, the stream is a relabel — stop, do not claim L2.

## Dependency
Consumes vidyut `AnnotatedSentence` with gold `karaka_parse` (fixture `paninian_v1.jsonl`).

## GB10 GPU mutex
Acquire `/home/sharaths/projects/PSALM/.gb10-gpu.lock` before any GPU op; release after; one at a time.

## Closure (GATE 2 — human sign-off required)
`make gate` green; round-trip tests; acceptance checks PASS (esp. `VISAYATA`==100% transitive,
H(structure|kāraka)≫0, template count≫2); closure JSON + Tarka; `psalm contract check`.
