# Foundation reframe index — 2026-06

Planning branch `planning/reframe-2026-06`. **Documentation and interface freeze only.**
No worktrees, training, or `matrix.py` edits on this branch.

## Artifacts

| Document | Path |
|---|---|
| PRD (reframe) | [docs/prd.md](../prd.md) |
| Spec (reframe) | [docs/spec.md](../spec.md) |
| ADR-0017 H1 null + relocation | [docs/decisions/0017-h1-proxy-null-scope-and-claim-relocation.md](../decisions/0017-h1-proxy-null-scope-and-claim-relocation.md) |
| ADR-0018 Paribhāṣā generator | [docs/decisions/0018-paribhasha-layer-2-typed-generator.md](../decisions/0018-paribhasha-layer-2-typed-generator.md) |
| ADR-0019 Śabdabodha pipeline | [docs/decisions/0019-shabdabodha-pipeline-full-vyutpattivada.md](../decisions/0019-shabdabodha-pipeline-full-vyutpattivada.md) |
| ADR-0020 BabyLM dual track | [docs/decisions/0020-babylm-dual-track.md](../decisions/0020-babylm-dual-track.md) |
| ADR-0021 Arm namespaces | [docs/decisions/0021-arm-namespace-competition-vs-h1-matrix.md](../decisions/0021-arm-namespace-competition-vs-h1-matrix.md) |
| ADR-0022 GB10 gate | [docs/decisions/0022-gb10-full-stack-validation-gate.md](../decisions/0022-gb10-full-stack-validation-gate.md) |
| ADR-0023 GB10 SDPA fallback | [docs/decisions/0023-flash-attn-gb10-sdpa-fallback.md](../decisions/0023-flash-attn-gb10-sdpa-fallback.md) |
| Wave-1 integration report | [integration-data-engine-v2-2026-06.md](./integration-data-engine-v2-2026-06.md) |
| Wave-2 integration report | [integration-data-engine-v2-wave2-2026-06.md](./integration-data-engine-v2-wave2-2026-06.md) |
| Wave-3 integration report | [integration-wave3-experiment-enablement-2026-06.md](./integration-wave3-experiment-enablement-2026-06.md) |
| Interface freeze | [docs/contracts/interface-freeze-2026-06.md](./interface-freeze-2026-06.md) |
| Aligned-pair JSON Schema | [docs/contracts/aligned-pair-schema.json](./aligned-pair-schema.json) |

**Evidence (closed H1):** [phase-2-h1-tarka-memo.md](./phase-2-h1-tarka-memo.md),
[../data/phase2-h1-cogs-se-100m.json](../data/phase2-h1-cogs-se-100m.json),
git tag `h1-proxy-null-2026-06` on `main`.

**Strategy sources (slm-1):** `psalm-consolidation-report.md` (§4–5, §9; §3 audit stale),
`non-scaling-research.md`, `babylm-res-*.md`, `slm-sanskrit-research-*.md`.

## Seven execution units

| Unit | Wave | Depends | Owns (typical) |
|---|---|---|---|
| U1 GB10 stack validation | 1 | — | `infra/dgx_spark/Dockerfile.verified` |
| U2 Sanskrit fixtures ≥10⁴ | 1 | — | `data/fixtures/`, jsonl export |
| U3 Dyck match + Hu ADR | 1 | U2 stats | `matching.py` targets, ADR |
| U4 Paribhāṣā generator | 1 | freeze | `generators/paribhasha/` |
| U6 BabyLM + manifest + tokenizer | 1 | freeze | `benchmarks/`, manifest |
| U5 Śabdabodha pipeline | 2 | U2, U4 | Vyutpattivāda engine |
| U7 ELC-PSALM backbone | 2 | U1, U6 | model code |

**Integration:** branch `integration/data-engine-v2` — sole owner of `matrix.py`,
`competition_matrix.py`, enum/assembly wiring.

## Wave-1 launch preconditions

All must be true before forking unit worktrees:

1. **Human sign-off** on this index + interface freeze + ADRs 0017–0022.
2. **Interface freeze accepted** — checklist in [interface-freeze-2026-06.md](./interface-freeze-2026-06.md) §7.
3. **Per-unit charters** written at worktree creation (not on planning branch).
4. **H1′ pre-registration** — thresholds in YAML + ADR (human decision; see below).
5. **U1 gate** — `GB10_STACK_VERIFIED` before any GPU training charter (ADR-0022).
   **Conditional verified (2026-06):** host torch/SDPA + `Dockerfile.verified` path per
   ADR-0023; flash-attn optional, not blocking Wave-1 data engine.
6. **Interface freeze honored on integration** — `AnnotatedSentence`, `DEFAULT_KEYS`,
   H1 matrix A–H unchanged; `PARIBHASHA` wired via extension process only
   ([integration-data-engine-v2-2026-06.md](./integration-data-engine-v2-2026-06.md)).

## Wave-2 launch preconditions

1. Wave 1 freeze still valid (no breaking change to §1–4 of interface freeze).
2. U2 fixture JSONL published (≥10⁴ lines, versioned path in charter).
3. U4 `types.py` + `renderer.py` API stable per U5 charter.
4. U6 tokenizer + manifest v0 frozen for U7.

## Wave-2 integration checklist (2026-06)

- [x] `workstream/shabdabodha` merged (Vyutpattivāda + aligned export + CI fixture)
- [x] `workstream/backbone` merged (ELC-PSALM + ADR-0029)
- [x] `paribhasha` public aligned API exports
- [x] `SHABDABODHA_ALIGNED` enum + assembly + `ShabdabodhaAlignedSource`
- [x] ELC-PSALM `ml` package exports + BabyLM PLL eval factory + `psalm eval --elc`
- [x] `resolve_architecture(elc_psalm_s|m)` (training-loop hook deferred)
- [x] Full gate green (361 tests, ≥80% coverage) — see Wave-2 integration report
- [x] Wave-2 FF merge to `main` (no remote push)

## Wave-3 integration checklist (2026-06)

- [x] `workstream/eval-train` merged (PLL minimal pairs, ELC trainer, HF export, ADR-0032)
- [x] `workstream/h1prime` merged (ADR-0030, `run_h1prime_pilot.py`, `configs/research/h1p/`)
- [x] `workstream/crystallization` merged (M1 pipeline, ADR-0031)
- [x] `default_h1p_matrix()` registered (`h1p:A`–`C`, optional `h1p:L`)
- [x] H1′ harness primary venue → `babylm_eval` minimal pairs (COGS fallback)
- [x] CLI: `psalm train elc-smoke` + `psalm eval babylm smoke`
- [x] Full gate green (382 tests, 89.46% coverage) — see Wave-3 report
- [ ] Human sign-off on H1′ proxy battery launch (ADR-0030; status UNBLOCKED/RUN-PENDING)

## Already implemented (do not re-open as gaps)

Experiment ledger, `comparison_tests.py`, `knowledge_store.py`, generators
(dyck, scramble, samsadhani, vidyut, jsonl, paribhasha, shabdabodha aligned),
`karaka_frames.py`, H1 matrix A–H, closure discipline, ADR-0002 three readings,
Vyutpattivāda engine (U5), ELC-PSALM encoder skeleton (U7).

## Genuinely absent (charter scope)

Official full BLiMP/EWoK zero-shot via installed pipeline (smoke PLL only),
trained ELC-PSALM HF export + competition YAML training hook,
`Dockerfile.verified` flash-attn proof,
U2 fixture-vs-live Dyck target parity, Hu k=64 alphabet extension,
Saṃsādhanī-live crystallization sentence yield.

## Human decisions before charters

| # | Decision | Blocks |
|---|---|---|
| 1 | H1′ primary metrics + thresholds (EWoK vs BLiMP-arg vs graph tasks) | `h1p` battery configs |
| 2 | Strict-Small vs Strict first submission target | U6 manifest mix |
| 3 | flash-attn waiver policy if sm_121 build fails | U1 closure |
| 4 | Paribhāṣā ASCII vs IAST for tokenizer v0 | U6/U4 renderer default |
| 5 | Crystallization Milestone 1 domain sign-off (charter exists) | C-S2 worktree |
