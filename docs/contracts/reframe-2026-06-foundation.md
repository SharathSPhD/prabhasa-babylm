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

## Wave-2 launch preconditions

1. Wave 1 freeze still valid (no breaking change to §1–4 of interface freeze).
2. U2 fixture JSONL published (≥10⁴ lines, versioned path in charter).
3. U4 `types.py` + `renderer.py` API stable per U5 charter.
4. U6 tokenizer + manifest v0 frozen for U7.

## Already implemented (do not re-open as gaps)

Experiment ledger, `comparison_tests.py`, `knowledge_store.py`, generators
(dyck, scramble, samsadhani, vidyut, jsonl), `karaka_frames.py`, H1 matrix A–H,
closure discipline, ADR-0002 three readings.

## Genuinely absent (charter scope)

Paribhāṣā generator, Śabdabodha pipeline, official BabyLM eval integration,
`Dockerfile.verified`, full GB10 validation, joint tokenizer + BabyLM manifest.

## Human decisions before charters

| # | Decision | Blocks |
|---|---|---|
| 1 | H1′ primary metrics + thresholds (EWoK vs BLiMP-arg vs graph tasks) | `h1p` battery configs |
| 2 | Strict-Small vs Strict first submission target | U6 manifest mix |
| 3 | flash-attn waiver policy if sm_121 build fails | U1 closure |
| 4 | Paribhāṣā ASCII vs IAST for tokenizer v0 | U6/U4 renderer default |
| 5 | Crystallization Milestone 1 domain sign-off (charter exists) | C-S2 worktree |
