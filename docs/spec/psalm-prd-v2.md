# PSALM product requirements document v2 (core rebuild — four workstreams)

> **Core-rebuild reframe (2026-06-03).** H1/H1′ nulls are artifacts (adversarial
> reviews). Mission shifts from "harden the skeleton" to **build the core**: faithful
> Vyutpattivāda (ADR-0034), Vidyut-native gold-kāraka realization (ADR-0033), and a
> GPU-only/no-mock measurement harness with an information-parity gate (ADR-0035).
> A fourth **measurement** workstream is added. Prior H1′ results are withdrawn
> (`docs/experiments/h1prime-invalidation-2026-06.md`).

Companion spec: [`psalm-spec-v2.md`](./psalm-spec-v2.md). Launch plan:
[`psalm-worktree-launch-plan.md`](./psalm-worktree-launch-plan.md).

---

## 1. Product summary

PSALM delivers a **reproducible small LM research platform** and optional **BabyLM
competition submission** where Sanskrit grammatical structure is a **generative prior**,
not a scraped corpus. After the H1 null was shown **venue-saturated** and the H1′ pilot
**invalidated**, the product thesis is: *prove the prior is real before spending GPU.*

- **Faithful L2 first:** real Vyutpattivāda (padārtha, viśeṣyatā, ākāṅkṣā/yogyatā) with a
  **lossless** render (the object role `VISAYATA` must survive) — not a kāraka relabel.
- **Vidyut-native realization:** gold kāraka **by construction** (~100% acceptance), no
  Saṃsādhanī Docker on the critical path (ADR-0033).
- **H1′ (fair):** Paribhāṣā vs matched Dyck on **official** EWoK/BLiMP, **GPU-only/no-mock**,
  off-saturation, ≥10 seeds — run only after the information-parity audit passes (ADR-0035).
- **Multi-target / Śabdabodha automation:** retained, downstream of a real L2 prior.

---

## 2. Users and outcomes

| User | Outcome | Acceptance signal |
|---|---|---|
| Research lead | Honest go/no-go with pre-registered thresholds | ADR + ledger + Tarka per phase |
| BabyLM reviewer | Strict-budget manifest + official eval | `comp:*` arms, OSF pipeline pass |
| Sanskrit / Nyāya collaborator | Auditable Vyutpattivāda coverage ledger | Published rule-id failure rates |
| Orchestrator | Four parallel worktrees merge safely after the info-parity gate | Interface freeze + integration PR |

---

## 3. Workstream: vidyut

### 3.1 Purpose

Implement **`VidyutFrameRealizer`** (ADR-0033): frame → per-pada subanta/tiṅanta via
`vidyut.prakriya` → order → forward sandhi → **gold `karaka_parse` by construction**.
Validate the **GB10 aarch64/sm_121** native stack. Saṃsādhanī is demoted to an optional
offline cross-check. Fix `ROLE_DHATUS` transitivity (akarmaka ≠ karma).

### 3.2 Requirements

| ID | Requirement | Priority |
|---|---|---|
| V-01 | `VidyutFrameRealizer` streams ≥10⁴ reproducible `AnnotatedSentence` with non-empty gold `karaka_parse` + `derivation`, **GPU-only on GB10** | P0 |
| V-02 | Realization acceptance **~100%** on n≥2000 real frames via Paninian derivability + agreement validator (**no Saṃsādhanī dependency**) | P0 |
| V-03 | `infra/dgx_spark/Dockerfile.verified` + `docs/infra/gb10-validation-2026-06.md` per ADR-0022 | P0 |
| V-04 | Export frozen fixture JSONL ≥10⁴ lines (`data/fixtures/paninian_v1.jsonl`) with gold kāraka + manifest hashes | P0 |
| V-05 | Role→vibhakti table + WX→(root,gaṇa,lakāra) map + forward-sandhi table (inverted rules CSV; partial on miss, **never fabricated**) | P0 |
| V-06 | `ROLE_DHATUS` transitivity annotation; akarmaka roots never paired with `karma` | P0 |
| V-07 | Optional: offline Saṃsādhanī probe as cross-check only (not on critical path) | P2 |

### 3.3 Interfaces

**Produces:** `AnnotatedSentence` per `application/data/ports.py` (frozen).

**Consumes:** `KarakaFrame` from crystallization; dhātu grid config; Saṃsādhanī base URL / Docker.

**Does not produce:** Paribhāṣā strings (downstream paribhasha workstream).

### 3.4 Falsifiable acceptance (standalone gate)

- `make gate` green in worktree (≥80% cov on touched packages).
- GB10 checklist: torch CUDA matmul, vidyut import, ≥100 `vidyut_source` sentences, gate in container.
- Realization: grammaticality_probe ≥0.80 on n=2000 **or** signed waiver with documented blocker (ADR-0025).
- Fixture export checksum recorded in experiment ledger.

### 3.5 Non-goals

- Paribhāṣā type system implementation
- BabyLM training / ELC backbone
- `matrix.py` / competition arm wiring
- Claiming crystallization dose victory before realization passes

---

## 4. Workstream: paribhasha

### 4.1 Purpose

First-class **Layer L2**: a **faithful** Vyutpattivāda Śabdabodha pipeline (ADR-0034) —
real semantics + a **lossless** render — that carries information **beyond** the kāraka
parse, not an isomorphic relabel.

### 4.2 Requirements

| ID | Requirement | Priority |
|---|---|---|
| P-01 | **Lossless linearization**: every edge incl. `VISAYATA` and inner `AVACCHEDAKA`-scoped edges appears in `paribhasha_string`; round-trip property tests fail CI on any dropped edge | P0 |
| P-02 | `serialize_line` emits `paribhasha_string` (not `sentence.text`) for `PrePretrainSource.PARIBHASHA`/`SHABDABODHA_ALIGNED`; delete the H1′ workaround source | P0 |
| P-03 | **Real padārtha assignment** from morphology (dravya/guṇa/kriyā/…), not all-DRAVYA | P0 |
| P-04 | **Viśeṣaṇa→viśeṣyatā/prakāratā** from actual modifiers; kāraka-specific graph topology | P0 |
| P-05 | **ākāṅkṣā / yogyatā / āsatti** comprehension gates before emission (skip + log `rule_id`, no fabrication) | P0 |
| P-06 | Information parity: H(structure\|kāraka) ≫ 0 and template count ≫ 2 on 10⁴ frames (else STOP) | P0 |
| P-07 | Open vocabulary tied to padārtha inventory (not the ~17-label toy lexicon); coverage ledger per `rule_id` | P0 |
| P-08 | Dual-task aligned export `paribhasha_aligned_v1` matching `aligned-pair-schema.json` | P1 |

### 4.3 Interfaces

**Consumes:** `AnnotatedSentence` with non-empty `karaka_parse` from vidyut fixtures/stream.

**Produces:**

- Training surface: Paribhāṣā ASCII lines (`AnnotatedSentence`, `meta.source=paribhasha`).
- Rich supervision: JSONL `paribhasha_aligned_v1` (four fields + `meta.schema_version`).

**Contract fields (frozen):** `text`, `karaka_parse`, `shabdabodha_graph`, `paribhasha_string`, `meta`.

### 4.4 Falsifiable acceptance

- **`VISAYATA` present in 100%** of transitive `paribhasha_string`; round-trip tests pass
  on graphs combining `AVACCHEDAKA` + `VISAYATA` + `SAMYOGATA`.
- **H(structure | kāraka) ≫ 0** and template count ≫ 2 on 10⁴ frames (information-parity).
- ≥50 type-constraint tests; 10⁴ seeded graphs, 0 validation failures.
- Pipeline: on ≥10⁴ fixture sentences, report `% covered` by rule_id; M2 target or STOP with Tarka.
- `serialize_line` proven to emit Paribhāṣā (not Sanskrit); module cov ≥80%.
- No edits to `DEFAULT_KEYS` or H1 `matrix.py`.

### 4.5 Non-goals

- Full historical Paribhāṣā operator inventory v1
- H1′ BLiMP battery execution (orchestrator / integration)
- Saṃsādhanī Docker provisioning (vidyut)
- Z3 / Nyāya kernel (H3)

---

## 5. Workstream: dyck

### 5.1 Purpose

Rigorous **matched** k-Shuffle Dyck control (L0): replicate Hu et al. hyperparameters,
match **sentence-level** Pāṇinian statistics, standalone test package.

### 5.2 Requirements

| ID | Requirement | Priority |
|---|---|---|
| D-01 | `hu_replication_config()` documented in ADR with citation to Hu et al. arXiv:2502.19249 | P0 |
| D-02 | `match_dyck` recomputes best config vs **sentence-level** Saṃsādhanī stats (not form-only Vidyut) | P0 |
| D-03 | Matched config distance logged; Δ on `DEFAULT_KEYS` ≤ ε (ε preregistered in workstream ADR stub) | P0 |
| D-04 | `DyckSentenceSource` isolated tests + property tests (well-formed brackets) | P0 |
| D-05 | Manifest snippet: frozen `dyck_config` blob per arm C / `h1p:C` / `comp:*:C` | P1 |
| D-06 | Arm-H scramble fairness check documented (reuse `scramble_source`) | P2 |

### 5.3 Interfaces

**Consumes:** `targets: dict[str, float]` from Pāṇinian stream measurement job (vidyut fixture stats).

**Produces:** `DyckConfig` + stream of `AnnotatedSentence` (`language="dyck"`, empty `karaka_parse`).

### 5.4 Falsifiable acceptance

- Property tests: 10⁵ generated sequences all bracket-valid.
- Match test: best candidate within ε of targets on `DEFAULT_KEYS` (record ε in closure report).
- Hu replication smoke: config fields match ADR-0025 Hu pin (k=64 cap per ASCII alphabet ADR).
- Standalone: worktree tests run without torch/Saṃsādhanī.

### 5.5 Non-goals

- Paribhāṣā or Vidyut changes
- Training loops
- Changing frozen H1 phase-2 stored Dyck configs retroactively

---

## 5A. Workstream: measurement

### 5A.1 Purpose

Make the experiment a **fair test** (ADR-0035): a **GPU-only, no-mock** battery on
**official** EWoK + BLiMP-arg with corrected paired statistics and the information-parity
preflight that **blocks GPU training** until the prior is shown to carry real signal.

### 5A.2 Requirements

| ID | Requirement | Priority |
|---|---|---|
| M-01 | Remove silent CUDA→CPU fallback; battery **raises** without CUDA | P0 |
| M-02 | Delete `RunMode.MOCK` / `MockUniformBaseline` from battery paths; smoke venues stamped `evidence=false`, never emit a verdict | P0 |
| M-03 | Official EWoK + full BLiMP-arg via `invoke_official_zero_shot`; **ELC-PSALM PLL** (not causal logprob) | P0 |
| M-04 | `comparison_tests` fixed: **paired bootstrap on per-seed diffs**, one method per contrast | P0 |
| M-05 | Held-out train/eval split with automated leakage (disjointness) check, fails closed | P0 |
| M-06 | Real SentencePiece tokenizer + real budgets (≥130M downstream); **≥10 seeds** for any gating contrast | P0 |
| M-07 | Information-parity audit harness: H(structure\|kāraka), `VISAYATA` coverage, template count, venue off-saturation, mock/CPU path scan | P0 |

### 5A.3 Falsifiable acceptance

- Battery aborts (non-zero) on CPU or with any mock venue in a reported contrast.
- Paired-diff bootstrap matches a hand-checked reference; independent-resample path removed.
- Official-shard smoke run produces real PLL numbers (stamped evidence-grade).
- Information-parity audit returns PASS/FAIL with the six ADR-0035 D6 checks.

### 5A.4 Non-goals

- Generator/renderer changes (vidyut/paribhasha/dyck)
- The H1′ scientific interpretation (orchestrator, Phase 4)

---

## 6. Cross-workstream integration (not owned by worktrees)

| Integration task | Owner branch | Trigger |
|---|---|---|
| `PrePretrainSource` enum + `assembly.py` | `integration/data-engine-v2` | All four standalone gates + info-parity gate pass |
| `competition_matrix.py` arms A–F | same | + manifest v0 |
| H1′ configs `configs/research/h1p/` | same | Human prereg sign-off |
| BabyLM eval + tokenizer joint | integration (U6) | GB10 gate + paribhasha renderer default |

**Merge protocol:** see launch plan §4.

---

## 7. Statistical and closure PRD

- **≥10 seeds** for any gating contrast (n=3 only for wiring smokes); **paired bootstrap on
  per-seed differences** (not independent resampling); Holm–Bonferroni across arm families.
- **GPU-only, no-mock** for any reported result (ADR-0035); **information-parity gate** blocks
  GPU training until the prior is shown real.
- Six-layer closure per workstream YAML + program `closure-contract.md`.
- A NULL declared **before** the information-parity gate passes is not evidence.
- Human sign-off on **interpretation** before merge to `main`.

---

## 8. Dependencies and risks

| Risk | Mitigation owner |
|---|---|
| GB10 build failure | vidyut (ADR-0022); blocks all GPU |
| Vidyut sandhi gaps (no forward joiner) | vidyut; inverted-rules table, `meta.sandhi=partial` on miss, never fabricate |
| Paribhāṣā stays a relabel (H(structure\|kāraka)≈0) | paribhasha; information-parity gate STOPs GPU spend (ADR-0035) |
| Dyck mismatch invalidates B/C contrast | dyck; re-fit to **real** vidyut sentence-level stats |
| Hidden mock/CPU path re-enters battery | measurement; automated path scan + fail-hard device resolver |
| Single GB10 | Serialize GPU charters via GB10 mutex (**GPU-only**, no CPU substitute) |

---

## 9. Deliverables checklist (program-level)

| Deliverable | Workstream primary |
|---|---|
| `Dockerfile.verified` | vidyut |
| `data/fixtures/paninian_v1.jsonl` | vidyut |
| `data/fixtures/paribhasha_aligned_v1.jsonl` | paribhasha |
| `configs/data/dyck_matched_<hash>.yaml` | dyck |
| GPU-only no-mock battery + paired-diff stats + info-parity audit report | measurement |
| Workstream closure JSON + Tarka | each |
| Integration manifest + `comp:*` configs | integration |
