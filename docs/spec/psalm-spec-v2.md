# PSALM system specification v2 (reframe 2026-06)

> **Core-rebuild reframe (2026-06-03).** Three independent adversarial reviews
> established that the H1/H1′ nulls are **implementation/measurement artifacts**, not
> science. This spec is now governed by the core-rebuild plan and
> **ADR-0033** (Vidyut-native realization), **ADR-0034** (faithful Vyutpattivāda +
> lossless render + dual-task), **ADR-0035** (GPU-only, no-mock, information-parity
> gate). Hard constraints: everything on GPU (no CPU fallback), no mocks, no proxy
> venues; **no GPU training until the information-parity audit passes**. Prior H1′
> results are withdrawn — see `docs/experiments/h1prime-invalidation-2026-06.md`.

Authoritative program spec after consolidation (`slm-1/docs/psalm-consolidation-report.md`),
Phase-2 H1 closure (ADR-0017, re-scoped), H1′ pilot (ADR-0030, **invalidated**), and
crystallization M1 (bounded domain). PRD: [`psalm-prd-v2.md`](./psalm-prd-v2.md). Launch
plan: [`psalm-worktree-launch-plan.md`](./psalm-worktree-launch-plan.md).

**Planning branch note:** `planning/reframe-2026-06` carries this document set; executable
code on `main` at `817384b` is ahead of this worktree’s `src/` snapshot. Audits below cite
`main` / integration tip unless noted.

---

## 1. Program identity

**PSALM** (Pāṇinian Structured pretraining for Small LAnguage Models) trains one small
multilingual LM read three ways (ADR-0002):

1. **English structural generalization** — official BabyLM pipeline (BLiMP, GLUE, EWoK, compositional).
2. **Sanskrit linguistic competence** — morphology, sandhi, kāraka (synthetic + DCS/GRETIL).
3. **Cross-lingual transfer** — English eval gap between grammar-prior arms and NL-only baseline
   while holding Sanskrit evals.

Pretraining text in all layers counts toward BabyLM caps (`babylm-res-2`).

---

## 2. Honest empirical state (do not over-claim)

| Result | Venue / scale | Verdict | Implication for v2 |
|---|---|---|---|
| H1 B vs C | COGS @ 100M proxy, arms B/C | **RE-SCOPED** (Δ AUC +0.003, CI includes 0; paired stats correct) | **Venue-saturated** (arm A ≥0.90 with zero dose) and **L1-only** — does not bear on L2 Paribhāṣā; instrument cannot adjudicate the prior on this capability |
| H1′ battery | BLiMP minimal-pair **mock** proxy @ 60M, `h1p:*` | **INVALIDATED / NOT EVIDENCE** | Train↔eval leakage on 4 hand-written pairs; renderer dropped `VISAYATA`; trained on Sanskrit not Paribhāṣā; proxy scale — withdrawn (`h1prime-invalidation-2026-06.md`) |
| Crystallization M1 | Bounded natural-kinds domain | **MIS-DIAGNOSED** — "0 unique sentences" was `--skip-generation`, not rejection; live Saṃsādhanī accepts ~92% | Realization is an **engine/wiring artifact**, not a linguistic ceiling; Vidyut-native realizer is the fix (ADR-0033) |
| Paribhāṣā module | `main` `generators/paribhasha/` | **SCAFFOLDING** — bijective kāraka relabel (2 templates/128, H(structure\|kāraka)≈0); renderer drops object role | Must be rebuilt faithful + lossless before any L2 claim (ADR-0034) |
| GB10 stack | ADR-0022 | **NOT VERIFIED** — no `Dockerfile.verified` | Blocks GPU training charters until validated |

**Live bet (core-rebuild):** prove the L2 prior is *real before spending GPU* — build a
faithful Vyutpattivāda engine (lossless render, real padārtha/viśeṣyatā/ākāṅkṣā, dual-task
aligned), a Vidyut-native gold-kāraka realizer (~100% acceptance, no Saṃsādhanī Docker), a
matched Dyck control re-fit to real sentence-level stats, and a GPU-only/no-mock harness on
official EWoK/BLiMP. Then re-run **H1′ (Paribhāṣā vs Dyck)** as a fair, off-saturation test.
The information-parity gate (ADR-0035) forbids GPU training on a stream that is a relabel.

---

## 3. Layered curriculum (L0–L4)

```
L0  k-Shuffle Dyck          — structural warmup, no semantics (Hu et al. ACL 2025 line)
L1  Pāṇinian synthetic      — Vidyut tiṅanta + Saṃsādhanī sentence-level kāraka (gold parse)
L2  Paribhāṣā + Śabdabodha  — typed graphs, Vyutpattivāda → paribhasha_aligned_v1
L3  Real Sanskrit           — DCS/GRETIL (Vidyut analysis), research-unlimited
L4  BabyLM English          — competition eval target, annealed dominant share
```

Layers compose in mixture sampling; they are not mutually exclusive ablations.

---

## 4. Arm matrices (namespaced)

### 4.1 H1 matrix — **frozen** (`A`–`H`, `configs/phase2/`)

Closed at proxy scale. Ledger uses bare letters. **Do not rename** (tag `h1-proxy-null-2026-06`).

Decisive historical pair: **B (paninian) vs C (dyck)** — null on COGS sample-efficiency/discrimination.

### 4.2 Competition matrix — `comp:{track}:{arm}`

| Arm | Pre-pretrain | Pretrain | Eval |
|---|---|---|---|
| A | none | English | EN BabyLM |
| B | paninian | English | EN |
| C | dyck | English | EN |
| D | paribhasha | English | EN (H1′ treatment namespace) |
| E | paninian | Sanskrit + English | EN + SA |
| F | paninian + paribhasha | Sanskrit + English | EN + SA full stack |

Tracks: `strict_small` (10M words), `strict` (100M). Wiring only on `integration/data-engine-v2`.

### 4.3 H1′ matrix — `h1p:{arm}`

Pre-registered contrast: **Paribhāṣā vs matched Dyck** on EWoK + BLiMP argument-structure +
Śabdabodha graph consistency (thresholds: human sign-off before battery).

Pilot artifact: `docs/data/h1prime-pilot-60m.json` (worktree `PSALM-h1run`).

---

## 5. Three execution workstreams (composition)

| Workstream | Owns | Upstream inputs | Downstream outputs |
|---|---|---|---|
| **vidyut** | L1 **Vidyut-native** realization (`VidyutFrameRealizer`), GB10 native build, gold kāraka by construction, `ROLE_DHATUS` transitivity (ADR-0033) | Dhātupāṭha grid, crystallization `KarakaFrame`, vidyut.prakriya subanta/tiṅanta | `AnnotatedSentence` (`text`, `karaka_parse`, `derivation`, `meta`) + `paninian_v1.jsonl` ≥10⁴ |
| **paribhasha** | L2 **faithful** Vyutpattivāda: lossless renderer, real padārtha/viśeṣyatā/ākāṅkṣā, dual-task aligned (ADR-0034) | `AnnotatedSentence` with gold kāraka | `paribhasha_aligned_v1` records (lossless; `VISAYATA` survives) |
| **dyck** | L0 control, Hu replication, `match_dyck` re-fit to **real** sentence-level stats | Sentence-level Pāṇinian **stats vector** (from vidyut realizer) | `DyckConfig` + `AnnotatedSentence` (`language=dyck`) |
| **measurement** | GPU-only/no-mock harness: paired bootstrap, official EWoK/BLiMP PLL, real tokenizer/budgets, information-parity audit (ADR-0035) | trained checkpoints, official shards | valid contrasts + closure JSON; the H1′ verdict |

**Integration branch** merges enum/assembly/matrix only after each workstream passes its
standalone contract (`docs/contracts/workstream-*-acceptance.yaml`).

### 5.1 Cross-workstream data contracts (frozen)

See `docs/contracts/interface-freeze-2026-06.md` and `aligned-pair-schema.json`.

**vidyut → paribhasha:** `AnnotatedSentence` with non-empty `karaka_parse`; roles in
`SUPPORTED_KARAKAS` for pipeline v1; `meta.source` ∈ `{vidyut, samsadhani, crystallization}`.

**paribhasha → training:** `paribhasha_aligned_v1` JSONL lines OR `PrePretrainSource.PARIBHASHA`
surface strings for L2-only pretrain; graph in `shabdabodha_graph` for dual-task heads.

**dyck → training:** matched `DyckConfig` serialized in manifest + `PrePretrainSource.DYCK`;
statistics distance logged vs Pāṇinian targets on `DEFAULT_KEYS`.

---

## 6. Paribhāṣā module (consolidation §5, code on `main`)

Path: `src/psalm/infrastructure/generators/paribhasha/`

| File | Role |
|---|---|
| `types.py` | `PadarthaCategory`, `SansaType`, graph validation |
| `ontology.py` | sapta-padārtha |
| `relations.py` | VISAYATA, PRAKARATA, … |
| `inference.py` | PAKSA, HETU, VYAPTI, … |
| `generator.py` | Strata 1–3 seeded graphs |
| `renderer.py` | Graph → ASCII Paribhāṣā line — **rebuild lossless (ADR-0034 D1): every edge incl. `VISAYATA` survives** |
| `shabdabodha.py` | Vyutpattivāda rules — **rebuild faithful (ADR-0034 D3): real padārtha/viśeṣyatā/ākāṅkṣā, not VYU-001…008 relabel** |

**Śabdabodha pipeline (free gold annotation):**

```
Sanskrit surface (L1)
  → kāraka_parse (gold)
  → Vyutpattivāda rules (coverage ledger per rule_id)
  → ShabdabodhaGraph
  → paribhasha_string
  → paribhasha_aligned_v1 export
```

Skips are **logged**, not hallucinated (`ShabdabodhaSkip`). ADR-0019 targets full Gadādhara
scope incrementally with M2 ≥90% on fixture strata.

---

## 7. Dyck control (existing — harden, do not greenfield)

- Domain: `src/psalm/domain/data/dyck.py` (`hu_replication_config()`, k-shuffle).
- Adapter: `src/psalm/infrastructure/generators/dyck_source.py`.
- Matching: `src/psalm/domain/data/matching.py` (`DEFAULT_KEYS` frozen).
- Tests on `main`: `test_dyck.py`, `test_dyck_properties.py`, assembly + generator contract.

Workstream scope: Hu et al. ADR, sentence-level target stats from **live** Pāṇinian stream,
standalone package boundary, manifest reproducibility — not reimplementing Dyck math.

---

## 8. Vidyut / Saṃsādhanī (L1)

| Component | Path | Status on `main` |
|---|---|---|
| Vidyut tiṅanta adapter | `vidyut_source.py` | Implemented; lazy import; derivation trace |
| Saṃsādhanī sentence adapter | `samsadhani.py` | Implemented; Docker/toolkit; fail-closed |
| Kāraka frames | `domain/data/karaka_frames.py` | Grid enumerator |
| Crystallization M1 | `infrastructure/crystallization/` | 15 tests / ~86% cov module; live gen blocker |
| aarch64 smoke | `test_vidyut_aarch64.py` | Version/doc gate, not full GB10 |

**Crystallization measured (2026-06, `main` CPU):**

- Candidate frame signatures: **8,386,128**
- Generator-ready (both entity stems validated): **32,976**
- Live Saṃsādhanī acceptance probe (n=2000 generator-ready frames): **~0%** per program
  charter run — realization is binding constraint

---

## 9. Evaluation and statistics

- **GPU-only, no-mock (ADR-0035):** battery fails hard without CUDA; no `RunMode.MOCK`,
  no proxy venues in any reported result. Smoke venues stamped `evidence=false`.
- Official BabyLM: `benchmarks/babylm_eval.py` via `invoke_official_zero_shot` on full
  EWoK + BLiMP-arg shards, **ELC-PSALM PLL** (not causal logprob). Held-out train/eval split.
- Historical H1: COGS/SCAN — **not** go/no-go; re-scoped venue-saturated.
- H1′: official EWoK + BLiMP-arg + graph tasks — **re-register** (ADR-0030 refresh) on the
  corrected off-saturation venue.
- Analysis: `analysis/comparison_tests.py` fixed to **paired bootstrap on per-seed diffs**,
  one method per contrast, **≥10 seeds**, Holm–Bonferroni.
- **Information-parity gate (ADR-0035):** no GPU training until H(structure|kāraka)≫0,
  `VISAYATA` in 100% transitive strings, off-saturation venue, zero mock/CPU paths.
- Closure: six layers in `closure-contract.md` + `psalm contract check`.

---

## 10. Tooling used in this reframe

| Tool | Use in planning |
|---|---|
| **TRIZ MCP** | Separation in time/space for frame-supply vs realization acceptance contradiction |
| **Attractor-flow MCP** | Regime note: post-H1 null → CONVERGING on relocated H1′ claims, not COGS rescue |
| **Consolidation + ADRs 0017–0022** | Evidence and namespace discipline |
| **ce-plan / foundation index** | Prior Wave-1 unit map; superseded for execution by three workstreams in launch plan |

---

## 11. Out of scope (v2 spec)

- 350M COGS rescue
- `matrix.py` edits outside `integration/data-engine-v2`
- **CPU or mock anything in a reported result** (ADR-0035)
- **GPU training before the information-parity audit passes** (ADR-0035 D6)
- Training runs before ADR-0022 GB10 gate (unless waiver ADR)
- Claiming a Navya-Nyāya null before the information-parity gate passes
- Claiming full Vyutpattivāda coverage before coverage ledger M3
