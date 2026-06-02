# ADR-0030 — H1′ Paribhāṣā vs Dyck: blind pre-registration (sample-efficiency primary)

- Status: **Proposed (blind pre-registration)** — written before any H1′ B-vs-C
  learning-curve number is computed; human sign-off required before the proxy
  battery launches.
- Date: 2026-06-02
- Supersedes: the *live* structural headline from closed H1 (ADR-0017); reuses the
  sample-efficiency machinery from ADR-0016 without reopening COGS as the primary gate.
- Depends on: ADR-0017 (claim relocation), ADR-0018/0019 (L2 generator + Śabdabodha),
  ADR-0021 (`h1p:` namespace), ADR-0025 (matched Dyck + Hu replication reporting),
  ADR-0020 (BabyLM dual-track eval).

## Context — why H1′ is not “H1 again”

Phase-2 H1 tested whether a **kāraka-level Pāṇinian** prior (arm B) beats a
**surface-matched Dyck** control (arm C) on **COGS argument-role** readouts. After
three blind instrument refinements (EM → discrimination → sample-efficiency AUC),
the decisive contrast closed **NULL** at proxy scale: discrimination AUC(B−C) ≈
+0.003 with CI crossing zero, and the venue **saturated** (all arms ≥0.90 role
discrimination without structural pretraining — ADR-0017,
`docs/data/phase2-h1-cogs-calib.json`).

The salvageable scientific read was **sample-efficiency** (ADR-0016): the same
instrument over training time. Even that axis could not separate B from C on COGS
because every arm climbed to the ceiling too quickly.

**H1′ relocates the structural claim** to a different prior and a different venue:

| Dimension | Closed H1 | H1′ (this ADR) |
|---|---|---|
| Prior under test | L1 Pāṇinian kāraka composition | **L2 Paribhāṣā / Śabdabodha typed graphs** |
| Control | k-Shuffle Dyck (matched stats) | Same matched Dyck (+ Hu config reported) |
| Eval venue | COGS argument-role LF | **EWoK + BLiMP argument-structure** (+ graph probe secondary) |
| Mechanism prediction | Role transfer in LF grammar | Ontology / saṃsarga / epistemic plausibility |

If a typed semantic prior helps anywhere in the program’s BabyLM-scale regime, it
should appear on tasks that reward **relational and ontological structure**, not
on a venue that already saturates for bare NL baselines.

## Decision (pre-registered, blind)

### D1 — Claim and falsifiable null

**Claim (H1′):** At a fixed proxy architecture and matched structural dose, a
**Śabdabodha-aligned Paribhāṣā** pre-pretraining stream (`h1p:B`) yields **faster**
acquisition (higher sample-efficiency AUC) on the pre-registered semantic minimal-pair
venue than a **surface-statistics-matched Dyck** control (`h1p:C`), with concordant
secondaries.

**Null (H1′):** Δ_auc = AUC(B) − AUC(C) ≤ 0 with 95% bootstrap CI upper bound ≤ 0,
and concordant secondaries do not favor B — *“typed L2 prior confers no measurable
sample-efficiency advantage over matched Dyck on the H1′ venue at proxy scale.”*
A confirmed null is a **real result** and is reported without venue-shopping.

### D2 — Arms (`h1p:` namespace, ADR-0021)

All arms share `param_count_m`, tokenizer, downstream English budget, and
**matched-epoch** structural token budget (ADR-0014 dose rule). Ledger `arm_id` uses
the full prefixed id.

| Arm | ID | `PrePretrainSource` | Role |
|---|---|---|---|
| Baseline | `h1p:A` | `none` | No structural prior |
| **Treatment** | `h1p:B` | `shabdabodha_aligned` | L2: frozen `paribhasha_aligned_v1` rows; training line = `paribhasha_string` (ASCII channel per ADR-0026) |
| **Control** | `h1p:C` | `dyck` | k-Shuffle Dyck matched to B’s stream stats (`match_dyck` grid + byte-length check per ADR-0025) |
| L1 contrast | `h1p:L` | `paninian` | Optional: same matched dose as B/C; isolates L2 vs L1 (not the decisive pair) |

**Decisive pair:** `h1p:B` vs `h1p:C`. Comparisons involving `h1p:A` or `h1p:L` are
diagnostic only.

**Integration note (not edited in this worktree):** register the four arms in
`default_h1p_matrix()` on `integration/data-engine-v2` when the battery lands;
configs live under `configs/research/h1p/`.

### D3 — Matched-dose protocol

1. **Structural budget:** `pre_budget_tokens` = diversity-capped unique whitespace
   tokens per arm (same cap methodology as Phase-2 H1; default 60k unless manifest
   revises).
2. **Matched epochs:** repeat the capped unique set `pre_epochs` times for **every**
   pre-pretrain arm (B, C, L) — identical schedule to ADR-0014.
3. **Dyck match:** recompute `match_dyck` targets from a 2k-line sample of the B
   stream (TTR, bigram/trigram entropy); store chosen `DyckConfig` in run metadata.
   Report Hu replication distance as concordant diagnostic (ADR-0025).
4. **Downstream:** BabyLM Strict-track English mix to the arm’s `token_budget`
   (proxy: 13M tokens @ 100M-param class; battery: 130M per PRD).

### D4 — Primary eval venue (chosen to avoid H1 saturation)

**Primary venue (battery):** official BabyLM **minimal-pair PLL** suites:

1. **EWoK** (entity / part / behavior plausibility) — ontology-aligned; typed
   padārtha/saṃsarga priors should sharpen *which* relational assertions are likely.
2. **BLiMP argument-structure subsets** — `anaphor_agreement`, `argument_structure`,
   `binding`, `control_raising`, `ellipsis`, `quantifiers` (exact shard list frozen in
   `configs/research/h1p/go_no_go.yaml` before launch).

**Why this avoids the COGS failure mode:**

- H1 saturated because **COGS role discrimination** was already ≥0.90 for arm A
  without structural pretraining — no headroom for B vs C.
- EWoK and BLiMP-arg test **English semantic/syntactic minimal pairs** that are
  **not** the same LF-role-transfer instrument; proxy pilots are expected to land
  in the 0.55–0.85 band (off-floor, below ceiling) for A at early checkpoints.
- The L2 prior’s mechanism (typed graphs, inference scaffolds) aligns with **world
  and epistemic structure** probed by EWoK, not bracket matching alone.

**Venue dependency:** zero-shot PLL on real shards requires the **eval-train**
workstream (`invoke_official_zero_shot`, ADR-0020 / integration charter). Until
that lands, harness smokes may use **COGS noncanonical discrimination** or
**mock BLiMP/EWoK** pairs — numbers from smokes are **wiring only**, not evidence.

**Secondary venue (pre-registered, not gating):** Śabdabodha **graph-consistency**
minimal pairs — (gold `paribhasha_string`, corrupted graph: swap `PRAKARATA` target
or invert `sansa` on one edge). Off-floor at chance=0.50; reports whether L2 prior
helps **graph-typed** discrimination even when English PLL is flat.

### D5 — Primary metric (reuse ADR-0016 machinery)

Train each arm; at fixed downstream checkpoints record **minimal-pair accuracy**
(teacher-forced logprob, no generation) on the **frozen primary shard union**
(EWoK + BLiMP-arg items above).

- **Checkpoints (eval_fracs):** 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8,
  1.0 of the downstream step budget.
- **Threshold θ = 0.70** (above chance 0.50, below expected ceiling ~0.88 on
  combined shard — chosen **blind** with more headroom than H1’s θ=0.80 on COGS).
- **PRIMARY Mse:** normalized AUC of accuracy-vs-downstream-tokens (trapezoidal /
  budget), per arm per seed. Higher = faster acquisition.
- **PRIMARY contrast:** Δ_auc = mean(AUC(B) − AUC(C)) over seeds; bootstrap 95% CI.

**Concordant secondaries (must agree in direction with primary):**

1. **Tokens-to-threshold** T_θ: interpolated tokens to first reach θ; contrast
   Δ_eff = (T_θ(C) − T_θ(B)) / T_θ(C) (positive favors B).
2. **Early gap:** acc(B) − acc(C) at the **0.1-budget** checkpoint.
3. **Final-state PLL accuracy** on the same shard union (sanity; not primary).

### D6 — Effect criteria and verdict mapping (pilot → battery)

**Learnedness precondition:** baseline `h1p:A` reaches θ within budget on the primary
shard union.

**Effect criterion (CONJUNCTIVE, ALL required for `LAUNCH_RECOMMENDED`):**

1. `h1p:A` reaches θ within budget;
2. Δ_auc ≥ **0.02** with 95% bootstrap CI lower bound **> 0**;
3. Δ_eff > 0 **and** early gap > 0 (secondaries concordant);
4. Graph-consistency secondary (if run) does not **contradict** the primary direction
   (opposite sign with CI excluding 0 → `MIXED_READOUT`, human review).

| Outcome | Verdict | Action |
|---|---|---|
| All criteria met | `LAUNCH_RECOMMENDED` | Human sign-off → proxy battery |
| A reaches θ; primary/secondaries not all positive | `NO_OR_WEAK_EFFICIENCY_GAIN` | Report null; **no sixth instrument** |
| All arms ≥ θ at earliest checkpoint (0.005) | `VENUE_SATURATES` | Do not escalate to 350M on this venue; document saturation parallel to H1 |
| A never reaches θ | `NOT_LEARNED_AT_SCALE` | Escalate scale or fix eval-train wiring |
| Eval shards unavailable | `BLOCKED_ON_EVAL_TRAIN` | Run harness smoke only; battery waits |

**Stopping rule:** ADR-0030 is the H1′ instrument charter. If `NO_OR_WEAK` or
`VENUE_SATURATES`, report honestly and pivot effort to multi-target identity and
C-S2 — **do not** revert primary venue to COGS for a “win.”

### D7 — Autonomy and reporting commitment

- Execute pilot per `scripts/run_h1prime_pilot.py`; **no auto-launch** of the full
  battery.
- Commitment: publish negative H1′ results with the same prominence as a positive
  H1′ result; cite ADR-0017 + ADR-0030 for scope boundaries in the paper.

## Consequences

- `docs/experiments/h1prime-plan.md` tracks status **PRE-REGISTERED**.
- Thresholds mirrored in `configs/research/h1p/go_no_go.yaml` (human-editable but
  frozen at battery lock).
- Competition arms remain `comp:*` (ADR-0021); do not conflate with `h1p:*`.

## Alternatives considered

- **Retain COGS as H1′ primary:** rejected — venue saturated; would repeat H1’s
  null for the wrong scientific question.
- **Final-state EWoK accuracy only:** rejected — same ceiling risk as ADR-0015;
  sample-efficiency is the program thesis (ADR-0016).
- **Drop matched Dyck:** rejected — confounds “structure in general” with “typed
  L2 content” (Tarka memo Objection 3).
- **θ = 0.80 on EWoK:** rejected blind — COGS showed 0.80 too close to ceiling;
  0.70 chosen for headroom on the new venue.

## Links

- Harness: `scripts/run_h1prime_pilot.py`
- Config: `configs/research/h1p/go_no_go.yaml`
- Ledger: `docs/experiments/h1prime-plan.md`
- Closed H1: ADR-0017, `docs/contracts/phase-2-h1-tarka-memo.md`
- L2 generator: ADR-0018, ADR-0019, `docs/contracts/aligned-pair-schema.json`
