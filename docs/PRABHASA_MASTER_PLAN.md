# PRABHĀSA Master Program Plan: Integrating Design + Research Tracks

**Date:** 2026-06-05  
**Status:** Locked plan (gates, dependencies, roles, iteration spec)  
**Owner:** Claude Code agents (orchestrated workflow)  
**Duration:** ~4–6 weeks (strict-small 3-seed → compute optimization → Small 1 seed → iterate to ≤0.5pp gain threshold)

---

## Executive Summary

PRABHĀSA (प्रभास, "illumination") is a **six-track research+infrastructure program** that unifies:

1. **Vision & roadmap** (Stage 0: BabyLM 2026, Stages 1–2: SLM/LLM scale)
2. **Compute optimization** (Fix A: seq_schedule tuning; Fix B: cuDNN backend; Fix C/D/E: longer-term)
3. **Corpus-from-grammar generation** (Vidyut → Paribhāṣā → Vyutpattivāda → Navya-Nyāya pipeline)
4. **Small-track corpus sourcing** (100M English + optional Sanskrit dose)
5. **Paper rigor audit & findings integration** (H1_MECHANISM table fills, H2 scoping, ablation design)
6. **HF Hub publishing & demos** (Model + dataset cards, Colab notebooks, reproducibility)

**Critical constraint:** GPU is 95% saturated with strict-small training (3 seeds running now). All work listed below is DESIGN + RESEARCH + WRITING. No training jobs launch until strict-small completes and compute fix is validated.

---

## Phase-Locked Sequence & Timeline

### PHASE 1: Strict-Small Battery (3 seeds) — IN PROGRESS, ~48h remaining

**What:** Training `prabhasa-b_ss-0.1` on 10M BabyLM tokens, 3 seeds, final model eval on BLiMP.

**Status:**
- Seed ~0 run: ~52.68% BLiMP (preliminary, 8M checkpoint; full 10M results pending)
- Seed ~1, ~2: in flight; ETA completion ~2026-06-06 noon UTC

**Deliverable:** Three 10M checkpoints + final mean±95% CI on BLiMP, go/no-go decision (≥70.0 threshold).

**Gate (Closure Contract):**
- TECHNICAL: Tests pass (ruff + mypy + coverage ≥80%); no infrastructure changes until results lock.
- EMPIRICAL: All 3 seeds complete; PLL scores computed; go/no-go logged.
- INTEGRITY: Tarka memo (strongest objections to interpretation) written; comparison fairness verified.
- ARTIFACTS: Results pushed to repo; results table frozen (no changes post-hoc).
- MEMORY: Ledger entry (seed #, checkpoint path, BLiMP score, interpretation) in `docs/memory/ORCHESTRATOR-STATE.md`.
- SIGN-OFF: Human review before proceeding to Phase 2.

**Owner:** [Currently running; human operator monitors]

---

### PHASE 2: Compute Optimization Fix A + Paper Rigor Tier-1 — IMMEDIATE (0.5–3h of CPU work)

**What:** Apply Fix A (reduce seq_len=256 phase fraction) while strict-small completes; write paper sections.

**Parallel tracks (non-GPU):**

#### Track 2a: Fix A Implementation & Validation Plan (Effort: 0.5h)
- **Code change:** Modify `seq_schedule` in `train_submission_model.py` (lines 211–214)
  - Cap final seq at 192 (avoid 0.25→4.4 s/step cliff)
  - Phase: 0–40% seq=64/128, 40–70% seq=192, 70–90% seq=192, 90–100% seq=256 (final hardening at slow speed)
  - Expected speedup: 1.5–2× (0.25 → 0.4–0.5 step/s)
- **Validation:** Draft smoke test script (`scripts/bench_seq_schedules.py`) to confirm 2× speedup at seq=192 vs 256
  - **Do NOT run on GPU** (saturated); note for execution after Phase 1 completes
- **Gate:** Code merged to feature branch; smoke test documented; ready to run post-Phase-1
- **Owner:** Agent (Design track)

#### Track 2b: Paper Rigor Tier-1 Blocker Fixes (Effort: 1.5h writing)

**Priority H (Must-fix before submission):**

1. **Fill H1_MECHANISM Table 3 (BLOCKER)**
   - Action: Wait for Phase-1 results; extract mean±95% CI from 3 seeds
   - Template: Create `docs/tables/h1_mechanism_final_blimp.csv` with placeholder structure now
   - Entries: BLiMP Text Average, BLiMP Math Average, go/no-go decision
   - **When Phase-1 completes:** Plug numbers in; no other changes to table
   - Effort: 30 min (template) + 10 min (fill) = 40 min total
   - Owner: Agent (Data/paper track)

2. **H2 Results Section Scope Decision**
   - Decision: Is H2 synergy test results available by submission? (Timeline check: fine-tuning takes ~2–4h on free compute)
   - If YES: Write preliminary results section (~1h writing)
   - If NO: Rewrite abstract + intro to highlight H1_MECHANISM as the single live contribution; move H2 to "Future Work" (30 min rewrite)
   - Gate: Decision locked by Phase-2 end; abstract rewritten by 2026-06-06 EOD
   - Owner: Agent (Paper track)

3. **Arm D Invalidation Clarification**
   - Write footnote in main text: "Arm D (Paribhāṣā) was rebuilt after ADR-0034/0035 validation found prior implementation did not faithfully implement Vyutpattivāda. Results from the corrected arm D will be reported [when run completes / in future work]."
   - Alternative: Exclude arm D from main ablation; reserve for supplementary/future work
   - Gate: One-sentence decision in paper by 2026-06-06 noon
   - Owner: Agent (Paper track)

**Timeline:**
- **2026-06-05 now:** Tracks 2a + 2b launch in parallel
- **2026-06-06 noon:** Phase-1 results expected; Paper Tier-1 fixes applied
- **2026-06-06 EOD:** Feature branch ready; paper decision points locked

**Gate (Pass/Fail):** Fix A code merged + validated; Tier-1 paper blocker fills submitted

---

### PHASE 3: Compute Validation & Paper Tier-2 (Medium Priority) — 2026-06-06 afternoon, ~2–4h

**What:** Verify Fix A works (smoke test on free GPU after Phase-1); write ablation specs + Related Work.

**Dependencies:** Phase-1 results in hand; Fix A code ready to test.

#### Track 3a: Fix A Validation Run (Effort: 0.5h active)
- **Action:** Run smoke test on 100-step batch at seq=192 vs seq=256; measure step/s
- **Target:** ≥1.5× speedup confirmed (0.25 → 0.375+ step/s)
- **Outcome:**
  - If ≥1.5×: Approve Fix A; proceed to Phase-4 (100M Small run with optimized schedule)
  - If <1.5×: Escalate to Fix B (cuDNN backend); apply in parallel with Phase-4 training
- **Gate:** Smoke test results logged in `docs/memory/compute_optimization_status.md`
- **Owner:** Agent (Compute track)

#### Track 3b: Mechanism Ablation Design & Supplementary Results Plan (Effort: 1.5h)

**Priority M (Strongly recommended for BabyLM 2026 workshop):**

1. **Ablation battery spec** (supplementary table or main, TBD):
   - 5 conditions: Baseline AMLM, +N-hot morpheme embeddings, +N-hot + kāraka masking, +masking + salience transfer, +all three
   - 3 seeds each → 15 total runs (out of scope for now; design spec only)
   - Holm–Bonferroni-corrected contrasts; identify minimum-mechanism subset to reach BLiMP ≥70.0
   - **Action:** Write spec + registration document; schedule for Phase-5+ (post-100M run)
   - Effort: 30 min spec + 20 min registration doc = 50 min
   - Owner: Agent (Experimental design track)

2. **Related Work Extension** (Effort: 40 min):
   - Search BabyLM 2024–2025 leaderboard winners (top 3–5 submissions); note mechanisms
   - Add 1–2 paragraphs comparing PSALM to recent structural-prior work (Coavoux & Crabbé 2019 + ≥1 recent 2023–2025 work on syntax/morpheme-aware masking)
   - Cite the BabyLM 2023 and 2025 winners explicitly; clarify PSALM novelty
   - Action: Expand `docs/bib/related_work_extended.bib`; integrate into paper Related Work section
   - Owner: Agent (Literature/writing track)

#### Track 3c: Hyperparameter Appendix Specification (Effort: 30 min)

**Priority M:**
- Muon optimizer: beta, epsilon, ortho threshold, update frequency
- LoRA fine-tune (H2): learning rate, warmup, epochs, patience (validation metric)
- Z3 vyāpti: predicate signatures, depth/arity bounds
- Action: Template `docs/hyperparameters/final_spec.yaml`; populate from code inspection + trial runs
- Owner: Agent (Reproducibility track)

**Timeline:**
- **2026-06-06 afternoon:** Track 3a (smoke test) runs; 2–3h turnaround
- **2026-06-06 EOD:** Ablation spec + Related Work + hyperparameter appendix drafted

**Gate (Pass/Fail):** Fix A validated (or Fix B escalation initiated); Tier-2 paper specs completed

---

### PHASE 4: 100M Small Track Training + Corpus Sourcing Prep — 2026-06-06 evening → 2026-06-10 EOD

**What:** Launch Small (100M token) pretraining with optimized compute schedule; prep corpus sources for iteration loop.

**Dependencies:** Phase-3 validation complete (Fix A ≥1.5×); compute schedule locked.

#### Track 4a: Small-Track (100M) Training Pipeline — Main Compute Effort
- **Config:** `configs/phase2/arm_B_paninian_en.yaml` (or equivalent Small variant)
- **Corpus:** BabyLM 2026 Strict (100M English) + optional Sanskrit dose (outside 100M cap per ADR-0020)
- **Sequence schedule:** Fixed-A optimized (70–80% at seq=192, final 20–30% at seq=256)
- **Model:** Same ELC architecture as Strict-Small; estimated wall-time ~10–12h (vs 20–30h at original speed)
- **Checkpoints:** Save at 1M, 5M, 10M (proxy), 50M, 100M (final) for ablation + analysis
- **Seed:** Single seed for now (per user timeline); 3-seed ablation in Phase-5
- **Owner:** [GPU-bound; human operator monitors training]

#### Track 4b: Corpus Sourcing Validation & Manifest (Effort: 1.5h CPU work)

**Goal:** Prepare corpus sources for iteration loop; verify manifests + download integrity.

**Actions:**
1. **English (BabyLM Strict 100M):**
   - Verify local cache (if available) against official HF dataset
   - Create `corpus_manifest.yaml` with per-source word counts, dedup hashes, license, epochs per seed
   - Effort: 30 min

2. **Sanskrit (optional dose, non-cap-counted per ADR-0020):**
   - Metadata fetch: Sangraha (HF API), GRETIL (URL verify), DCS (GitHub clone)
   - Estimate tokenized word counts (sampling 1000-line subsets)
   - Log license status: Sangraha (CC-BY-4.0 ✓), DCS (CC-BY-4.0 ✓), GRETIL (⚠️ UNCLEAR—flag in docs)
   - Effort: 45 min

3. **Manifest finalization:**
   - Output: `corpus_manifest.yaml` with structure:
     ```yaml
     english:
       babylm_strict: {words: 100M, sources: 5, hash: ..., license: MIT}
     sanskrit:
       sangraha: {words: 14.9B, license: CC-BY-4.0, status: verified_metadata}
       dcs: {words: 10–15M, license: CC-BY-4.0, status: verified_cloned}
       gretil: {words: 100–200M, license: UNCLEAR_FLAG, status: needs_contact}
     ```
   - Effort: 15 min

**Gate (Pass/Fail):** Manifest complete + hashes verified; Small training launched cleanly

---

### PHASE 5: Iteration Loop → Version Bumps (2026-06-10 → 2026-06-20)

**What:** Evaluate Small (100M) 1-seed results; decide mechanism/corpus changes; iterate until <0.5pp BLiMP gain threshold hit (STOP RULE).

**Timeline:** Assumed Small training completes ~2026-06-10 EOD; iteration loop begins 2026-06-11.

**Iteration Loop Specification (see section below):** Each round = {research → propose change → train 1 seed → eval → compare → version bump if gain ≥0.5pp}.

**Owner:** Claude Code subagent (orchestrates research + training cycles)

---

### PHASE 6: Paper Finalization + HF Publishing (Post-iteration, ~2026-06-20 onward)

**What:** Lock paper findings; fill ablation results; publish model + dataset to HF Hub.

**Dependencies:** Iteration loop complete (stop rule triggered); final Small model checkpoint selected.

#### Track 6a: Paper Tier-3 Polish (30–60 min)
- Figure quality: BLiMP dev curve with error bands (3 seeds if ablation done)
- Config hash in results table
- Limitations update: arm D impact, mechanism non-independence
- Grammar review + citation scan (verify all links resolve)

#### Track 6b: HF Hub Publishing (Effort: 1–2h)
- Export final model to HF format (safetensors + config + tokenizer)
- Create model card from template; fill [RESULT] placeholders with actual BLiMP scores
- Create dataset card for corpus
- Upload both to HF Hub under `qbz506/` org
- Test loading in Colab; verify demo notebooks run
- Add Colab badges to model card

**Owner:** Agent (Reproducibility + publishing track)

---

## Iteration Loop Specification (PRABHĀSA Core Research Cycle)

Each iteration round follows this precise flow:

```
ROUND N (N=1,2,3,...):
  1. Research & Ideation (2–4h)
     - Analyze Round N-1 results (BLiMP gaps, error patterns, loss curves)
     - TRIZ contradiction analysis (if needed): identify bottleneck
     - Brainstorm mechanism changes (e.g., adjust masking probabilities, add new loss objective)
     - Brainstorm corpus changes (e.g., rebalance Sanskrit dose, change sampling rate)
     - Hypothesis: "Changing X will gain ≥0.5pp on BLiMP because Y"

  2. Design Specification (1h)
     - Write change spec: code diffs, config adjustments, corpus modifications
     - Pre-register: expected effect size, validation metric, freeze decision before training

  3. Training (1 seed, ~10–12h wall-time for 100M)
     - Config: Small (100M), single seed
     - Run Small training with proposed changes
     - Checkpoint every 10M tokens

  4. Evaluation (1h)
     - Compute BLiMP (Text Average + Math Average) on final checkpoint
     - Compute PLL for all ≥12 phenomena; confidence intervals (bootstrap, n=1000 resamples)
     - Compare to baseline (Round N-1): Δ BLiMP = (Round N) - (Round N-1)

  5. Comparison & Decision (30 min)
     - Is Δ BLiMP ≥ 0.5 percentage points?
       - YES: Approve change; keep model; bump version (0.1 → 0.2)
       - NO: Reject change; roll back; investigate why

  6. Ledger & Memory Update (30 min)
     - Log: Round #, hypothesis, change spec, BLiMP score, Δ BLiMP, decision, interpretation
     - Example entry:
       ```yaml
       round: 3
       hypothesis: "Raising masking prob for obliques (karma, karaṇa) from 25% → 30% will reduce compositional errors in COGS by improving role discrimination"
       change: "Modified kaaraka_masking_config.yaml: karma_mask_prob: 0.25 → 0.30"
       result:
         blimp_text_avg: 68.2%  # vs 67.8% Round 2
         delta_blimp: +0.4pp
         decision: REJECT (gain < 0.5pp threshold)
       interpretation: "Role-awareness is saturated at current corpus size; gain requires more diverse training data, not masking tuning."
       next_hypothesis: "Try Round 4: expand corpus diversity (add more verb frames) instead."
       ```
     - Update `docs/memory/ORCHESTRATOR-STATE.md` + version file

STOP RULE: If Δ BLiMP < 0.5pp for 2 consecutive rounds → stop iteration loop
           (diminishing returns detected; lock final model)
```

**Key Metrics & Measurement:**

| Metric | Definition | Target | Measurement |
|--------|-----------|--------|-------------|
| **BLiMP Text Average** | Mean accuracy across 12 phenomena (grammaticality judgment) | ≥70.0 | PLL scoring (pseudo-log-likelihood) via `psalm eval blimp` |
| **BLiMP Math Average** | Subset of logical/mathematical phenomena (control, quantifier scope, etc.) | ≥70.0 | Same as Text |
| **Δ BLiMP (round delta)** | Improvement Round N vs N-1 | ≥0.5pp to continue; <0.5pp = stop | Bootstrap CI on difference; Holm–Bonferroni if comparing >2 arms |
| **Config Hash** | SHA-256 of resolved config + seed | Recorded per run | `python -c "from psalm.config import config_hash; print(config_hash())"` |
| **Iteration Count** | Number of completed rounds | N/A (stop rule terminates) | Ledger count |

**Version Numbering:**
- Major version: research iteration (0 = BabyLM 2026 / H1_MECHANISM, 1 = H1' evolved / SLM scale)
- Minor version: recipe tweak within iteration (0.1 → 0.2 after approved round; 0.2 → 0.3 after second approved round, etc.)
- Example: `prabhasa-b_s-0.3` = PRABHĀSA BERT, Small corpus, iteration 0, version 3 (3 approved changes applied)

**Research Agents' Responsibilities:**

1. **Research Agent (TRIZ + Ideation):** Analyzes failure modes; proposes mechanisms; runs TRIZ matrix if contradictions arise
2. **Config Agent (Spec & Validation):** Writes change spec; updates configs; validates schema
3. **Training Agent (GPU Orchestration):** Launches Single-seed Small runs; monitors job status
4. **Eval Agent (Metrics & Comparison):** Computes BLiMP + confidence intervals; logs results; decides go/no-go
5. **Memory Agent (Ledger & Documentation):** Updates experiment ledger; archives configs; writes interpretations

---

## Master Gate Map: Closure Contract Alignment

Each phase must satisfy the **six-layer Ralph-loop closure contract** (from CLAUDE.md):

### Closure Gates by Phase

| Phase | TECHNICAL | EMPIRICAL | INTEGRITY | ARTIFACTS | MEMORY | SIGN-OFF |
|-------|-----------|-----------|-----------|-----------|--------|----------|
| **1 (3-seed Strict-Small)** | Tests ≥80% coverage; no CI errors | 3 seeds complete; BLiMP mean±CI; go/no-go decision | Tarka memo (strongest objections written) | Results pushed; tables frozen | Ledger entry + config hash | Human review before Phase 2 |
| **2 (Fix A + Paper Tier-1)** | Code compiles; ruff/mypy clean | Fix A smoke test plan drafted; Paper Table 3 template ready | N/A (design phase) | Feature branch merged | Spec documented | — |
| **3 (Validation + Tier-2)** | Smoke test runs cleanly | Fix A speedup confirmed (≥1.5×) OR escalate to Fix B | Ablation spec registered (frozen before training) | Specs + Related Work + hyperparams drafted | Compute status doc updated | — |
| **4 (100M Small)** | Small config passes schema validation | Small 1-seed training complete; 100M checkpoint saved; BLiMP computed | Corpus manifest verified; license clarity | Checkpoints pushed; manifest in repo | Round-1 ledger entry | — |
| **5 (Iteration loop)** | Each round config valid | Round N BLiMP computed; Δ BLiMP logged; stop rule evaluated | Each round hypothesis pre-registered (frozen before training) | Configs versioned; results table updated per round | Ledger entry per round | Go/no-go decision per round |
| **6 (Paper + HF Pub)** | Model exports cleanly; Colab runs | Ablation results filled in; all placeholders resolved | Final paper peer-reviewed; citation scan passed | Checkpoint + dataset on HF; READMEs complete | Paper section finalized; ledger archived | — |

---

## Roles & Teams

### Design Tracks (CPU, concurrent with GPU training)

| Track | Owner (Role) | Responsibility | Timeline | Gate |
|-------|--------------|-----------------|----------|------|
| **Compute Opt** | Agent (Compute) | Fix A implementation + smoke test spec; Fix B/C/D as fallback | Phase 2–3 | Fix A ≥1.5× validated by Phase-3 EOD |
| **Paper Rigor** | Agent (Writing) | Tier-1 blocker fills; Tier-2 specs; Tier-3 polish | Phase 2→6 | Tier-1 by Phase-2 EOD; full paper by Phase-6 |
| **Corpus Sourcing** | Agent (Data) | Manifest generation; license verification; download prep | Phase 4 | Manifest complete + hashes verified |
| **Ablation Design** | Agent (Experiments) | 5-condition spec; registration document; result interpretation | Phase 3→5 | Spec frozen before Phase-4 training |
| **HF Publishing** | Agent (Reproducibility) | Model/dataset export; card templates; Colab testing | Phase 6 | Model + dataset live on HF Hub by Phase-6 EOD |

### Compute (GPU-bound)

| Phase | Task | Expected Duration | Resource |
|-------|------|-------------------|----------|
| **Phase 1** | 3-seed Strict-Small (10M) | ~48h total (3 seeds × 16h) | DGX Spark (95% saturated now) |
| **Phase 4** | 1-seed Small (100M) | ~10–12h wall-time | DGX Spark (freed after Phase-1) |
| **Phase 5** | Iteration loop (N rounds × 100M) | ~10–12h per round × N rounds | DGX Spark (streaming, continuous) |

---

## Critical Dependencies & Blockers

### Hard Dependencies (Must complete before next phase)

| Dependency | Trigger | Consequence |
|------------|---------|-------------|
| **Phase 1 → Phase 2** | 3 seeds complete | Cannot fill H1_MECHANISM Table 3; paper remains incomplete |
| **Phase 2 → Phase 3** | Fix A code merged | Cannot validate speedup; compute timeline at risk |
| **Phase 3 → Phase 4** | Fix A validated (≥1.5×) | If fails, escalate to Fix B; delays Small training by 2–4h |
| **Phase 4 → Phase 5** | 100M Small checkpoint saved | Cannot begin iteration loop; schedule slips |
| **Phase 5 → Phase 6** | Stop rule triggered (Δ BLiMP <0.5pp ×2) | Cannot finalize paper; publish timing deferred |

### Soft Dependencies (Recommended for quality; can slip)

| Dependency | Trigger | Consequence |
|------------|---------|-------------|
| **Tier-2 paper** | By Phase-3 EOD | If delayed, paper review timeline compresses |
| **Ablation spec freeze** | By Phase-3 EOD | If delayed, Phase-5 ablation runs may not start on time |
| **HF export** | Before Phase-6 | If delayed, publication announcement window shrinks |

---

## Risk Registry & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| **Fix A doesn't yield ≥1.5× speedup** | Medium (20%) | Phase-4 training delayed; timeline slips 1–2 days | Escalate to Fix B (cuDNN backend) in parallel; budget 2h for testing |
| **Iteration loop hits stop rule after 1 round** | Medium-High (30%) | Paper claims weaken; requires reframing to "diminishing returns" narrative | Pre-register stop rule in abstract; rewrite to honest null finding (valid closure state) |
| **H2 fine-tuning does not complete by deadline** | High (40%) | Reposition H2 as future work; H1_MECHANISM becomes sole live claim | Decision gate at Phase-2 EOD; plan abstract rewrite contingently |
| **Corpus sourcing reveals license issue (e.g., GRETIL)** | Low (10%) | May exclude Sanskrit dose; limits novelty narrative | Verify GRETIL license with Göttingen U.; use Sangraha + DCS only if GRETIL unclear |
| **BLiMP submission gate fails (paper incomplete, placeholders unfilled)** | Low (5%) if Phase 1 completes on time | Miss workshop deadline; deferred to future venue | Lock Phase-1 results by 2026-06-06 noon; freeze paper table fills by EOD |
| **Single GPU failure (DGX Spark crash)** | Very Low (2%) | Loss of Phase 4 checkpoint; restart from last saved | Enable checkpointing every 10M; keep 3 recent checkpoints on disk (redundancy) |

---

## Iteration Loop: Worked Example (Rounds 1–3 scenario)

Assume Phase-4 (Small 100M) produces baseline: **BLiMP = 67.2%** (below 70.0 target).

### Round 1: Masking Probability Tuning

**Research (2h):**
- Analysis: COGS (argument role) accuracy is 65.2%, dragging down mean. Likely cause: current masking does not distinguish kāraka roles well.
- Hypothesis: "Increase masking probability for oblique roles (karaṇa, adhikaraṇa) from 25% → 35%; keep core roles (kartā, karma) at 35%; this forces the model to infer oblique semantics from context."
- TRIZ framing: Trade-off between masking coverage (drive learning) vs. robustness (preserve signal). Principle 2 (Extract/Take out): remove masking uniformity; apply role-based hierarchy.

**Design Spec (30 min):**
```yaml
change_id: "round1_masking_oblique_boost"
file: "configs/mechanisms/kaaraka_masking_config.yaml"
diff:
  - line: 24
    old: "oblique_mask_prob: 0.25"
    new: "oblique_mask_prob: 0.35"
  - line: 26
    old: "core_mask_prob: 0.35"
    new: "core_mask_prob: 0.40"
expected_effect_size: "+1.5pp BLiMP"
validation_metric: "blimp_text_avg"
stop_condition: "if delta_blimp < +0.5pp → reject"
```

**Training (10h):**
- Launch Small (100M) with updated config; seed ~1
- Checkpoints: 1M, 10M (proxy), 100M (final)

**Evaluation (1h):**
- Result: **BLiMP = 68.6%** (vs 67.2% baseline)
- Δ BLiMP = +1.4pp ✓ (≥0.5pp threshold met)

**Decision:** **APPROVE**
- Version bump: `prabhasa-b_s-0.1` → `prabhasa-b_s-0.2`
- New baseline for Round 2: 68.6%

**Ledger Entry:**
```yaml
round: 1
hypothesis: "Masking oblique roles at higher probability forces role-semantic learning"
change_spec: "kaaraka_masking_config.yaml: oblique_mask_prob 0.25→0.35, core_mask_prob 0.35→0.40"
training_duration: "10h, seed~1"
result:
  blimp_text_avg: 68.6%
  blimp_math_avg: [TBD]
  delta_blimp: +1.4pp
decision: APPROVE (Δ ≥ 0.5pp)
interpretation: "Role-stratified masking is effective; core roles benefit from tighter masking. Mechanism shows promise."
version_bump: "0.1 → 0.2"
next_round_baseline: 68.6%
```

---

### Round 2: Salience Transfer from Dose Stage

**Research (2h):**
- Analysis: Round 1 improved role discrimination (COGS up); but agreement phenomena (Subject-Verb, Determiner-Noun) still below 70%.
- Hypothesis: "The dose (Pāṇinian pre-pretraining) computed morpheme salience; we should transfer this prior as an initialization for the English pretraining stage. Reweight English masking by 1 + α × dose_morpheme_freq, where α=0.2."
- Expected effect: +1–2pp on phenomena where morpho-syntactic agreement drives accuracy.

**Design Spec (30 min):**
```yaml
change_id: "round2_salience_transfer"
file: "configs/mechanisms/salience_transfer_config.yaml"
new_file: true
content:
  enabled: true
  dose_morpheme_freq: "path/to/dose_stage_freq.json"
  alpha: 0.2
  apply_to_roles: ["kartA", "karma", "kriya"]  # Core roles only
expected_effect_size: "+1.5pp BLiMP"
validation_metric: "blimp_agreement_phenomena"  # Subject-Verb, Det-N, etc.
```

**Training (10h):**
- Launch Small with salience transfer enabled; seed ~2

**Evaluation (1h):**
- Result: **BLiMP = 69.5%** (vs 68.6% baseline)
- Δ BLiMP = +0.9pp ✓ (meets threshold)
- Agreement phenomena: +2.1pp; object agreement: +0.3pp

**Decision:** **APPROVE**
- Version bump: `prabhasa-b_s-0.2` → `prabhasa-b_s-0.3`

**Ledger Entry:**
```yaml
round: 2
hypothesis: "Dose-stage morpheme salience prior accelerates agreement-phenomenon learning in English stage"
change_spec: "salience_transfer_config.yaml (new): alpha=0.2, apply to core roles"
training_duration: "10h, seed~2"
result:
  blimp_text_avg: 69.5%
  agreement_phenomena: +2.1pp (strong)
  object_agreement: +0.3pp (weak)
  delta_blimp: +0.9pp
decision: APPROVE (Δ ≥ 0.5pp; strong on targeted phenomena)
interpretation: "Morpho-syntactic priors transfer effectively; further tuning may reach 70.0 threshold."
version_bump: "0.2 → 0.3"
next_round_baseline: 69.5%
```

---

### Round 3: Corpus Diversification

**Research (2h):**
- Analysis: Rounds 1–2 plateaued at 69.5%; remaining 0.5pp gap narrows toward diminishing returns. Error analysis: ellipsis, filler-gap, and quantifier phenomena remain <65%.
- Hypothesis: "Current corpus (BabyLM 100M) under-represents complex embedded structures (relative clauses, long-range dependencies). Add synthetic Paribhāṣā corpus (50K examples) to boost training diversity on these phenomena."
- Expected effect: +0.7pp on complex phenomena; risk: if corpus is low-quality, may hurt overall BLiMP.

**Design Spec (30 min):**
```yaml
change_id: "round3_corpus_augment_paribhasha"
file: "configs/corpora/small_track_curriculum.yaml"
diff:
  - layer: "L4"  # Dominant layer (BabyLM English)
    weight: 0.95 (was 1.0)
  - layer: "L3"  # Augmentation layer (Paribhāṣā synthetic)
    new: true
    corpus: "prabhas-paribhasha-50k-round3.jsonl"
    weight: 0.05
    sampling_strategy: "every_Nth"
expected_effect_size: "+0.5–1.0pp BLiMP (on complex phenomena)"
validation_metric: "blimp_filler_gap, blimp_ellipsis, blimp_quantifiers"
risk: "If synthetic corpus has low quality, may degrade overall mean"
```

**Training (10h):**
- Launch Small with augmented corpus; seed ~3

**Evaluation (1h):**
- Result: **BLiMP = 69.8%** (vs 69.5% baseline)
- Δ BLiMP = +0.3pp ✗ (below 0.5pp threshold)
- Filler-gap: +0.8pp (good)
- Ellipsis: –0.2pp (regressed)
- Quantifier: +0.1pp (minimal)

**Decision:** **REJECT** (Δ < 0.5pp; mixed signal on targeted phenomena)
- Revert corpus to BabyLM 100M only
- Baseline for Round 4 remains: 69.5%

**Ledger Entry:**
```yaml
round: 3
hypothesis: "Paribhāṣā synthetic corpus augmentation boosts complex-structure generalization"
change_spec: "small_track_curriculum.yaml: L4 weight 1.0→0.95, add L3 (prabhas-paribhasha-50k) weight 0.05"
training_duration: "10h, seed~3"
result:
  blimp_text_avg: 69.8%
  filler_gap: +0.8pp (positive)
  ellipsis: -0.2pp (regressed)
  quantifier: +0.1pp (minimal)
  delta_blimp: +0.3pp
decision: REJECT (Δ < 0.5pp; mixed signal suggests corpus quality issue)
interpretation: "Paribhāṣā corpus has high-quality examples (filler-gap improved) but some noise (ellipsis regressed). Larger or curated subset needed."
next_hypothesis: "Round 4: try smaller, hand-curated Paribhāṣā subset (10K) or focus on improving English corpus diversity instead."
```

---

### Round 4 Onwards → Stop Rule

**Research (2h):**
- Analysis: Rounds 2–3 achieved Δ < 0.5pp; we are in diminishing-returns regime.
- Decision: **STOP ITERATION** (2 consecutive rounds with Δ < 0.5pp detected per stop rule)

**Final Model:** `prabhasa-b_s-0.3` (BLiMP = 69.5%)
- Below the 70.0 pre-registered threshold; **finding is NULL with 2 interventions documented** (Rounds 1–3)
- Paper reframes: "PSALM achieves 69.5% on BLiMP after 3 optimization rounds; gap to 70.0 threshold indicates that the current mechanisms and corpus size approach saturation. H2 (Nyāya scaffold) is positioned as the post-pretraining pathway to further gains."

**Ledger Entry (STOP):**
```yaml
stop_rule_triggered: true
reason: "2 consecutive rounds (2, 3) with Δ BLiMP < 0.5pp; diminishing returns detected"
final_model: "prabhasa-b_s-0.3"
final_blimp: 69.5%
threshold_status: "below 70.0 pre-registered target"
interventions: 3
finding_classification: "MARGINAL NULL with documented interventions (valid closure)"
recommendation: "Post-pretraining via H2 (Nyāya scaffold) may unlock remaining 0.5pp gap; deferred to Phase 6 experiments or future work"
```

---

## Implementation Workflow & Execution Notes

### For Claude Code Subagents

1. **Phase Orchestration:**
   - Each phase has a **captain agent** (e.g., Compute Agent for Phase 2–3)
   - Captain coordinates parallel tracks; reports daily status in `docs/memory/ORCHESTRATOR-STATE.md`
   - Gate criteria are binary; no partial gates

2. **Configuration Management:**
   - All changes are **config-first** (no magic numbers in code)
   - Config diffs are version-controlled; each change is tagged with `change_id` (e.g., `round1_masking_oblique_boost`)
   - Resolved config hash is recorded per run for reproducibility

3. **Ledger & Memory:**
   - Experiment ledger is the source of truth for all decisions
   - Entries are **immutable** (append-only); edits are marked with timestamps + reason
   - Example structure: `docs/memory/phase_X_ledger.yaml`

4. **Stopping & Reversions:**
   - If a round is rejected, the agent rolls back code/configs to the last approved state
   - No "partial approvals"; either change is adopted (version bump) or rejected (revert)

---

## Timeline Summary (Gantt-style)

```
2026-06-05 now
  ├─ Phase 1 (Strict-Small 3 seeds) [████████░░] ~48h remaining, ETA 2026-06-06 noon
  │
  ├─ PHASE 2 (Parallel): Compute Fix A + Paper Tier-1 [2026-06-05 → 2026-06-06 EOD]
  │  ├─ Track 2a: Fix A impl + smoke test spec (0.5h)
  │  └─ Track 2b: Paper Tier-1 blocker fills (1.5h writing)
  │
  ├─ PHASE 3: Validation + Tier-2 [2026-06-06 afternoon → EOD]
  │  ├─ Track 3a: Fix A smoke test (2–3h turnaround)
  │  ├─ Track 3b: Ablation spec + Related Work (1.5h writing)
  │  └─ Track 3c: Hyperparameter appendix (0.5h)
  │
  ├─ PHASE 4: 100M Small training + corpus sourcing [2026-06-06 evening → 2026-06-10 EOD]
  │  ├─ Track 4a: Small training (10–12h wall-time, GPU-bound)
  │  └─ Track 4b: Corpus manifest (1.5h CPU)
  │
  ├─ PHASE 5: Iteration loop [2026-06-11 → 2026-06-18 (assumed 3 rounds)]
  │  └─ Each round: 2h research + 1h spec + 10h training + 1h eval + 30min decision = ~14.5h per round
  │
  └─ PHASE 6: Paper finalization + HF publishing [2026-06-18 → 2026-06-20]
     ├─ Track 6a: Paper Tier-3 polish (1h)
     └─ Track 6b: HF Hub upload (1–2h)

Total elapsed (critical path): ~15 days (Phase 1 → 6)
```

---

## Success Criteria (Closure Gates)

| Gate | Criterion | Verification |
|------|-----------|--------------|
| **Phase 1 Complete** | 3 seeds finish; mean±95% CI reported; go/no-go decision logged | `psalm contract check <phase1_report.json>` exits 0 |
| **Phase 2 Complete** | Fix A code merged; Table 3 template ready; H2 scope decision made | Code review + paper outline reviewed |
| **Phase 3 Complete** | Fix A speedup ≥1.5× confirmed OR Fix B escalated; Tier-2 specs drafted | Smoke test log + spec documents in repo |
| **Phase 4 Complete** | 100M checkpoint saved; BLiMP computed; corpus manifest finalized | Checkpoint file exists; manifest YAML validates |
| **Phase 5 Complete** | Stop rule triggered; final model selected; ≥2 rounds documented | Ledger shows Δ < 0.5pp ×2; final version locked |
| **Phase 6 Complete** | Paper tables filled; model + dataset on HF Hub; Colab notebooks run | HF Hub repos public; model loads via `transformers.AutoModel.from_pretrained()` |

---

## References & Linked Documentation

- **VISION.md**: PRABHĀSA roadmap (Stages 0–2, hypothesis definitions, naming scheme)
- **compute_optimization_gb10.md**: Detailed analysis + ranked fixes; Fix A spec
- **corpus_from_grammar_design.md**: Pipeline architecture; Vidyut→Paribhāṣā→Vyutpattivāda→Navya-Nyāya stages
- **small_track_corpus_sourcing.md**: BabyLM 100M + Sanskrit dose sourcing; license verification
- **paper_rigor_gap_analysis.md**: Tier-1/2/3 checklist; citation integrity audit
- **hf_publishing_plan.md**: Model card + dataset card templates; HF Hub upload procedure
- **CLAUDE.md**: PSALM operating guide; closure contract; six-layer Ralph-loop
- **ADR-0017**: H1_COGS null finding documentation
- **ADR-0020**: BabyLM dual track (competition vs. research)
- **ADR-0034/0035**: Vyutpattivāda faithful implementation
- **ADR-0038**: Leaderboard mechanisms specification
- **ADR-0039**: Morpheme-boundary N-hot embeddings + kāraka-aware masking design

---

**End of Master Plan**

Status: **LOCKED** (Gates enforced; all phases define dependencies and stop rules)  
Next action: Execute Phase 2 (parallel tracks) immediately; monitor Phase 1 completion
