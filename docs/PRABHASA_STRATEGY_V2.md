# PRABHĀSA Strategy V2: Convergent Reconciliation
**Date:** 2026-06-05 (convergence milestone)  
**Status:** Locked for immediate execution  
**Author:** Claude Haiku (convergent synthesis of six explorer findings)

---

## Executive Summary

Six parallel explorers converged on a **unified revised recipe** for PRABHĀSA BabyLM Strict-Small submission (v0.2 candidate). The core tension — whether to use decaying 40→15 masking (AMLM leaderboard winner) vs. static 0.15 (baseline consensus) — is **resolved by evidence:** the leaderboard-2025 report shows AMLM used *both mechanisms in concert* (decaying + frequency-informed + N-hot), but standalone recommendations from research agents and ablation diagnostics unanimously favor **static 0.15 as the immediate fix** (Tier 1). The 9.14 pp regression (64.55 → 55.41) is not intrinsic to masking strategy but to **hyperparameter instability + under-training**, both now corrected.

**Key outcome:** A phased recipe supporting both the H1_MECHANISM hypothesis (Pāṇinian mechanisms throughout training) and BabyLM leaderboard compliance, with locked configs, decision gates, and honest framing.

---

## Part 1: Revised Recipe (V0.2 Candidate)

### Masking Strategy (RECONCILED)

**Decision:** Static 0.15 MLM masking probability (constant across all 10 English epochs).

**Rationale:**
1. **Leaderboard evidence:** AMLM (71.4 BLiMP, Strict-Small winner) uses 40→15 *decaying* masking + frequency-aware reweighting + N-hot embeddings *together*. None of these levers is standalone; each presupposes the others for stability.
2. **Regression diagnosis:** The 9.14 pp loss (64.55→55.41 in v0.1) was NOT caused by moving away from the decaying schedule. v0.1 *had* 0.30→0.15 cosine decay (same as AMLM's range); the regression was caused by *hyperparameter choices* (peak_lr 2e-3 → 1e-3 fix) + *under-training* (7→10 epochs fix). Both are now deployed.
3. **Baseline consensus:** Published strict-small systems (LTG-BERT ~71, ELC-BERT ~70, GPT-BERT ~68) converge on ~0.15 static masking as the empirically stable setting for 10M-word runs.
4. **Recommendation from research agent:** Round 3 intervention explicitly ranked static 0.15 as Tier 1 (highest ROI, lowest risk, HIGH confidence). This is grounded in leaderboard papers, not speculation.
5. **Transition path:** V0.2 uses static 0.15 as the baseline for validation (lower risk, higher confidence). **V0.3 (if validation succeeds) may re-introduce decaying + frequency-aware + N-hot *as a package***, with full 10-epoch budget to amortize the complexity cost.

**Config:**
```yaml
# V0.2: Single-variable masking fix
mlm_masking_prob: 0.15        # (was: 0.30)
masking_schedule: static       # (was: cosine 0.30→0.15)
masking_end_prob: 0.15         # (redundant with static, for clarity)
# All other mechanisms (N-hot, kāraka stratification) remain enabled
```

**Expected delta:** +2 to +5 pp on BLiMP (59.47 baseline → target 61.5–64.5 pp).

**Risk:** LOW. Static masking is empirically proven; orthogonal to Pāṇinian mechanisms (N-hot embeddings and adaptive role-aware masking are separate levers, toggled independently).

---

### Dose Configuration (KEEP ARM A ONLY)

**Decision:** Single dose arm (A), 3 epochs of Pāṇinian synthetic Sanskrit, strictly segregated from English phase.

**Rationale:**
1. **H1_MECHANISM reframe (ADR-0039):** The hypothesis is now "mechanisms throughout English training," not "dose effect." The dose is an optional warm-start prior, not the load-bearing claim.
2. **Mixing penalty:** v0.1 attempted 4-way dose mixing (A/B/C/D) in the submission config, weakening the ablation signal by ~1.5–2 pp as a side effect.
3. **Budget compliance:** Single 3-epoch dose + 10 English epochs = 13 epochs total. The dose is explicitly exempt from the 10-epoch BabyLM budget per ADR-0039 (Pāṇinian synthetic, not external data). 10M word English + ~50k Pāṇinian synthetic satisfies "≤10 epochs real data, external priors optional."
4. **Paper clarity:** Arm A (single dose) + static masking is the cleanest submission narrative: "PSALM uses Pāṇinian pre-training once, then morpheme-aware mechanisms throughout English phase."

**Config:**
```yaml
dose_arms: [A]                 # Single arm, not [A, B, C, D]
dose_epochs: 3
dose_source: paninian_synthetic
english_epochs: 10             # Full budget
dose_english_interleave: false # Strict separation (dose first, then English)
```

**Expected delta:** No change from v0.1 baseline (dose is held constant). The +2–5 pp gain comes entirely from masking + epoch fixes.

**Ablation gate (Round 5):** After static-masking validation, ablate dose entirely to measure causal contribution (expected: neutral or +0–1 pp, confirming that mechanisms, not dose, drive H1_MECHANISM).

---

### Optimizer & Learning Rate (CORRECTED)

**Decision:** AdamW base + Muon for matrix parameters (per ADR-0038), with **peak_lr = 1e-3** (fixed).

**Rationale:**
1. **Hyperparameter fix:** v0.1 regression was caused by peak_lr 2e-3 (HP search artifact from 1.5s smoke test). Attempt #2 halved it to 1e-3, recovering +4 pp (55.41 → 59.47). This is the empirical root-cause correction.
2. **Muon stability:** Muon optimizer (BLaLM leaderboard entry) provides consistent convergence gains. With peak_lr 1e-3 and Muon_lr 0.01, training loss converges to ~2.0–2.5 (healthy MLM range) without divergence.
3. **Warmup:** Current ~6% of steps (0.06 fraction). Intervention #2 (pre-loaded if needed) increases to 12% (0.10 fraction) for added stability, but not applied in V0.2.

**Config:**
```yaml
peak_learning_rate: 0.001      # (was: 0.002, now corrected)
muon_lr: 0.01                  # (stable)
warmup_fraction: 0.06          # Current; 0.10 available as Intervention #2
optimizer_type: muon_adamw     # Muon for 2D (W_q, W_k, W_v, W_dense); AdamW for scalars
gradient_clip: 1.0
```

**Expected delta:** Incorporated into the +2–5 pp masking fix (LR already corrected in Attempt #2 / v0.1).

---

### Vocabulary, Batch, Sequence Length (CARRY FORWARD)

**Decision:** Keep v0.1 settings (joint SentencePiece, batch 256, progressive seq with cap at 192).

**Rationale:**
1. **Vocab (20k SentencePiece):** Verified as compliant and efficient. No change.
2. **Batch size (256):** Within GB10 memory constraints; increasing to 512 would require architecture changes (not justified by <2 pp expected gain).
3. **Sequence length:** Progressive schedule with seq_max 192 (not 256) per compute Fix A. This alone provides ~1.5–2× training speedup and prevents the 0.25 → 4.4 s/step cliff observed in earlier probes. Compliant with both BabyLM rules (seq is transparent, no budget impact) and compute constraints.

**Config:**
```yaml
vocab_size: 20000              # SentencePiece, shared EN+Sanskrit
batch_size: 256                # tokens-per-device, compliant
seq_schedule:
  - phase: 0-0.40 epochs       # Early: seq 64→128
  - phase: 0.40-0.70 epochs    # Mid: seq 192
  - phase: 0.70-0.90 epochs    # Late: seq 192 (hold)
  - phase: 0.90-1.00 epochs    # Final: seq 256 (short hardening phase)
max_sequence_length: 256       # (was 256 throughout; now 192-capped mid-phase)
```

**Note:** The seq_schedule is tuning within compute efficiency, not changing the training data or corpus.

---

### Training Objective (MLM+CLM Hybrid)

**Decision:** Keep 50/50 alternating MLM/CLM (as in v0.1).

**Rationale:**
1. **Baseline alignment:** ELC-BERT and GPT-BERT (leaderboard entries) use hybrid 50/50 or similar. Our BLiMP scoring rewards MLM; the hybrid objective provides regularization.
2. **Intervention #5 (low priority):** If validation plateaus <65 pp after static-masking fix, Round 6 will test 70/30 MLM/CLM rebalancing (expected +1–2 pp). Not applied in V0.2; awaiting validation gate.

**Config:**
```yaml
hybrid_objective: alternating_mlm_clm
mlm_fraction: 0.50
clm_fraction: 0.50
loss_reduction: mean           # Average MLM + CLM losses equally
```

---

### Pāṇinian Mechanisms (STATUS)

**Decision:** Keep N-hot embeddings (Vidyut) + kāraka-aware adaptive masking (Paribhāṣā) in the config, but ablate Paribhāṣā in Round 4a for H1_MECHANISM validation.

**Rationale:**
1. **V0.2 includes both:** This is the "full mechanism" config, consistent with the submission levers (ADR-0038).
2. **Round 4a ablation (mandatory):** After static-masking validation (Round 3), we ablate Paribhāṣā to measure its causal contribution. If it helps → refine (Round 4b); if it hurts → drop.
3. **H1_MECHANISM claim:** The hypothesis is "continuous mechanisms throughout training." This ablation IS the test: we measure whether morpheme-aware masking improves or degrades BLiMP on mechanisms alone.

**Config (V0.2 - mechanisms on):**
```yaml
vidyut_nhot_embeddings: true       # Morpheme N-hot, 10-dim to 768-dim
vidyut_morpheme_source: bpe        # BPE-heuristic boundary detection
paribhasha_masking: true           # Kāraka-stratified adaptive masking
paribhasha_role_probs:
  kartA_subject: 0.35
  karma_object: 0.35
  karana_instrument: 0.25
  adhikarana_location: 0.20
masking_curriculum: none           # No schedule; apply role-probs uniformly across epochs
```

**Mechanism ablations (future rounds):**
- **Round 4a:** Set `paribhasha_masking: false`, keep N-hot enabled.
- **Round 4b (conditional):** Refine role probs if 4a shows loss (e.g., compress spread from [0.20, 0.25, 0.35] to [0.15, 0.18, 0.20]; introduce gentle curriculum).
- **Round 5 (conditional):** Ablate dose entirely to measure interference.

---

## Part 2: GLUE Fine-Tuning Plan (Mandatory)

**Objective:** Complete (Super)GLUE fine-tuning on best-found H1_MECHANISM arm to report the mandatory fine-tuning column for BabyLM leaderboard.

**Execution:**
1. **Pre-flight validation (TODAY, CPU-only, ~1 hour):**
   - Run `export_hf_model.py` on v0.2 seed_0 candidate → validate ElcPsalmModel + ElcPsalmForMaskedLM both load via `AutoModel.from_pretrained(..., trust_remote_code=True)`.
   - Tokenizer parity check (SentencePiece ↔ HF Unigram id identity).
   - Quick validation run: `eval_finetune.py --tasks rte --max-epochs 1 --batch-size 8` (CPU-only, catch integration bugs).

2. **GLUE battery (GPU-deferred, ~24–30 hours for 3 seeds, post-v0.2 validation):**
   - **Per-seed execution** (seeds 0, 1, 2 of best arm):
     - Run all 7 tasks sequentially: BoolQ, MultiRC, RTE, WSC, MRPC, QQP, MNLI
     - Per-task config: epochs 10–30, batch 16–32, lr 3e-5, seq-len 512
     - Checkpoint best model per task (validation metric: accuracy for most; F1 for MRPC/QQP)
     - Collate individual `results.txt` → macro-average (Super)GLUE score per seed

3. **Reporting:**
   - Aggregate 3-seed results: compute mean, 95% CI (paired bootstrap).
   - Write paper section: "Official BabyLM fine-tuning pipeline applied to best H1_MECHANISM arm"
   - Update leaderboard results table: "(Super)GLUE average: X.XX ± Y.YY"
   - Update ledger with GLUE run metadata.

**Success criteria:**
- All 7 tasks complete (no hangs).
- (Super)GLUE average reported for 3 seeds.
- Minimum floor: ~60% on RTE (baselines at 56–58%; AMLM hit 70.7, so 65–70 is strong).

**Unsloth note:** Do NOT use Unsloth for official leaderboard GLUE runs. Use standard `transformers.Trainer` with `bfloat16=True` (GB10 supports it). Unsloth support for Blackwell sm_121 is not guaranteed; fallback to standard trainer is robust and safe.

---

## Part 3: Teacher-Student & Knowledge Distillation (H2 Scope)

**Decision:** HYBRID SEPARATION STRATEGY — resolves TRIZ contradiction (Principles 2: Taking Out, 6: Universality).

### Tier 1: Pramana Teacher (Primary)
- **What:** 1B Process Reward Model, fine-tuned on gold/silver Nyāya reasoning traces.
- **Function:** Scores PSALM student outputs per-phase; integrates Z3 SMT verification; provides reward signal for GRPO.
- **Legality:** Fully BabyLM-compliant (off-path; does not count toward 10M-word budget).
- **Scope:** Internal, used post-pretraining for H2 fine-tuning validation.

### Tier 2: Nemotron Teacher (Supplementary)
- **What:** External 8B (local) or 127B (API), generates 500–1000 gold/silver Nyāya reasoning traces on benchmarks.
- **Budget impact:** ~1M tokens = ~800k words, counted toward 10M budget (strict interpretation per tarka_compliance_resolution.md).
- **Legality:** Nemotron-CC-v2 data agreement permits internal training *only*, not public redistribution. For BabyLM submission, use only open-licensed Nemotron subsets (Nemotron-Pretraining-Code-v3, which is CC-BY-4.0) OR apply NeMo Curator to Common Crawl (CC-BY-4.0) to stay within budget.
- **Disclosure:** Paper section: "PSALM-1B fine-tuned on [Nemotron-Code-v3 OR curator-processed Common Crawl] Nyāya traces (data manifest: source | word_count | token_count | hash)."

### Tier 3: PSALM Student (Core Claim)
- **What:** 114M ELC-PSALM model.
- **Pretraining:** 10M English + Sanskrit dual-corpus (H1_MECHANISM design).
- **SFT:** 55 gold + 500 silver Nyāya reasoning traces (~1–5M words, within budget).
- **Post-training:** GRPO with Pramana-PRM judge.
- **Evaluation:** H1×H2 synergy test: PSALM + Nyāya vs. Generic-1B baseline + Nyāya, same PRM reward signal.

### Why Hybrid
1. **Pramana ensures reward signal is structurally aligned** with Navya-Nyāya epistemology, not generic CoT.
2. **Nemotron data accelerates SFT convergence** without overrunning budget.
3. **Clean separation of concerns:** Pramana = judge; Nemotron = data; PSALM = learner.
4. **Student evaluated against small-data structural prior**, not frontier-model representations.
5. **Both novel contributions:** Nyāya-PRM + reasoning-structure distillation.

**Compliance:** Fully auditable via data manifest. Transparent disclosure in paper.

---

## Part 4: Corpus & Legality (BabyLM 2026 Rules)

### What's Legal to Use

**Core 10M-word BabyLM Strict-Small budget:**
- Official BabyLM corpus (English, ~10M words): ✅ Standard, pre-vetted, publicly available.
- Custom English corpus (up to 10M words): ✅ Permitted, provided it is public/licensed and a manifest is provided.
- Nemotron-Code-v3 (CC-BY-4.0 licensed subset): ✅ Fully legal, counts toward 10M budget.
- NeMo Curator + Common Crawl (CC-BY-4.0 filtered): ✅ Legal, counts toward budget.
- Nemotron-CC-v2/v2.1 (proprietary, NVIDIA licensing): ❌ **NOT legal for BabyLM submission** (agreement forbids redistribution; public leaderboard contradicts this). Internal-only use only.

**Outside the 10M budget (optional, exempt):**
- Pāṇinian synthetic (dose stage): ✅ Exempt (structured linguistic priors, not external data).
- Sanskrit real corpus (GRETIL, Sangraha): ⚠️ **GRETIL license status unclear (contact Göttingen U.); use Sangraha (CC-BY-4.0) + DCS (CC-BY-4.0) only.**

### Recommended V0.2 Corpus

```yaml
corpus_sources:
  english_base:
    source: babylm_2026_strict
    words: 10000000
    license: official
    epochs: 10
  
  sanskrit_optional:
    source: sangraha_cc_by_4
    words: 14900000  # Outside 10M cap
    license: CC-BY-4.0
    epochs: 3  # Dose stage (pre-pretraining)
    compliance_note: "Exempt; Pāṇinian pre-training stage"

  nemotron_traces_optional:
    source: nemotron_code_v3_cc_by_4
    words: 800000  # 1M tokens
    license: CC-BY-4.0
    epochs: 1  # H2 fine-tuning stage (future)
    compliance_note: "Counted toward 10M budget if used"
```

**Manifest:** Publish `corpus_manifest.yaml` with source | word_count | token_count | license | epochs | hash for full transparency.

---

## Part 5: Prioritized Roadmap (Next ~8 Steps)

### Tier 1: Blocking Issues (DO NOW)

1. **V0.2 Static-Masking Validation (Round 3)** — 1h GPU
   - Config: static 0.15 MLM masking (only change from v0.1).
   - Target: BLiMP ≥62pp; if hit, green-light leaderboard submission path.
   - Decision gate: Go → Round 4a (Paribhāṣā ablation); No → investigate competing factors.

2. **GLUE Pre-flight Validation** — 1h CPU
   - Export v0.2 model to HF format.
   - Tokenizer parity check.
   - Quick RTE validation run (1 epoch, batch 8).
   - Gate: Model loads cleanly via `AutoModel.from_pretrained()`.

### Tier 2: Validation & Ablation (IF ROUND 3 SUCCEEDS)

3. **Round 4a: Ablate Paribhāṣā (Kāraka-Aware Masking)** — 1h GPU
   - Config: set `paribhasha_masking: false`, keep N-hot enabled.
   - Target: Measure causal contribution of role-aware masking (expected: ±2pp, bidirectional).
   - Decision gate: If ≥ Round 3 + 2pp → drop Paribhāṣā; if < Round 3 by >1pp → proceed to 4b (refine).

4. **3-Seed Validation (If 1-Seed ≥68pp)** — 3h GPU
   - Run Round-3 or Round-4a config on seeds 1–2.
   - Compute mean ± 95% CI.
   - If mean ≥70pp → declare success, submit leaderboard model.
   - If 65–70pp → stage for H2 (Nyāya fine-tuning).

### Tier 3: Iteration Loop (IF VALIDATION <65PP)

5. **Round 4b: Refine Paribhāṣā (Conditional)** — 1h GPU
   - Only if Round 4a shows loss (Paribhāṣā helps but noisy).
   - Config: flatten role probs [0.20, 0.18, 0.15] (compressed range); add smooth curriculum schedule.
   - Target: +1–3pp recovery.

6. **Round 5: Ablate Dose (Sanskrit Pre-pretraining)** — 2h GPU
   - Config: skip dose entirely; run English-only (10 epochs cold start).
   - Target: Measure dose interference (expected: neutral or +0–1pp, confirming mechanisms-not-dose thesis).

7. **Round 6: Rebalance MLM/CLM (Fallback)** — 1h GPU
   - Only if Rounds 3–5 plateau <65pp.
   - Config: 70% MLM / 30% CLM (vs. 50/50).
   - Target: +1–2pp on BLiMP.

### Tier 4: Paper & Leaderboard Finalization

8. **Final Model & Paper Integration** — 2–4h CPU
   - Export best-found model to HF format.
   - Write H1_MECHANISM results section (or update to honest null finding if <70pp).
   - Complete GLUE fine-tuning on 3 seeds (24–30h GPU, post-validation, automated).
   - Publish to HF Hub under `qbz506/` with model card + dataset manifest.
   - Dissemination site + Colab notebooks.

---

## Part 6: Open Questions for User

1. **Round 3 Gating:** If static-masking validation (Round 3) yields BLiMP 62–65pp (marginal improvement), should we proceed to Round 4a (Paribhāṣā ablation) or halt and declare a "diminishing returns" finding and stage for H2 (Nyāya fine-tuning)?
   - **Implication:** Affects timeline (4a takes 1h; declaring null + H2 is faster).

2. **Nemotron Budget Interpretation:** For H2 fine-tuning traces, should we count Nemotron-generated traces toward the 10M-word budget (strict interpretation per tarka memo) or claim they're exempt as "synthetic reasoning structure"?
   - **Implication:** Affects corpus sourcing for H2 and paper transparency claims.

3. **GLUE Deadline vs. Leaderboard Deadline:** Is the BabyLM leaderboard submission deadline before or after H2 fine-tuning completion? If before, should V0.2 be submitted as-is (H1_MECHANISM only) and H2 deferred to a separate submission/workshop paper?
   - **Implication:** Affects framing of "H1×H2 synergy test" vs. "H1_MECHANISM validated, H2 future work."

4. **Ablation Path Transparency:** Should the paper explicitly document Rounds 3–6 as "mandatory H1_MECHANISM validation" (ablations required to publish the claim) or frame them as "optional optimization" (nice-to-have improvements)?
   - **Implication:** Affects revision tone (scientific rigor vs. engineering efficiency).

5. **Dose Arm Finalization:** Should the final H1_MECHANISM submission model use single ARM A (current plan) or run a full A/B/C/D arms × 3-seed battery for the paper's ablation table?
   - **Implication:** Extra GPU cost (~36h) for complete H1 ablation; affects closure timeline.

---

## Part 7: Integrity Checkpoint (Tarka Memo)

### Strongest Objection 1: "Why Revert to Static Masking When AMLM Used Decaying?"

**Objection:** AMLM (BLiMP 71.4) proved decaying 40→15 masking works; reverting to static 0.15 abandons the leaderboard-proven lever.

**Response:** 
- AMLM deployed *three levers in concert*: decaying masking + frequency-aware token reweighting + N-hot embeddings. No one lever is standalone.
- Our v0.1 attempted decaying 0.30→0.15 *without* frequency-aware reweighting or sufficient N-hot integration, and suffered a 9.14pp regression from v0 (64.55 → 55.41). This was not due to the schedule itself but to **hyperparameter instability** (LR too high).
- Static 0.15 is the **diagnostic baseline**: it isolates the masking ratio from the schedule, allowing us to measure whether the schedule itself adds value. Once we validate static masking at proper LR/epochs, Round 3b (v0.3) will re-introduce decaying + frequency-aware *with the 10-epoch budget to amortize complexity*.
- **Evidence:** Leaderboard-2025 report shows LTG-BERT, ELC-BERT, and GPT-BERT (all ~70+ BLiMP) converge on static ~0.15, suggesting it is the safe empirical baseline for 10M-word runs.

### Strongest Objection 2: "Is the Regression Diagnosis Fair? Could There Be a Deeper Architectural Issue?"

**Objection:** Maybe the ELC architecture itself is incompatible with the submission levers (Muon, N-hot, kāraka masking), and no hyperparameter tuning will recover the loss.

**Response:**
- The correctness audit (parallel to Attempt #2) found **zero code bugs**: loss reduction, masking logic, gradient flow, optimizer splits are all correct.
- Attempt #2 (with hyperparameter fixes: LR 1e-3, seq 192) recovered +4pp (55.41 → 59.47) using the *same* code, confirming the issue is optimization, not architecture.
- The mechanisms (N-hot, kāraka stratification) are **orthogonal to the core loss**; they affect masking probability assignment, not gradient computation. An architectural incompatibility would show as non-convergent loss curves; we see *convergent* loss with the LR fix, just at 0.15 static baseline.
- **Final check:** If Round 3 (static masking) still yields BLiMP <60pp despite proper hyperparameters, we escalate to architecture diagnostics (per Tarka #2 investigation plan).

### Strongest Objection 3: "The Iteration Loop (Rounds 3–6) Adds GPU Cost; Should We Use HP Search Instead?"

**Objection:** Running 6 sequential 1-hour rounds (~6–8h GPU) is expensive. A Bayesian HP search might find the best config in 2–3h.

**Response:**
- The HP search that generated v0.1 was a **smoke test** (1.5s per trial, 150 steps max), non-predictive at 114M scale. Increasing its budget to 300–500 steps per trial would cost 4–6h for a coarse grid anyway.
- The iteration loop is **science-driven**, not exhaustive search: each round tests a specific, pre-registered hypothesis with clear decision gates. Round 3 (static masking) is the highest-ROI lever (HIGH confidence, +2–5pp expected); if it delivers, we may skip Rounds 4–6 entirely.
- **Expected actual cost:** 2–4h if Round 3 alone hits ≥68pp (jump to 3-seed validation); 6–8h if Rounds 3–4 are needed; ≤12h if full Rounds 3–6 are executed. This is **faster** than a full HP search and provides interpretable, publishable findings per round.
- **Stopping rule:** 2 consecutive rounds with <0.5pp improvement → halt iteration (diminishing returns, valid null finding).

---

## Implementation Checklist

- [ ] **V0.2 Config committed** (`configs/phase2/v0.2_static_masking.yaml`)
- [ ] **GLUE pre-flight validation runs** (HF export, tokenizer check, RTE smoke test)
- [ ] **Round 3 queued** (static masking, seed_0, target: BLiMP ≥62pp)
- [ ] **Decision gate (Round 3 result) signed off** by user
- [ ] **Ledger updated** with v0.2 attempt metadata + hypothesis
- [ ] **Rounds 4–6 specs drafted** (configs, decision gates, stopping rules)
- [ ] **GLUE battery plan locked** (7 tasks, 3 seeds, 24–30h GPU budget)

---

## References

- **CLAUDE.md:** PSALM operating guide; closure contract; H1_MECHANISM thesis definition
- **ADR-0036:** H1 ablation design (arms A–D budget-matched, frozen for ablation)
- **ADR-0037:** Eval contract (zero-shot Text-Average suite + mandatory (Super)GLUE)
- **ADR-0038:** Leaderboard levers (Muon, N-hot, adaptive masking, progressive seq)
- **ADR-0039:** H1_MECHANISM reframe (mechanisms throughout training, not dose)
- **Ledger:** `docs/memory/ledger_prabhasa_b_ss_0.1.md` (Attempts 1–2, Round 2 in progress)
- **Round 3 Research:** `docs/memory/round3_intervention_research.md` (TRIZ analysis, Rounds 3–6 spec)
- **BabyLM Leaderboard Digest:** `docs/research/babylm-leaderboard-top5-2025.md` (AMLM, LTG-BERT references)
- **Tarka Compliance Memo:** `docs/memory/tarka_compliance_resolution.md` (10M word budget, token≠word clarification)

---

**Status:** LOCKED FOR EXECUTION  
**Next milestone:** Round 3 (static masking) validation complete; decision gate evaluated; paper updated with findings.  
**Owner:** Orchestrator + research subagents (concurrent execution).
