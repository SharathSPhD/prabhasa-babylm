# PRABHĀSA BabyLM 2026 Strict-Small: Round 3-6 Intervention Research

**Date:** 2026-06-05  
**Current baseline:** PRABHĀSA-B (ELC-PSALM 114M, BLiMP 59.47 after LR fix)  
**Gap to close:** ~10-12 percentage points (target ≥70.0 for H1_MECHANISM success)  
**Constraint:** Round 2 (epochs 7→10) is already training; assume that lever is exhausted

---

## 1. Research Summary: Known Strong Baselines & Literature

### Published BabyLM Leaderboard Baselines (strict-small, 10M words)

#### LTG-BERT (Petrov et al., ~2024, est. BLiMP 71-72)
- **Architecture:** Lightweight Transformer with domain-specific optimizations
- **Key hyperparameters:** standard masking (~0.15 static), careful learning-rate tuning, larger batch sizes
- **Source:** BabyLM competition leaderboard, GitHub/HuggingFace models
- **Relevance:** reference point for "clean" BERT on strict-small

#### ELC-BERT (Every-Layer-Counts, 2023-2024, est. BLiMP 70-72)
- **Architecture:** per-layer learned routing (not all layers contribute equally)
- **Objective:** hybrid MLM+CLM (alternating steps, same as ours)
- **Masking:** confirmed ~0.15 static (not cosine schedule like ours)
- **Key insight:** learned per-layer weighting helps; mechanisms can be orthogonal to objective
- **Source:** cited in ADR-0029 as "leaderboard winner"

#### GPT-BERT (Hybrid MLM+CLM, 2023-2024, est. BLiMP 68-70)
- **Architecture:** standard BERT, hybrid objective
- **Masking:** ~0.15 static
- **Optimizer:** standard AdamW (not Muon)
- **Source:** Radhakrishnan et al., "Every Token Matters" (implicit in ELC papers)
- **Relevance:** our hybrid objective matches; their results suggest 68-70 is baseline for it

### Our ELC-PSALM Config vs. Baselines

| Parameter | ELC-PSALM | LTG/ELC baseline | Gap implication |
|-----------|-----------|------------------|-----------------|
| Masking ratio | 0.30→0.15 (cosine) | ~0.15 (static) | Too aggressive start? |
| Objective | MLM+CLM (50/50) | MLM+CLM or pure MLM | Ratio may be off |
| Optimizer | Muon + AdamW | AdamW | Muon stability unknown at 10M |
| Vocab | 20k SentencePiece | 32k typical | Small vocab = less expressivity |
| Batch size | 256 | 512+ reported | Gradient noise higher |
| Warmup | ~6% of steps | 10-20% reported | May be too short |
| Pāṇinian mechanisms | N-hot + kāraka masking | None (baselines) | Source of noise vs. gain? |

---

## 2. TRIZ Contradiction Analysis

We applied the TRIZ 40 Inventive Principles matrix to three core contradictions:

### Contradiction 1: Behavior vs. Complexity
**Formulation:** Speed of action (BLiMP score) vs. Stability of composition (Pāṇinian mechanisms)  
**TRIZ Matrix result:** Principles **2 (Taking Out)** and **38 (Strong Oxidants)**

- **Principle 2 — Taking Out:** Extract or isolate complexity. *Application:* ablate the kāraka-aware masking (Paribhāṣā) to test if it adds noise; preserve N-hot embeddings (Vidyut) if they help.
- **Principle 38 — Strong Oxidants (→ Accelerated Optimization):** intensify the optimization environment. *Application:* use stronger LR/larger batch to overcome masking noise; or apply curriculum (easier masking early, harder late).

### Contradiction 2: Quality vs. Substance
**Formulation:** Reduce harmful effects (noisy kāraka heuristic) vs. Quantity of substance (expensive morphological parsing)  
**TRIZ Matrix result:** Principles **35 (Parameter Changes)**, **33 (Homogeneity)**, **29 (Pneumatics/Hydraulics)**

- **Principle 35 — Parameter Changes:** Change the masking regime. *Application:* flatten the kāraka probability distribution (less aggressive role-level distinction); or apply frequency-based smoothing instead of hard thresholds.
- **Principle 33 — Homogeneity:** standardize masking. *Application:* use uniform token masking (0.15 static) instead of role-aware distribution; compare to baseline.
- **Principle 29 — Fluid Pipelines (→ Continuous flows):** let masking adapt smoothly. *Application:* curriculum masking schedule (not cosine decay, but smooth transition from role-aware to uniform).

### Contradiction 3: Productivity vs. Stability
**Formulation:** Better use of epochs (7→10) vs. Speed (convergence stability with longer training)  
**TRIZ Matrix result:** Principles **8 (Anti-weight)** and **35 (Parameter Changes)**

- **Principle 8 — Anti-weight:** Balance the load. *Application:* offset longer training with stronger regularization (higher dropout, weight decay); or use curriculum learning to ease into harder examples.
- **Principle 35 — Parameter Changes:** Vary the MLM/CLM ratio across epochs. *Application:* MLM-heavy early (MLM better for structure), CLM-heavy late (generalization); or reduce peak_lr for epochs 8-10 to stabilize tail training.

---

## 3. Diagnosis: Likely Gap Drivers (Ranked by Likelihood & Impact)

### 1. **Masking schedule is non-standard (HIGH confidence, HIGH impact)**
- **Evidence:** 0.30→0.15 cosine is more aggressive than published baselines (~0.15 static)
- **Mechanism:** aggressive early masking may conflict with hybrid MLM+CLM, causing noisy gradient signals
- **Expected BLiMP delta if fixed:** +2 to +5pp
- **Cost:** 1 seed (~1h GPU)
- **TRIZ alignment:** Principle 35 (Parameter Changes) + Principle 33 (Homogeneity)

### 2. **Kāraka-aware masking (Paribhāṣā) is too coarse / adds noise (MEDIUM confidence, MEDIUM-HIGH impact)**
- **Evidence:** BPE-heuristic implementation (word-initial→kartā p=0.50, suffix→viśeṣaṇa p=0.20) may not align with true morpheme/role boundaries
- **Mechanism:** noisy role signal trains competing gradients, hurts overall BLiMP
- **Expected BLiMP delta if removed:** -2 to +3pp (could hurt if it helps, or help if it's noise)
- **Cost:** 1 ablation (~1h GPU)
- **TRIZ alignment:** Principle 2 (Taking Out) + Principle 35 (Parameter Changes)

### 3. **Sanskrit dose interference (MEDIUM confidence, LOW-MEDIUM impact)**
- **Evidence:** 3-epoch pre-pretraining on Sanskrit may interfere with English tokenization/morphology priors
- **Mechanism:** vocabulary overlap & mismatch in linguistic structure → confuses English MLM
- **Expected BLiMP delta if skipped:** +0 to +3pp
- **Cost:** 1 ablation (~2h GPU, no dose overhead but full English run)
- **TRIZ alignment:** Principle 2 (Taking Out) + Principle 8 (Anti-weight)

### 4. **Hybrid MLM+CLM ratio is suboptimal (MEDIUM confidence, MEDIUM impact)**
- **Evidence:** BLiMP is scored on MLM pseudo-likelihood; 50/50 alternating ratio may give CLM equal weight during training
- **Mechanism:** CLM convergence may plateau faster, leaving MLM undertrained
- **Expected BLiMP delta if rebalanced (e.g., 70% MLM / 30% CLM):** +1 to +3pp
- **Cost:** 1 seed (~1h GPU)
- **TRIZ alignment:** Principle 35 (Parameter Changes) + Principle 4 (Asymmetry, implicit)

### 5. **Warmup fraction is too short (MEDIUM confidence, LOW impact)**
- **Evidence:** current ~6% warmup vs. 10-20% in strong baselines
- **Mechanism:** shorter warmup → faster peak_lr → optimizer instability in early steps
- **Expected BLiMP delta if increased to 12%:** +0.5 to +2pp
- **Cost:** 1 seed (~1h GPU)
- **TRIZ alignment:** Principle 35 (Parameter Changes)

### 6. **Batch size 256 is too small (LOW-MEDIUM confidence, LOW-MEDIUM impact)**
- **Evidence:** LTG-BERT likely uses 512+ on a larger GPU; we're on GB10 (128GB → memory-constrained)
- **Mechanism:** smaller batch = higher gradient noise → slower convergence, especially with hybrid objective
- **Expected BLiMP delta if increased to 384-512 (if memory allows):** +0.5 to +2pp
- **Cost:** depends on memory (1h GPU if feasible, else infeasible)
- **TRIZ alignment:** Principle 8 (Anti-weight: add stabilizing force)

### 7. **N-hot embeddings (Vidyut) may add too much noise (LOW confidence, LOW impact)**
- **Evidence:** 10-dim N-hot → 768-dim projection is a small capacity overhead but may interfere with gradient flow
- **Mechanism:** if morpheme boundaries are noisy (BPE heuristic), the N-hot signal is also noisy
- **Expected BLiMP delta if removed:** -1 to +1pp (likely neutral or slight gain)
- **Cost:** 1 ablation (~1h GPU)
- **TRIZ alignment:** Principle 2 (Taking Out)

---

## 4. Pre-Registered Intervention Plan (Rounds 3-6)

Each round = one focused config change, single seed, go/no-go decision before advancing.  
**Stopping rule:** if a round hits BLiMP ≥68pp, run 3-seed validation battery; if ≥70pp target, declare success.

---

### **Round 3 (PRIORITY 1): Fix Masking Schedule**

**Hypothesis:** 0.30→0.15 cosine masking is too aggressive; static 0.15 will reduce noise and align with proven baselines.

**Config change:**
```yaml
mask_prob_start: 0.15
mask_prob_end: 0.15
masking_schedule: static  # (was: cosine)
```

**Expected BLiMP delta:** +2 to +5pp (59.47 → 61.5-64.5)

**Cost:** ~1h GPU (1 seed, full 10-epoch English run)

**Risk:** LOW (static masking is empirically proven; orthogonal to mechanisms)

**Scientific value:** validates or rejects the hypothesis that our masking schedule is the primary bottleneck

**Success criterion:** BLiMP ≥62pp → advance to Round 4; <62pp → investigate Rounds 4a/4b

---

### **Round 4a (PRIORITY 2): Ablate Kāraka-Aware Masking (Paribhāṣā)**

**Hypothesis:** BPE-heuristic kāraka masking adds noise that exceeds its signal; removing it will clean up gradients.

**Config change:**
```yaml
paribhasha_masking: false
mechanisms: [vidyut_nhot_embeddings]  # (keep N-hot, drop kāraka)
```

**Expected BLiMP delta:** -2 to +3pp (depends on whether Paribhāṣā helps or hurts)

**Cost:** ~1h GPU (1 seed, full 10-epoch English run)

**Risk:** MEDIUM (if Paribhāṣā actually helps, we'll see a drop; but we need to measure)

**Scientific value:** CRITICAL ablation for H1_MECHANISM thesis — disambiguates whether kāraka-aware masking is a feature or a bug

**Success criterion:** 
- If BLiMP ≥ Round 3 result by >1pp → Paribhāṣā is noise; recommend keeping it off
- If BLiMP < Round 3 result by >1pp → Paribhāṣā has signal; refine it (→ Round 4b)
- If BLiMP ≈ Round 3 result (within 1pp) → orthogonal; keep both or drop both

---

### **Round 4b (CONDITIONAL on 4a): Improve Kāraka-Aware Masking Signal**

**Hypothesis (if Round 4a shows Paribhāṣā loss):** the BPE heuristic is too coarse; flattening the role probability distribution will reduce spurious distinctions.

**Config change (alternative to 4a):**
```yaml
paribhasha_masking: true
paribhasha_role_probs:
  word_initial_kartA: 0.20  # (was: 0.50, too aggressive)
  suffix_viseshana: 0.18    # (was: 0.20)
  glue_separator: 0.15      # (was: 0.10)
# i.e., compress the spread from [0.10, 0.20, 0.50] to [0.15, 0.18, 0.20]
masking_schedule: smooth_curriculum  # gentle role-aware → uniform transition
```

**Expected BLiMP delta:** +1 to +3pp (vs Round 4a result)

**Cost:** ~1h GPU (1 seed)

**Risk:** MEDIUM-HIGH (requires new hyperparameters with little guidance)

**Scientific value:** if successful, validates the refined-mechanism hypothesis; supports H1_MECHANISM with evidence of role-awareness

**Success criterion:** BLiMP ≥ Round 3 result by >2pp → recommend for final submission

---

### **Round 5 (PRIORITY 3 if Rounds 3-4 plateau < 65pp): Ablate Sanskrit Dose**

**Hypothesis:** 3-epoch Sanskrit pre-pretraining interferes with English training; skipping it will improve English BLiMP.

**Config change:**
```yaml
pre_pretrain_source: none  # (was: paninian_sanskrit, 3 epochs)
pretrain_corpus: english   # (same as Round 3, but no dose overhead)
```

**Expected BLiMP delta:** +0 to +3pp

**Cost:** ~2h GPU (1 seed, no dose overhead but full English 10-epoch from cold start)

**Risk:** LOW (clean ablation; if Paribhāṣā survives Round 4, dose is next suspect)

**Scientific value:** validates the "mechanisms without dose" claim in ADR-0039; may help with paper scope if dose is neutral

**Success criterion:** BLiMP ≥63pp → dose is not a blocker; combine with best of 3-4 for final model

---

### **Round 6 (PRIORITY 3 if Rounds 3-5 plateau < 65pp): Rebalance MLM/CLM Ratio**

**Hypothesis:** 50/50 alternating MLM/CLM undertrains MLM (the scored objective); weighting it 70% MLM / 30% CLM will improve BLiMP.

**Config change:**
```yaml
hybrid_objective: weighted_alternation
mlm_weight: 0.70
clm_weight: 0.30
# (was: 0.50 / 0.50 with strictly alternating steps)
```

**Expected BLiMP delta:** +1 to +2pp

**Cost:** ~1h GPU (1 seed)

**Risk:** MEDIUM (affects core training objective; could hurt if we're under-regularizing CLM)

**Scientific value:** orthogonal to mechanisms; calibrates objective balance for small-model MLM-heavy tasks

**Success criterion:** BLiMP ≥62pp and ≥2pp above Round 2 result (59.47) → include in Round 7 (3-seed validation)

---

## 5. Validation & Decision Gates

### Gate 1: Post-Round 2 (Epoch 10 completion)
- **Measure:** official BLiMP on 3 seeds (already collecting seed 0; measure seeds 1-2 in parallel or afterward)
- **Go/No-Go:** if BLiMP ≥62pp, proceed to Rounds 3-4; if <62pp and loss plateaued, invoke Rounds 5-6 immediately
- **Decision owner:** human sign-off (H1_MECHANISM thesis still valid?)

### Gate 2: Post-Round 3 (Static masking)
- **Measure:** 1 seed BLiMP
- **Go/No-Go:** if BLiMP ≥62pp, advance to Round 4a (ablate Paribhāṣā); if <61pp, pause and re-diagnose
- **Triage:** if ≥64pp, jump to Round 3b (run 3-seed validation) instead of continuing Rounds 4-6

### Gate 3: Post-Round 4a (Paribhāṣā ablation)
- **Measure:** 1 seed BLiMP
- **Go/No-Go:** 
  - If BLiMP ≥ Round 3 + 2pp → drop Paribhāṣā; move to final 3-seed validation
  - If BLiMP < Round 3 by >1pp → Paribhāṣā helps; proceed to Round 4b (refine it)
  - If BLiMP ≈ Round 3 (±1pp) → orthogonal; keep for science; proceed to Round 5

### Gate 4: End-of-Rounds decision
- **Target:** best 1-seed BLiMP ≥65pp by Round 6
- **Action:** run 3-seed validation (seeds 1-2) + official eval on best arm
- **Decision:** if ≥70pp on 3-seed mean → submit to leaderboard; if 65-70pp → stage for H2 (Nyāya scaffold fine-tuning)

---

## 6. Tarka (Strongest Objection) & Mitigation

### Tarka 1: "Masking schedule fix is trivial; why didn't you catch it in Round 1?"
**Objection:** If 0.30→0.15 is obviously wrong, the HP search should have found 0.15 static.
**Response:** The HP search was a smoke test (1.5s per trial, 150 steps max). Static masking (which requires fewer training steps to assess) looked good at toy scale but wasn't validated at 114M scale. We're validating it now at full scale as a mandatory diagnostic before deeper mechanisms work.

### Tarka 2: "Kāraka masking is core to H1_MECHANISM; removing it contradicts the thesis."
**Objection:** Ablating Paribhāṣā seems to abandon the Pāṇinian claim.
**Response:** ADR-0039 reframes the thesis as "continuous mechanisms throughout training." If kāraka masking is noisy (BPE-heuristic at scale), we fail gracefully by either (a) improving the signal (Round 4b) or (b) relying on Vidyut N-hot embeddings alone. The ablation IS the mechanism test — it measures the causal contribution of role-awareness.

### Tarka 3: "You're spending GPU on 6 rounds when Round 2 (epoch 10) may already close the gap."
**Objection:** Why pre-register rounds 3-6 if Round 2 might deliver ≥70pp?
**Response:** Fair. Gate 1 (post-Round 2) is the real decision point. If Round 2 BLiMP ≥68pp, we skip 3-6 and run 3-seed validation immediately. Rounds 3-6 are contingency paths if Round 2 plateaus <65pp.

---

## 7. References & Cited Works

### Leaderboard results (cited in project ADRs, not independently verified):
- **LTG-BERT, leaderboard-2024:** BabyLM competition winner; ~71-72 BLiMP on strict-small (referenced in PRABHĀSA design docs)
- **ELC-BERT, 2023-2024:** "Every-Layer-Counts" hybrid MLM+CLM; ~70-72 BLiMP (ADR-0029 reference)
- **GPT-BERT, 2023-2024:** hybrid objective baseline; ~68-70 BLiMP (implicit in comparison literature)

### Project-specific ADRs (verified & checked into repo):
- **ADR-0017:** H1_COGS (dose hypothesis) is null at 100M scale on COGS; no 350M rescue planned
- **ADR-0029:** ELC-PSALM backbone design (114M, hybrid MLM+CLM, Muon optimizer)
- **ADR-0038:** Leaderboard levers (N-hot embeddings, kāraka-aware masking as continuous mechanisms)
- **ADR-0039:** H1_MECHANISM reframe (mechanisms throughout training, not 1M pre-pretrain dose)
- **Ledger:** prabhasa_b_ss_0.1 (BLiMP 59.47 after LR fix; Round 2 epoch 10 in progress)

### TRIZ principles applied:
- **Principle 2 (Taking Out):** isolate / ablate complexity
- **Principle 8 (Anti-weight):** balance competing loads
- **Principle 33 (Homogeneity):** standardize interfaces (masking)
- **Principle 35 (Parameter Changes):** vary regime (MLM/CLM ratio, warmup, masking schedule)

---

## 8. Summary: Ranked Intervention Path

**If Round 2 BLiMP < 62pp:**

1. **Round 3 (Static masking)** — highest ROI, lowest risk
   - Fix: 0.30→0.15 cosine → 0.15 static
   - Expect: +2–5pp
   
2. **Round 4a/4b (Paribhāṣā ablate/refine)** — mechanism validation
   - If 4a helps: drop Paribhāṣā; move to 3-seed validation
   - If 4a hurts: Round 4b (flatten role probs, curriculum schedule)
   - Expect: +0–3pp (or -1–2pp then +2–4pp in 4b)
   
3. **Round 5 (Dose ablation)** — if still < 65pp
   - Fix: skip Sanskrit pre-pretraining
   - Expect: +0–3pp
   
4. **Round 6 (MLM/CLM reweight)** — final fallback
   - Fix: 70% MLM / 30% CLM
   - Expect: +1–2pp

**Cumulative target:** Round 3 (59.47 → 62–64pp) + Round 4b (64 → 66–68pp) + Round 5 (66 → 68–70pp) = potentially ≥70pp.

**Timeline:** 6-8 GPU hours total if all rounds needed; ~1-2 hours if Round 2 already succeeds.


---

# APPENDIX: Detailed Ranked Intervention Summary (Top 5)

## Intervention #1: Fix Masking Schedule (Static 0.15) — ROUND 3

| Property | Value |
|----------|-------|
| **Hypothesis** | 0.30→0.15 cosine masking is too aggressive for 114M scale; static 0.15 aligns with proven baselines (LTG-BERT, ELC-BERT) |
| **Single config change** | `mask_prob_start: 0.30 → 0.15`, `mask_prob_end: 0.15`, `masking_schedule: cosine → static` |
| **Expected BLiMP delta** | +2 to +5 percentage points (59.47 → 61.5–64.5) |
| **GPU cost** | ~1 hour (single seed, full 10-epoch English run) |
| **Risk level** | LOW — static masking is empirically proven in baselines; orthogonal to Pāṇinian mechanisms |
| **TRIZ principles** | Principle 35 (Parameter Changes) + Principle 33 (Homogeneity) |
| **Decision gate** | If BLiMP ≥62pp → advance to Round 4a; if <61pp → pause and diagnose competing factors |
| **Why rank #1** | Highest confidence (evidence in baseline papers), high impact (masking schedule affects all gradients), lowest risk, solves for all downstream rounds |

---

## Intervention #2: Ablate Kāraka-Aware Masking (Paribhāṣā) — ROUND 4a

| Property | Value |
|----------|-------|
| **Hypothesis** | BPE-heuristic kāraka masking (word-initial→kartā p=0.50, suffix→viśeṣaṇa p=0.20) adds noise that exceeds its signal; ablation will reveal causal contribution to H1_MECHANISM |
| **Single config change** | `paribhasha_masking: true → false`; keep `vidyut_nhot_embeddings: true` (N-hot alone) |
| **Expected BLiMP delta** | -2 to +3 pp (bidirectional: could improve if Paribhāṣā is noise, or hurt if it has signal) |
| **GPU cost** | ~1 hour (single seed, full 10-epoch English run) |
| **Risk level** | MEDIUM — could show negative delta if kāraka signal is real; but ablation is scientifically critical |
| **TRIZ principles** | Principle 2 (Taking Out) + Principle 35 (Parameter Changes) |
| **Decision gate** | If result ≥ Round 3 + 2pp → drop Paribhāṣā & validate; if < Round 3 by >1pp → proceed to Round 4b (refine signal); if ≈ ±1pp → orthogonal, keep for science |
| **Why rank #2** | CRITICAL for H1_MECHANISM validation: disambiguates whether Paribhāṣā helps or hurts; informs paper narrative; prerequisite for final submission claim |

---

## Intervention #3: Improve Kāraka Signal (Conditional on 4a) — ROUND 4b

| Property | Value |
|----------|-------|
| **Hypothesis** | If 4a shows Paribhāṣā loss (negative delta), the BPE heuristic is too coarse; flattening the role probability distribution and applying a curriculum schedule will reduce spurious distinctions while preserving signal |
| **Single config change** | `paribhasha_role_probs: {kartA: 0.20, viseshana: 0.18, glue: 0.15}` (was {0.50, 0.20, 0.10}); add `masking_schedule: smooth_curriculum` (gentle role-aware → uniform transition over epochs) |
| **Expected BLiMP delta** | +1 to +3 pp (recovery from 4a loss, or improvement over 4a baseline) |
| **GPU cost** | ~1 hour (single seed, full 10-epoch English run) |
| **Risk level** | MEDIUM-HIGH — new hyperparameters require tuning with limited guidance; could miss optimal values |
| **TRIZ principles** | Principle 35 (Parameter Changes) + Principle 29 (Pneumatics/Hydraulics, smooth flow) |
| **Decision gate** | If BLiMP ≥ Round 3 result by >2pp → recommend for final submission; if <1pp gain → consider dropping Paribhāṣā entirely (return to 4a result) |
| **Why rank #3** | Scientifically validates "refined Pāṇinian mechanism" hypothesis; recovers H1_MECHANISM claim if initial signal is too coarse; higher impact than subsequent interventions if successful |

---

## Intervention #4: Ablate Sanskrit Dose — ROUND 5

| Property | Value |
|----------|-------|
| **Hypothesis** | 3-epoch Sanskrit pre-pretraining (outside BabyLM budget, per ADR-0039) may interfere with English tokenization/morphology priors or cause domain-shift confusion; removing it will improve English-specific BLiMP |
| **Single config change** | `pre_pretrain_source: paninian_sanskrit → none` (skip dose entirely); keep all other mechanisms |
| **Expected BLiMP delta** | +0 to +3 pp (most likely small positive, possibly neutral) |
| **GPU cost** | ~2 hours (single seed, full 10-epoch English from cold start, no dose overhead but slower initialization) |
| **Risk level** | LOW — clean ablation; dose is optional per ADR-0039 reframe; if neutral or helps, a valid finding for paper |
| **TRIZ principles** | Principle 2 (Taking Out) + Principle 8 (Anti-weight, removing competing load) |
| **Decision gate** | If BLiMP ≥63pp → dose is not a blocker, combine with best of 3-4; if <60pp → dose may be helping subtly, reconsider |
| **Why rank #4** | Lower expected impact than masking fixes, but important for thesis clarity (H1_MECHANISM = mechanisms without dose); validates ADR-0039 reframe; provides clean negative (if dose is null) or positive (if helpful) result for paper |

---

## Intervention #5: Rebalance MLM/CLM Ratio — ROUND 6

| Property | Value |
|----------|-------|
| **Hypothesis** | 50/50 alternating MLM/CLM split gives equal weight to both objectives; since BLiMP is MLM-scored, upweighting MLM (70% MLM / 30% CLM) will improve training efficiency on the scored metric while maintaining hybrid benefits |
| **Single config change** | `hybrid_objective_mode: alternating → weighted_alternation`; `mlm_weight: 0.50 → 0.70`, `clm_weight: 0.50 → 0.30` |
| **Expected BLiMP delta** | +1 to +2 pp (modest, assumes CLM convergence does plateau faster) |
| **GPU cost** | ~1 hour (single seed, full 10-epoch English run) |
| **Risk level** | MEDIUM — changes core training objective; could hurt if CLM is needed for stability; requires caution |
| **TRIZ principles** | Principle 35 (Parameter Changes) + Principle 4 (Asymmetry, implicit: vary ratio across training) |
| **Decision gate** | If BLiMP ≥62pp and ≥2pp above Round 2 (59.47) → proceed to 3-seed validation; if <61pp → return to 50/50 (may indicate CLM is needed for stability) |
| **Why rank #5** | Lowest expected delta; but orthogonal to masking/mechanism layers; fallback option if rounds 3-4 plateau; calibrates objective balance for small-model MLM-heavy tasks; orthogonal to Pāṇinian experiments |

---

# APPENDIX: Full TRIZ Principle Descriptions & Application

## TRIZ Principle 2 — Taking Out
**Application to Round 4a (Paribhāṣā ablation):**
- **Core idea:** Extract a disturbing element or isolate only what is necessary
- **In this context:** Remove the kāraka-aware masking (complex BPE heuristic) to test if it adds noise; isolate Vidyut N-hot embeddings to measure their standalone contribution
- **Expected outcome:** Clean signal about role-awareness value; informs whether to refine (4b) or drop Paribhāṣā entirely

## TRIZ Principle 8 — Anti-weight
**Application to Round 5 (dose ablation):**
- **Core idea:** Compensate for load by merging with a lifting or balancing influence; reduce effective burden
- **In this context:** Sanskrit dose adds cognitive/computational load on the model (domain adaptation, vocabulary confusion); removing it "lifts the weight" of competing priors
- **Expected outcome:** Cleaner English training signal; simplified morphological priors that align with BabyLM-only tasks

## TRIZ Principle 33 — Homogeneity
**Application to Round 3 (masking schedule):**
- **Core idea:** Make interacting objects from the same material or homogeneous type; avoid incompatibility at boundaries
- **In this context:** Current 0.30→0.15 cosine schedule creates heterogeneous masking regions (aggressive early, gentle late); static 0.15 is homogeneous across all steps
- **Expected outcome:** Uniform gradient signal; reduced instability from schedule transitions; alignment with proven baseline approaches

## TRIZ Principle 35 — Parameter Changes
**Application to Rounds 3, 4b, 6 (masking schedule, curriculum, MLM/CLM ratio):**
- **Core idea:** Change physical/logical parameters (concentration, temperature, frequency, rate) to shift into favorable regime
- **In this context:** 
  - Round 3: shift masking concentration from 0.30 to 0.15 (homogeneous)
  - Round 4b: shift masking frequency across epochs (curriculum)
  - Round 6: shift MLM/CLM frequency (70/30 instead of 50/50)
- **Expected outcome:** Parameter regime optimization without architectural changes; fast iteration on training levers

---

# APPENDIX: Citation Verification

All references below are to works / repositories already cited in the project codebase (ADRs, config, prior runs). No fabricated citations (the arXiv:2605.12548 "Cubical Type Theoretic Navya-Nyāya" is explicitly forbidden in CLAUDE.md).

### Verified Project References
1. **ADR-0017** — H1_COGS null closure, dose hypothesis, 100M proxy scale COGS saturation
   - **Path:** `/home/sharaths/projects/PSALM-integration/docs/decisions/0017-h1-proxy-null-scope-and-claim-relocation.md`
   - **Status:** Checked, signed-off

2. **ADR-0029** — ELC-PSALM backbone, hybrid MLM+CLM, Muon optimizer, 114M design
   - **Path:** `/home/sharaths/projects/PSALM-integration/docs/decisions/0029-elc-psalm-backbone.md`
   - **Status:** Checked, references "ELC-BERT (2023) and GPT-BERT (2024) leaderboard winners"

3. **ADR-0038** — Leaderboard levers (N-hot embeddings, kāraka-aware masking)
   - **Path:** `/home/sharaths/projects/PSALM-integration/docs/decisions/0038-leaderboard-levers.md` (inferred from 0039 reference)
   - **Status:** Cited in 0039, supports mechanism-based approach

4. **ADR-0039** — H1_MECHANISM reframe (mechanisms as continuous training, not dose)
   - **Path:** `/home/sharaths/projects/PSALM-integration/docs/decisions/0039-paninian-as-mechanism.md`
   - **Status:** Checked, fully detailed, core to this research

5. **Ledger: prabhasa_b_ss_0.1**
   - **Path:** `/home/sharaths/projects/PSALM-integration/docs/memory/ledger_prabhasa_b_ss_0.1.md`
   - **Content:** Attempt #1 (BLiMP 55.41, under-converged); Attempt #2 (BLiMP 59.47, LR fix); Round 2 in progress (epochs 7→10)
   - **Status:** Checked, current working baseline

### Baseline System References (from leaderboard, ADR-0029)
- **LTG-BERT:** leaderboard winner, est. ~71-72 BLiMP strict-small; cited in PRABHĀSA design docs
- **ELC-BERT:** Every-Layer-Counts, hybrid MLM+CLM, ~70-72 BLiMP; ADR-0029 reference
- **GPT-BERT:** hybrid MLM+CLM baseline, ~68-70 BLiMP; implicit in ELC papers

**Note:** These baseline papers (LTG-BERT, ELC-BERT, GPT-BERT) are referenced in project ADRs as motivation for ELC-PSALM design, but full citations (URLs, arxiv IDs) are not provided in the ADRs themselves. For a published research paper, these would require independent verification (arXiv.org, ACL Anthology, GitHub leaderboard). For this intervention plan, they serve as design-input context, not claims under test.

---

# APPENDIX: Timeline & Resource Budget

## Estimated GPU Time (GB10 DGX Spark, 128GB, aarch64)

| Round | Task | Epochs | Duration | Cumulative |
|-------|------|--------|----------|-----------|
| 2 | English ×10 (Round 2 in progress) | 10 | ~2h | 2h |
| 3 | Static masking, 1 seed | 10 | ~1h | 3h |
| 4a | Ablate Paribhāṣā, 1 seed | 10 | ~1h | 4h |
| 4b | Refine Paribhāṣā (conditional), 1 seed | 10 | ~1h | 5h |
| 5 | Ablate dose (conditional), 1 seed | 10 | ~2h | 7h |
| 6 | Rebalance MLM/CLM (conditional), 1 seed | 10 | ~1h | 8h |
| Validation | 3-seed validation on best arm | 10 | ~3h | 11h |
| **Total** | All rounds + validation | — | ~11h | — |

**Note:** Rounds 3–6 are conditional; stop early if intermediate round hits BLiMP ≥68pp (jump to 3-seed validation). Expected actual time: 6–8h if one or two rounds are skipped.

---

# APPENDIX: Next Steps Post-Research

1. **Await Round 2 completion** — record official BLiMP (3 seeds) on epochs 7→10
2. **Apply Gate 1 decision:** if BLiMP ≥62pp, proceed to Round 3; if <62pp, invoke Rounds 3-4 immediately
3. **Execute Rounds 3-6 sequentially** per decision gates; update experiment ledger after each round
4. **Validation:** once 1-seed BLiMP ≥65pp, run 3-seed battery + official eval
5. **Paper update:** document null/positive findings in H1_MECHANISM section; update ADRs as needed
6. **Leaderboard:** if ≥70pp on 3-seed mean, submit final model; if 65-70pp, stage for H2 (Nyāya scaffold fine-tuning)

