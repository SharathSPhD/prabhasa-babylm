# SPEC 0001 — PRAJÑĀ: Autonomous Pāṇinian–Nyāya Research Harness

Status: ACTIVE · Owner: autonomous agent · Spec-driven (OpenSpec-style: this spec is
the contract; code/experiments conform to it; changes amend the spec first).

## 1. Purpose

A self-restarting, 24/7-capable research system that conducts **real** experimental
research at the intersection of **Pāṇinian grammar** (Aṣṭādhyāyī, prakriyā, kāraka),
**Navya-Nyāya** (śābdabodha = verbal cognition, vyāpti = pervasion), and
**vyutpattivāda / śābdabodha** theories of word-derivation and meaning-composition —
applied to **small/efficient language models** (BabyLM tracks, scaling to SLM/LLM).

The system **generates its own research questions, designs experiments, runs them on
real data/compute, validates honestly, and loops** — refining the paper, the GitHub
Pages site, and its own tooling as it goes. It is its own adversarial reviewer.

## 2. Invariants (NEVER violated — inherited from CLAUDE.md)

- **No mock / synthetic / surface results.** Every reported number comes from a real
  run on real data on the GPU. No fabrication.
- **Citation integrity.** No fabricated citations; `arXiv:2605.12548` is banned. Every
  cite resolves to a real work.
- **Closure contract** (6 layers) gates every finding: TECHNICAL, EMPIRICAL, INTEGRITY
  (Tarka self-objection), ARTIFACTS, MEMORY, SIGN-OFF. Never "failed" on attempt 1;
  never a NULL without ≥2 documented interventions.
- **Statistical honesty.** ≥3 seeds for headline claims; mean ± CI; the seed RNG must
  genuinely vary (verified by checkpoint md5 divergence — see the seed-collapse bug).
- **GPU-only training** (single DGX Spark GB10). No CPU training. One run at a time;
  never launch GPU work while the GPU is occupied.
- **Honest framing.** Sample efficiency + compositional generalization + epistemic
  discipline are the targets, not frontier parity.

## 3. Architecture

```
                ┌──────────────────────────────────────────┐
   keep-alive → │  CYCLE RUNNER (research/cycles/run.md)     │ ← cron + ScheduleWakeup
   (cron)       │  read state → pick next action → execute   │
                │  → validate → log → update state → commit  │
                └───────────────┬──────────────────────────┘
                                │ dispatches (Workflow / Agent / teams)
        ┌───────────────────────┼───────────────────────────────┐
   specialist subagents (.claude/agents/):                       │
   - panini-vyakarana   (Aṣṭādhyāyī, prakriyā, sandhi, kāraka)    │
   - nyaya-darshana     (śābdabodha, vyāpti, hetvābhāsa)          │
   - shabdabodha-vyutpatti (derivation/meaning composition)      │
   - babylm-experimentalist (train/eval/ablate, real GPU)        │
   - adversarial-reviewer (Tarka: attack every finding)          │
   - paper-smith        (LaTeX paper + Pages, citation-checked)   │
   - memory-keeper      (journal, ledger, knowledge store)        │
        └───────────────────────┴───────────────────────────────┘
   memory: research/memory/*  +  docs/memory/*  +  experiment ledger
```

## 4. Durable state (survives session/context limits)

- `research/memory/state.json` — single source of truth: `cycle`, `phase`,
  `current_rq`, `gpu_lock`, `open_questions[]`, `closed_questions[]`, `next_action`.
- `research/memory/journal.md` — append-only narrative log (every cycle writes).
- `research/memory/findings.md` — validated findings + nulls (closure-gated).
- `research/open_questions.md` — the RQ backlog (prioritised).
- Each wake-up READS state.json first and resumes deterministically. Idempotent steps;
  a step that already produced its artifact is skipped.

## 5. The research loop (one cycle)

1. **Orient** — read state.json + journal tail. Detect GPU lock (nvidia-smi).
2. **Select** — pick the highest-value open action: (a) a pending experiment result to
   harvest, (b) the next RQ to design, (c) an adversarial review, (d) a paper/Pages
   refinement, (e) a tooling/agent improvement. Never start GPU work if GPU busy →
   do GPU-free work instead.
3. **Act** — execute via the right specialist agent / Workflow / team. Real runs only.
4. **Validate** — adversarial-reviewer attacks the result; verify reproducibility,
   seed-variance, statistical test, comparison fairness.
5. **Record** — append journal, update findings/ledger, update state.json, commit
   (canonical identity, no Co-Authored-By — single-contributor repo).
6. **Schedule** — ensure the next wake-up is queued (cron alive; ScheduleWakeup backup).

## 6. Research program (initial RQ backlog — see research/open_questions.md)

Anchored on the validated result (pure-MLM + kāraka masking + Vidyut N-hot → Strict
BLiMP 73.06). The program deepens the Pāṇinian/Nyāya core:

- **RQ-A (mechanism causality):** Does kāraka-aware masking causally lift BLiMP
  agreement/argument-structure paradigms vs a frequency-matched control, at 100M?
- **RQ-B (śābdabodha objective):** An auxiliary head predicting the kāraka relational
  graph (verbal-cognition structure) — does it improve compositional generalization
  (COGS/CFQ) per-token?
- **RQ-C (vyutpattivāda curriculum):** prakriyā-ordered (derivation-step) data
  presentation — sample-efficiency effect?
- **RQ-D (Nyāya inference):** vyāpti-scaffolded fine-tuning — inference-quality readout.
- **RQ-E (real engine at scale):** does the real Vidyut/spaCy-kāraka engine beat the
  heuristic at 100M (10M was null)?
The harness will ADD RQs as findings open new questions (self-directed).

## 7. Keep-alive

- Primary: `cron` (CronCreate) firing the cycle runner on an interval with the
  autonomous-loop sentinel. Backup: `ScheduleWakeup` re-queues if a session ends.
- Guardrail: the cycle is cheap when idle (GPU busy / nothing to do → GPU-free tooling
  or a short sleep), so 24/7 firing does not waste compute.

## 8. Done / success

The program never "completes" (it's a standing research loop), but each FINDING closes
via the 6-layer contract. Milestones: validated mechanism causality (RQ-A), a published
refined paper + Pages, a reusable Pāṇinian-NLP toolkit, and an honest body of
findings+nulls. Human sign-off is deferred (user unavailable 1 week) → findings are
staged as "SIGN-OFF PENDING" and merged to a `research/autonomous` line, not main.
