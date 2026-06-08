# Open Research Questions (PRAJÑĀ backlog)

Prioritised. Each RQ: hypothesis, pre-registered metric+threshold, design, status.
The harness pulls from the top; findings spawn new RQs (append below).

## P0 — close current plan first
- [running] SS pure-MLM 3-seed CI (seed0 65.22, seed1 63.31, seed2 ⏳) → honest CI.
- [ ] Reconcile git worktrees; stage paper/Pages/HF (sign-off-pending branch).

## P1 — mechanism causality (the H1 core, at scale)
- **RQ-A** Does kāraka-aware masking causally lift BLiMP *agreement* + *argument-structure*
  paradigms vs a **frequency-matched random-masking control**, at 100M, ≥3 seeds?
  - H: kāraka-on > control by ≥1.0pp on the agreement/arg-structure BLiMP subset (paired bootstrap, Holm).
  - Design: 100M pure-MLM, arm1=kāraka-masking, arm2=freq-matched control (same total mask budget),
    eval BLiMP per-paradigm; compare on the linguistically-targeted subset.
  - Why it matters: turns "mechanisms are ON in the 73.06 model" into a *causal* claim.

## P2 — śābdabodha-structured objective (verbal cognition)
- **RQ-B** Auxiliary head predicting the kāraka relational structure (kartā/karma/kriyā graph)
  on top of pure-MLM — effect on compositional generalization (COGS / CFQ / BLiMP-supplement)?
  - H: +aux improves COGS exact-match by ≥2pp at equal pretraining tokens.
  - Build: a real kāraka-graph target from spaCy/Vidyut parses; multi-task loss; ablate weight.

## P3 — vyutpattivāda curriculum (derivation order)
- **RQ-C** prakriyā-ordered presentation (simpler→derived forms; morphologically-graded curriculum)
  vs random order — sample efficiency on Sanskrit morphology + English BLiMP transfer?
  - Build on the real Vidyut prakriyā engine (already generates derivations).

## P4 — Navya-Nyāya inference scaffold
- **RQ-D** vyāpti(pervasion)-structured fine-tuning data — inference-quality readout
  (BLiMP subset or a small annotated entailment set); H2 reframed for scale.

## P5 — real engine at scale
- **RQ-E** Real Vidyut/spaCy-kāraka masking at 100M vs heuristic (10M was null 62.08<64.09).
  Does linguistic fidelity pay at 10× data? (Partial run was killed; re-decide value vs RQ-A.)

## P6 — depth (theory → mechanism)
- **RQ-F** Encode Aṣṭādhyāyī *paribhāṣā* (meta-rules) ordering as an attention/masking prior.
- **RQ-G** śābdabodha ākāṅkṣā (expectancy) / yogyatā (compatibility) / sannidhi (proximity) as
  three concrete inductive biases — operationalise + test each.

## Appended by the harness (self-directed)
<!-- new RQs spawned by findings go here, dated -->

## Appended by harness
- **RQ-H** (from F1, 2026-06-08): At what corpus scale does pure-MLM overtake hybrid, and
  WHY? Hypothesis: the CLM head competes for capacity; at small data both objectives
  underfit (neutral), at larger data the MLM signal saturates faster and the CLM steps
  waste half the gradient. Test: 25M + 50M intermediate scales, pure-MLM vs hybrid,
  3 seeds, locate the crossover. Cheap (≤2h each) and scientifically clean.
