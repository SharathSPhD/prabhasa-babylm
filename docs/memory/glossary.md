# Glossary

Shared vocabulary for PSALM. Linguistic and logical terms are defined as the
program uses them, not exhaustively.

## Pāṇinian grammar (Vyākaraṇa)

- **Aṣṭādhyāyī** — Pāṇini's grammar of Sanskrit in ~4,000 sūtras; a formal,
  generative rule system with metarules and rule-ordering.
- **Sūtra** — an aphoristic rule.
- **Kāraka** — the syntactic-semantic role a noun plays relative to the verb
  (agent/kartā, object/karma, instrument/karaṇa, recipient/sampradāna,
  source/apādāna, locus/adhikaraṇa). PSALM uses kāraka labels as gold parse
  annotations from the generator.
- **Sandhi** — euphonic combination at morpheme/word boundaries; motivates a
  sandhi-aware tokenizer.
- **Saṃsādhanī** — the computational Sanskrit toolset (Univ. of Hyderabad)
  whose generator PSALM wraps to stream `(sentence, kāraka parse, derivation)`.

## Navya-Nyāya (logic / epistemology)

- **Pramāṇa** — a valid means of knowledge (perception, inference, comparison,
  testimony). The H2 scaffold types reasoning steps by pramāṇa.
- **Saṃśaya** — articulated doubt; the first phase of the 6-phase method.
- **Pañca-avayava** — the five-membered syllogism (claim, reason, example,
  application, conclusion).
- **Vyāpti** — pervasion / invariable concomitance ("where smoke, there fire");
  the validity relation the Z3 verifier checks in H3.
- **Upādhi** — a defeating condition that limits a vyāpti.
- **Hetvābhāsa** — a fallacious reason (pseudo-probans); the H3 filter rejects
  these classes.
- **Nirṇaya** — ascertainment / settled conclusion; calibration (ECE) measures
  how well confidence tracks correctness here.
- **Tarka** — counterfactual / reductio reasoning. The closure contract's
  integrity gate borrows the name for the mandatory self-objection memo.

## Modeling & evaluation

- **Pre-pretraining** — a structural warm-up stage before main pretraining
  (Hu et al. use k-Shuffle Dyck; PSALM uses Pāṇinian synthetic Sanskrit).
- **k-Shuffle Dyck** — interleaved balanced-bracket language; the matched
  structural control for H1.
- **BabyLM** — a fixed-budget pretraining benchmark (10M / 100M words); PSALM's
  peer-comparison frame.
- **Compositional generalization** — generalizing to novel combinations of known
  parts; measured by SCAN, COGS/ReCOGS, CFQ.
- **μTransfer** — hyperparameter transfer across model widths (μP) used to
  promote the best arm to 350M without re-tuning.
- **GRPO** — Group Relative Policy Optimization; RL post-training used in H2.
- **GBNF** — GGML BNF grammar format for constrained decoding (H3).
