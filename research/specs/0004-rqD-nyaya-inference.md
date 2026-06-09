# SPEC 0004 — RQ-D: Hetvābhāsa discrimination as a Nyāya inference-quality probe

Status: DESIGN (queued behind RQ-A/F2 + RQ-B). Parent: SPEC 0001. Author: nyaya-darshana
(human-controlled citations). GPU: a small fine-tune/probe (≤1h), not a pretrain.

## Honest reframe of the H2 null
H2 (generate valid Pañcāvayava syllogisms) was a documented NULL — chance (52.75%) at 114M.
A 100M encoder cannot *generate* valid multi-step inference. RQ-D reframes to what IS feasible
at this scale: **discrimination**, not generation. The model classifies an inference snippet as
*valid* or as one of the Navya-Nyāya **hetvābhāsa** (pseudo-probans / fallacy) types. This is a
supervised classification readout — tractable at 100M — and a genuine test of whether the model
represents inferential validity.

## The Nyāya grounding (real)
Navya-Nyāya's **anumāna** (inference) is licensed by **vyāpti** (invariable concomitance,
hetu⇒sādhya). An inference fails via a **hetvābhāsa** — a reason that *appears* probative but is
not. The classical taxonomy (Gautama's Nyāya-sūtra; analysed in Matilal):
- **savyabhicāra** (anaikāntika) — inconclusive / deviating hetu (present where sādhya is absent)
- **viruddha** — contradictory (hetu proves the opposite)
- **asiddha** — unestablished (the hetu itself is not proven of the pakṣa)
- **satpratipakṣa** — counterbalanced (an equal opposing inference exists)
- **bādhita** — sublated (the sādhya is contradicted by stronger pramāṇa)
Each is a PRECISE failure mode of vyāpti — a real, typed label set, not invented.

## Question
Does the Pāṇinian-mechanism model (best Strict recipe) discriminate valid inference from
hetvābhāsa types more **sample-efficiently** than a matched no-mechanism baseline?

## Design (feasible)
- **Task:** 6-way classification {valid, savyabhicāra, viruddha, asiddha, satpratipakṣa, bādhita}
  on short inference snippets (pakṣa + hetu + sādhya + optional udāharaṇa).
- **Data (REAL, not synthetic noise):** start from the pramana repo's vyāpti-coherent Pañcāvayava
  chains (the valid class). Construct each fallacy by a PRINCIPLED, logic-faithful perturbation of
  a valid chain (e.g., savyabhicāra = insert a counter-instance breaking the vyāpti; viruddha =
  swap sādhya for its negation; asiddha = replace the hetu with one not predicable of the pakṣa).
  Each perturbation is defined by the fallacy's definition — auditable, not random corruption.
- **Models:** fine-tune (LoRA, small) the best Strict checkpoint (pure-MLM 73.06 / or the RQ-A/B
  winner) vs a matched Generic baseline; sample-efficiency curve (accuracy vs #training examples).
- **Metric:** 6-way macro-F1; sample-efficiency = #examples to reach a fixed F1. ≥3 seeds.
- **NULL allowed only after ≥2 interventions** (e.g., label-noise audit; class-balance fix).

## Why this differs from H2
H2 demanded generation+validity (impossible at 114M). RQ-D demands *recognition* of validity/
failure — a supervised probe the scale can support — and isolates the Pāṇinian mechanism's effect
on inferential representation, the honest residue of the H2 ambition.

## Citations (human-vouched real; verify exact pages before paper)
- Matilal, B.K., *The Word and the World* (Oxford UP, 2001) — verified (cycle 13).
- Ganeri, J., *Semantic Powers* (Oxford UP, 2011) — verified (cycle 13).
- Matilal, B.K., *Epistemology, Logic, and Grammar in Indian Philosophical Analysis* — to confirm.
- Gautama, *Nyāya-sūtra* (hetvābhāsa taxonomy) — primary source; cite a real critical edition/translation.
- (Do NOT cite arXiv:2605.12548 — fabricated.)

## Feasibility / cost
Small LoRA fine-tune + probe, ≤1h GPU per arm.

## Existing assets — REUSE, do NOT rebuild (cycle 19 inspection)
- **Labeled data already exists:** `psalm.domain.nyaya_generator.PanchaAvayavaGenerator.generate(n, seed)`
  yields labeled examples — valid + 4 hetvābhāsa types (SAVYABHICHARA, VIRUDDHA, ASIDDHA,
  SATPRATIPAKSHA), each constructed by its DEFINITION (savyabhicāra = hetu not pervaded by sādhya;
  viruddha = proves ¬sādhya; asiddha = pakṣa lacks the hetu; satpratipakṣa = equal counter-inference).
  This IS the principled, logic-faithful construction RQ-D needs — no separate perturbation generator.
  NOTE: bādhita is NOT generated → RQ-D is a **5-way** task {valid + 4 fallacies}, not 6-way. (Adding
  bādhita = one new `_generate_fallacy_badhita` method if 6-way is wanted later.)
- **Fine-tune harness exists:** `scripts/run_nyaya_h2_finetune.py` (LoRA + classification head + accuracy).
  Currently a 3-way NLI head — RQ-D's ONLY new code is: swap to a **5-way fallacy-type head**, feed the
  chain text → `fallacy_type` label (valid=class 0), and add the sample-efficiency curve (#examples→F1).
- So RQ-D ≈ ready: generate() → 5-way fine-tune (adapt run_nyaya_h2_finetune) → macro-F1, ≥3 seeds,
  Pāṇinian-mechanism arm vs matched baseline. Audit 50 generated examples for label fidelity first.
