# PSALM — project overview (Serena memory)

PSALM (Pāṇinian Structured pretraining for Small LAnguage Models) tests whether
Pāṇini's grammar, used as an unbounded generative data engine, gives small LMs a
stronger structural prior than artificial formal languages (k-Shuffle Dyck), and
whether a Navya-Nyāya layer makes their reasoning formally disciplined.

One multilingual model, three readings: English structural generalization,
Sanskrit competence, cross-lingual transfer. H1-heavy.

Hypotheses: H1 grammar prior (≥20% token savings vs Dyck OR ≥3pt compositional
gain), H2 Nyāya scaffold (sample-efficiency synergy test vs Generic-1B), H3
epistemic constraint kernel (GBNF + Z3 vyāpti + hetvābhāsa filter).

Single DGX Spark GB10 (aarch64). Public artifacts under HF `qbz506/psalm-*`.
Deliverables: dataset, model, repo, arXiv/IEEE paper, Astro site, Colab + HF
Space demos.

The program is contract-bound: phases close only via the six-layer Ralph-loop
closure contract (see closure_contract memory).
