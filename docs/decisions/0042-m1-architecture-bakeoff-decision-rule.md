# ADR-0042 — M1 architecture bake-off: pre-registered decision rule

- **Status:** Accepted (2026-06-21), pre-registered BEFORE the runs (no p-hacking)
- **Branch:** `feature/v0.2-arch-sweep`
- **Depends on:** ADR-0041 (official scorer = inner-loop metric)

## Context

v0.1 ELC = ELC-BERT every-layer-counts routing **+** GPT-BERT-style hybrid MLM/CLM objective.
The objective is already general (`--objective {mlm,hybrid,clm}`), so the only architectural
element separating ELC from a clean "GPT-BERT-class" hybrid is the **every-layer routing**
(`LayerRouteCombiner`), now toggleable via `--no-layer-routing` (commit 6e2cea3). v0.1 ELC sits
below the GPT-2 baseline on the official scorer (SS 59.46 < 65.08), so we must learn whether the
routing helps or hurts before scaling.

## Decision — the bake-off

Compare two arms that are **identical except for routing**, on the OFFICIAL scorer at the 10M
Strict-Small proxy:

| Arm | Flag | Meaning |
|---|---|---|
| `m1_elc`     | (default)            | routing ON  (ELC every-layer-counts) |
| `m1_vanilla` | `--no-layer-routing` | routing OFF (vanilla pre-norm stack = GPT-BERT-class) |

**Held identical (both arms):** pure-MLM objective (F1 winner; official-scorer-aligned), RoPE,
N-hot embeddings, structured masking, dose OFF (`--dose-arms A --dose-epochs 0`, english-only),
`--english-epochs 10`, `--batch-size 256 --max-seq-len 192`, `--muon-lr 0.02`, dropout 0.1,
grad-clip 1.0, vocab 20000, corpus `data/corpora/strict_small`.

**Procedure (staged, GPU-frugal):**
1. Cut: 1 seed per arm → eval official BLiMP (`run_official_eval.py --stage zero-shot`).
2. Add a 2nd seed to BOTH arms (architecture decision needs ≥2 seeds + gives per-arm CV — the
   folded M0c seed-stability check on the official scorer).

**Pre-registered decision rule (on mean official BLiMP, n=2):**
- vanilla − elc ≥ **+1.0 pt** → adopt **vanilla** (routing off) as the v0.2 backbone.
- elc − vanilla ≥ **+1.0 pt** → keep **ELC** routing.
- within ±1.0 pt (noise) → keep **ELC** (incumbent, already integrated) unless vanilla is
  materially faster/more stable at equal score.
- Report per-arm **CV(BLiMP)** over the 2 seeds. If the chosen arm's CV > 5%, intervention:
  add Muon LR warmup + a 3rd seed before locking.

**Escalation (intervention discipline, ≥2 before any null):** if NEITHER arm shows a path toward
the SS baseline (both well below ~63 at 10M), that is intervention #1 → diagnose the official/internal
gap drivers (length-mismatch handling, objective, tokenizer) and run intervention #2 (e.g. a faithful
GPT-BERT port, or scoring-gap-targeted changes) before declaring the architecture a null.

## Consequences

- A clean, cheap, fair answer to "does ELC's routing earn its place on the leaderboard scorer?"
- The chosen backbone carries forward to M2 (Pāṇinian rigour) and M3 (ACD circuits).
- Pāṇinian mechanisms (N-hot, structured masking) are ON in both arms, so the bake-off does not
  confound the architecture question with the mechanism question (mechanisms are studied in M2).

## Outcome (2026-06-21)

Bake-off ran on GB10 (10M, official scorer, 2 seeds/arm). **Official BLiMP:** ELC (routing ON)
61.32/61.64 → mean **61.48** (CV 0.37%); vanilla (routing OFF) 62.74/62.55 → mean **62.65** (CV 0.21%).
Δ(vanilla−elc) = **+1.17 pt ≥ +1.0** → decision rule fires.

**DECISION: adopt VANILLA (`--no-layer-routing`) as the v0.2 backbone.** ELC's every-layer routing costs
~1.2 BLiMP on the leaderboard scorer; it is retired. Pāṇinian mechanisms graft onto the vanilla GPT-BERT-class
stack. Both arms seed-stable at 10M (CV < 0.5%; folded M0c satisfied). Both beat v0.1 official SS (59.46);
vanilla 62.65 approaches baseline 65.08 (≈2.4 pt gap = the M2/M3 leading-indicator target). The 100M
backbone-confirm is **folded into M4** (the first 100M finals run validates vanilla at scale) to avoid a
separate RunPod job. See findings F5.
