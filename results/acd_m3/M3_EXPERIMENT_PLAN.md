# M3 ACD Experiment Plan — Final Checklist

**Status:** Provisioning in progress (ETA: ~10 more minutes)  
**Pod:** A40 at 69.30.85.102:22090 ($0.44/h)  
**Branch:** feature/v0.2-arch-sweep  
**Date:** 2026-06-22

## Objective

Determine whether Active Circuit Discovery (ACD) circuit-targeting of the weak BLiMP paradigms (NPI licensing, filler-gap, islands) can push the validated 100M-Strict base (BLiMP 72.46) toward/past the GPT-2 baseline (74.53) — within the BabyLM 10-epoch limit.

## Gate-Staged Pipeline

### Phase 1: Provision
- [ ] Bootstrap pod: `uv sync`, eval pipeline, BabyLM corpus
- [ ] Verify corpus byte-size match (strict/english_base.bin == 313,669,342 bytes)
- **Status:** IN PROGRESS (uv sync ~70% complete)

### Phase 2: Reference Model (100M-Strict base)
- [ ] Train on Strict 100M corpus, 10 epochs, AdamW 3e-4
- [ ] Recipe: `--no-layer-routing --no-muon --peak-lr 3e-4 --english-epochs 10 --base-dir data/corpora/strict`
- [ ] Official eval (BLiMP full + supp)
- [ ] Expected: BLiMP ~72.46 (the validated v0.2 recipe)
- [ ] Checkpoint → feed into faithfulness gate
- **Estimated time:** ~60 min (100M, 10 epochs, A40)

### Phase 3: Faithfulness Gate (DECISIVE)
- [ ] Load reference model checkpoint
- [ ] Run gradient-based attribution on NPI/filler-gap/islands minimal pairs
- [ ] Ablation study: zero-out top-K heads vs random K heads, measure BLiMP drop
- [ ] Gate decision rule: Δ(targeted − random) ≥ 2pp on weak paradigms → PASS
- **If FAIL:** Stop here, report F10_NULL (circuits not real)
- **If PASS:** Proceed to Phase 4
- **Estimated time:** ~15 min (CPU, 50 pairs per paradigm)

### Phase 4: Circuit-Targeted Training (IF gate passes)
- [ ] Train 100M model with circuit-aware targeting
- [ ] Strategy: curriculum upweighting NPI/filler-gap/island examples
- [ ] Constraints: ≤10 epochs total (BabyLM hard constraint)
- [ ] Recipe: Same base + circuit curriculum
- [ ] Official eval (BLiMP full + supp)
- **Pre-registered decision rule:**
  - BLiMP ≥74.53: POSITIVE (beat baseline)
  - 73.0 ≤ BLiMP < 74.53: MARGINAL (progress but not baseline)
  - BLiMP < 73.0: FAIL (no improvement)
- **Estimated time:** ~60 min (100M, 10 epochs, A40)

### Phase 5: Compile F10 Finding
- [ ] Fetch both model checkpoints + official scores
- [ ] Write F10 entry: gate decision + final verdict
- [ ] Append to research/memory/findings.md
- [ ] Commit + push

## Honest Caveats

1. **Faithfulness uncertainty:** Lane-C's attribution identified layer-0 heads, but early-layer dominance could be noise. The faithfulness gate is the real validation.

2. **Circuit-targeting upside is modest:** ~2pp gap under baseline. If circuits are real, they'll help, but no guarantee of beating 74.53.

3. **Bayesian prior:** ~40% chance of >1pp gain from circuit targeting (Lane-C's own assessment). Most likely outcome: NULL finding (circuits not localized or not causal).

4. **10-epoch hard cap:** No fine-tuning beyond initial training — that would violate BabyLM rules. Single run only.

## Cost & Time Budget

| Phase | Resource | Time | Cost |
|-------|----------|------|------|
| Provision | Pod setup | ~15 min | ~$0.11 |
| Reference | 100M train | ~60 min | ~$0.44 |
| Gate | Attribution + ablation | ~15 min | ~$0.11 (CPU) |
| Circuit-target | 100M train (if pass) | ~60 min | ~$0.44 |
| **Total** | | ~150 min (~2.5 hr) | ~$1.10 |

**Cost discipline:** If reference train blows the budget (>$0.60), report partial results and stop.

## Orchestration Commands

From GB10:

```bash
# Wait for provision + start full pipeline
bash scripts/acd_m3_orchestrate.sh

# If manual steps needed:
# 1. Start reference training
bash scripts/runpod/run_pod.sh exec 69.30.85.102 22090 \
  'cd /workspace/psalm && EXP=seed0 bash scripts/runpod/run_experiments.sh'

# 2. Run faithfulness gate (on GB10, once model ready)
uv run python scripts/acd_m3_faithfulness_gate.py

# 3. If gate passes, start circuit-targeted training
bash scripts/runpod/run_pod.sh exec 69.30.85.102 22090 \
  'cd /workspace/psalm && EXP=circuit_curriculum bash scripts/runpod/run_experiments.sh'

# 4. Fetch results
bash scripts/runpod/run_pod.sh fetch 69.30.85.102 22090 \
  /workspace/psalm/data/official_scores /tmp/m3_scores
```

## Deliverables

1. **faithfulness_gate_report.json** — per-paradigm ablation results + gate decision
2. **reference_model.log** — training log (100M base, 72.46)
3. **circuit_targeted.log** (if gate passes) — training log (100M circuit-targeted)
4. **official_scores/*.json** — BLiMP + supplement scores
5. **F10_finding.md** — final finding (gate verdict ± circuit-targeted BLiMP)
6. **m3_orchestration.log** — full orchestration transcript

## Closure Criteria (Ralph-loop contract)

1. **TECHNICAL:** Tests pass, linting clean (code is from committed feature branch).
2. **EMPIRICAL:** All phases complete, gate decision logged, BLiMP scores recorded.
3. **INTEGRITY:** Faithfulness validated (or null clearly documented); comparison fair (both models same 10-epoch budget).
4. **ARTIFACTS:** Code pushed, results logged in results/acd_m3/, paper section updatable.
5. **MEMORY:** F10 entry appended to findings.md, experiment ledger updated.
6. **SIGN-OFF:** Human review of F10 interpretation before merge-to-main.

## Next Steps

1. **Pod provisioning:** Wait for PROVISION_DONE + CORPUS_MATCH_OK (~5–10 min remaining)
2. **Start orchestration:** `bash scripts/acd_m3_orchestrate.sh`
3. **Monitor pod logs:** `bash scripts/runpod/run_pod.sh exec ... tail -30 /workspace/reference_model.log`
4. **Compile F10:** `uv run python scripts/acd_m3_compile_f10.py`
5. **Delete pod when done:** `uv run mcp_runpod delete-pod <POD_ID>` (cost control)

---

**Prepared:** 2026-06-22 by Claude Code  
**Expected completion:** ~03:30 UTC (full pipeline, both phases)  
**Pod deletion:** Automatic after F10 appended (if gate fails), or after official eval (if gate passes)
