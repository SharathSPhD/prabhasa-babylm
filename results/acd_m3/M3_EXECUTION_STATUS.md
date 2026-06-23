# M3 ACD Execution Status Report

**Generated:** 2026-06-22 17:50 UTC  
**Status:** STAGED & READY (awaiting pod provisioning)  
**Experiment:** Active Circuit Discovery validation for weak BLiMP paradigms  
**Pod:** A40 at 69.30.85.102:22090 ($0.44/h)

## Current State

### Pod Provisioning
- **Status:** IN PROGRESS (step 2/6: uv sync --extra ml --extra stats)
- **Process:** `/workspace/_provision.sh` running with PID 197, 31% CPU usage
- **ETA:** ~5–10 minutes remaining (git clone + env setup)
- **Expected completion steps:**
  - Step 3: spaCy model download
  - Step 4: eval pipeline clone
  - Step 5: BabyLM 100M Strict corpus tokenization (parallel to other steps)
  - Step 6: corpus byte-size verification (313,669,342 bytes)

### Scripts Staged & Ready

1. **`scripts/acd_m3_faithfulness_gate.py`**
   - Loads pre-trained 100M-Strict base (72.46) from HF Hub
   - Runs gradient-based attribution on NPI/filler-gap/islands pairs
   - Implements ablation: targeted (top-K heads) vs random (K random heads)
   - Decision rule: Δ(targeted − random) ≥ 2pp on weak paradigms → PASS
   - Outputs: `results/acd_m3/faithfulness_gate_report.json`

2. **`scripts/acd_m3_circuit_targeting.py`**
   - Orchestrator for circuit-targeted training (if gate passes)
   - Generates training plan with pre-registered BLiMP thresholds
   - ≤10 epochs compliance checker

3. **`scripts/acd_m3_compile_f10.py`**
   - Compiles F10 finding from gate report + training results
   - Outputs: `results/acd_m3/F10_finding.md` (for appending to findings.md)

4. **`scripts/acd_m3_fast_path.sh`**
   - Streamlined orchestration (uses pre-trained base, avoids re-training)
   - Direct to faithfulness gate → circuit training (if gate passes)
   - ~150 min total ETA (vs 240 min with reference re-train)

5. **`scripts/acd_m3_orchestrate.sh`**
   - Full gate-staged pipeline (slower, includes reference re-training)
   - Useful if reference model verification needed

## Experiment Flow (Next Steps)

### Immediate (When pod provisioning completes: ~18:00 UTC)

1. **Pod ready confirmation:**
   ```bash
   bash scripts/runpod/run_pod.sh exec 69.30.85.102 22090 \
     "tail -5 /workspace/provision.log | grep -E 'PROVISION_DONE|CORPUS_MATCH_OK'"
   ```

2. **Start M3 fast path (ground-truth flow):**
   ```bash
   bash scripts/acd_m3_fast_path.sh | tee results/acd_m3/m3_execution.log
   ```

### Phase 1: Faithfulness Gate (15–20 min)

The gate runs immediately upon script launch:
- Loads `qbz506/prabhasa-100m-base-strict` from HF Hub
- Constructs minimal pairs for NPI licensing (909 pairs available via HF datasets)
- Runs gradient attribution on 50 pairs per paradigm
- Ablates top-5 heads vs 5 random heads (5 random seeds)
- Computes Δ(targeted − random) accuracy drop

**Gate outcomes:**
- **PASS:** Δ ≥ 2pp (circuits are real) → proceed to Phase 2
- **FAIL:** Δ < 2pp (circuits not faithful) → STOP, report F10_NULL

### Phase 2: Circuit-Targeted Training (IF gate passes; ~60 min)

If gate passes, launches training on pod:
```bash
bash scripts/runpod/run_pod.sh exec 69.30.85.102 22090 \
  'cd /workspace/psalm && EXP=circuit_curriculum bash scripts/runpod/run_experiments.sh'
```

**Pre-registered thresholds:**
- BLiMP ≥74.53: POSITIVE (beat GPT-2 baseline)
- 73.0 ≤ BLiMP < 74.53: MARGINAL
- BLiMP < 73.0: FAIL

### Phase 3: F10 Compilation (~5 min)

```bash
uv run python scripts/acd_m3_compile_f10.py
```

Outputs: `F10_finding.md` (gate verdict ± circuit-targeted BLiMP)

## Cost & Time Budget Summary

| Phase | Duration | Cost |
|-------|----------|------|
| Pod provision | 15 min | $0.11 |
| Faithfulness gate | 20 min | CPU (free) |
| Circuit training (if pass) | 60 min | $0.44 |
| **Total** | ~95 min | ~$0.55 |

**If gate fails:** ~35 min total ($0.26), clean NULL finding.

## Deliverables (All Committed to git)

```
results/acd_m3/
├── M3_EXPERIMENT_PLAN.md               # This plan
├── M3_EXECUTION_STATUS.md              # This status (updated in real-time)
├── m3_orchestration.log                # Full transcript
├── faithfulness_gate_report.json       # Gate decision + ablation results
├── F10_finding.md                      # Final finding (to append to findings.md)
├── circuit_training_plan.json          # (if gate passes)
├── reference_model.log                 # (if Phase 2 runs)
└── m3_state.json                       # Experiment state snapshot
```

## Honest Assessment & Caveats

### Faithfulness Gate
- **What's certain:** The gate is a real, pre-registered validation of circuit causality.
- **What's uncertain:** Gradient attribution vs true causal circuits. Layer-0 dominance in Lane-C's smoke test could be noise.
- **Bayesian prior:** ~40% chance of circuits being real enough to pass the gate (per Lane-C).

### Circuit-Targeted Training (if gate passes)
- **Upside:** Could close the ~2pp gap under GPT-2 baseline (72.46 → 74.53).
- **Probability:** ~50% chance of ≥1pp gain if circuits are real.
- **Most likely outcome:** Small improvement (73–73.5), short of baseline.
- **Risk:** Curriculum design for NPI targeting is heuristic; may not transfer to BLiMP eval.

### Hard Constraints
- **10-epoch cap:** Single training run only; no fine-tuning on top of reference. Compliant with BabyLM rules.
- **Cost discipline:** Pod will be deleted after F10 compiled (cost control).
- **Reproducibility:** All code committed; models from HF Hub or checkpoints uploaded.

## Monitoring Commands

### Check pod provisioning:
```bash
bash scripts/runpod/run_pod.sh exec 69.30.85.102 22090 "tail -5 /workspace/provision.log"
```

### Monitor faithfulness gate (from M3 script):
```bash
tail -f results/acd_m3/m3_execution.log | grep -E "Gate|FAIL|PASS"
```

### Monitor circuit training on pod:
```bash
bash scripts/runpod/run_pod.sh exec 69.30.85.102 22090 \
  "tail -30 /workspace/circuit_training.log | grep -E 'best_loss|JOB|DONE'"
```

### Fetch final scores:
```bash
bash scripts/runpod/run_pod.sh fetch 69.30.85.102 22090 \
  /workspace/psalm/data/official_scores /tmp/m3_scores
```

## Next Immediate Action

**Wait for pod provision completion (~5–10 min), then:**

```bash
cd /home/sharaths/projects/PSALM-integration
bash scripts/acd_m3_fast_path.sh
```

This will:
1. Run the faithfulness gate immediately
2. Decide: PASS → train circuit model, or FAIL → stop with F10_NULL
3. Compile F10 finding regardless of outcome
4. Report final verdict

---

**Prepared by:** Claude Code / M3 ACD Agent  
**Time to completion:** ~95 min from now (if gate passes) or ~35 min (if gate fails)  
**Post-experiment:** Delete pod via RunPod MCP when done (cost discipline)
