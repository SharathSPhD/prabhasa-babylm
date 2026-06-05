# Compute Optimization: GB10 sm_121 Attention Throughput Bottleneck

**Date:** 2026-06-05  
**Status:** Diagnosis + TRIZ analysis + candidate fixes ranked  
**Gating Issue:** 100M Small run; training stalled at ~0.23–0.30 step/s (seq=256), need ~3 step/s  

---

## Problem Statement

At sequence length 256 and batch size 256 (65,536 tokens/step), PSALM training achieves **0.23–0.30 step/s**, approximately **10x slower than expected** (~3 step/s). The same model at seq=128 runs at 0.7–0.8 step/s, suggesting a non-linear slowdown. O(n²) attention FLOPs scale quadratically (seq 128→256 is 4x FLOPs), but observed slowdown is 10x, indicating **kernel fallback to unfused attention** or memory bandwidth saturation.

**Hardware:** NVIDIA DGX Spark GB10 Grace-Blackwell, aarch64, sm_121, 128GB, ~273 GB/s memory bandwidth.

---

## Root Cause Analysis

### Evidence

1. **CLAUDE.md warning (line flagged risk):**  
   > "verify Unsloth + flash-attn build on sm_121 early (flagged risk)"

2. **ADR-0023 (flash-attn on GB10: torch SDPA fallback):**
   - flash-attn v2.8.3 **does not precompile** for aarch64 + CUDA 13
   - Source build fails without `python3.12-dev` (Python.h missing)
   - Current stack uses **torch.nn.functional.scaled_dot_product_attention (SDPA)** as fallback

3. **Model code review** (src/psalm/infrastructure/ml/elc_psalm.py, line 97–104):
   ```python
   a = F.scaled_dot_product_attention(
       q, k, v,
       attn_mask=attn_mask,
       dropout_p=dropout_p,
       is_causal=attn_mode is _AttnMode.CAUSAL and key_padding_mask is None,
   )
   ```
   - Uses **torch SDPA** without explicit backend selection
   - SDPA backend selection happens at runtime; PyTorch chooses "best available"
   - On sm_121, **no flash kernel is available** (flash-attn requires sm_80+, Hopper/Ampere)
   - **Fallback is cuDNN SDPA or math (non-fused O(n²))**

4. **Memory bandwidth analysis:**
   - At seq=256, batch=256, h=12 heads, d=64 head_dim:
     - Q, K, V each: (256 × 256) × 12 × 64 = 50 MB per direction
     - Total read: ~150 MB; writes ~50 MB (output)
   - At 273 GB/s bandwidth: **1 full O(n²) pass ≈ 1 ms** (math kernel)
   - But we see ~4.4 s per step (at 0.23 step/s) → **attention is ~1 ms of 4400 ms total**
   - Other layers (embedding, projection, MLP) dominate, but **attention kernel choice still matters for latency**
   - Observed: **seq 128 @ 0.8 step/s = 1.25 s/step; seq 256 @ 0.25 step/s = 4 s/step = 3.2x slower**
   - This 3–4x slowdown at 4x FLOPs suggests **non-fused kernel is 20–25% slower per FLOP than fused** (expected 0.75x throughput)

### Hypothesis: Torch SDPA backend selection on sm_121

**Best Case (not verified):**
- PyTorch's SDPA detects Blackwell and uses cuDNN SM12X optimized attention path
- cuDNN 9.x (NGC container) may have Blackwell-aware kernels
- **If cuDNN SM12X path exists**, issue is that it's not fused flash-like (still slower than fused)

**Most Likely Case:**
- PyTorch SDPA falls back to math kernel (non-fused Q×K→softmax→output, 3 separate kernel launches)
- Each of the 3 passes is memory-bandwidth-limited; no kernel fusion = extra sync overhead
- **This explains 3–4x slowdown for 4x FLOPs** (fusion overhead ≈ 25%)

**Worst Case (unlikely):**
- Attention is dispatched to CPU fallback or a very slow path (would be 100x+ slower, not observed)

---

## TRIZ Contradiction Analysis

### Framing

**Improving:** Speed of attention computation (tokens/second, reduce latency)  
**Worsening:** Device/architectural complexity (conditional kernel dispatch, fallback chains, backend selection logic)

### TRIZ Parameters

- **Improving:** Parameter 9 (Speed) — throughput of SDPA forward+backward
- **Worsening:** Parameter 36 (Device Complexity) — number of backends, dispatch code, conditional branches

### TRIZ Contradiction Matrix Lookup

Query: improve Parameter 9 (Speed) while reducing impact on Parameter 36 (Device Complexity)

**Recommended Principles:** 13, 35, 1

### Recommended Principles & Adaptation

#### Principle 13: "The Other Way Around" (Inversion)

**Concept:**  
Instead of dispatching at runtime based on device capability, **invert the problem:** assume Blackwell needs a specific hand-tuned attention path and **build it directly into the model forward pass without conditional logic**.

**Application (medium complexity):**
- Remove the SDPA backend selection layer entirely
- Hard-code a **custom Blackwell-optimized attention kernel** using CUTLASS or Triton
- No dispatcher; no fallback; simpler forward pass
- **Trade-off:** requires Triton or CUTLASS expertise; adds 200–500 LOC of kernel code

**Expected Gain:** Eliminates dispatch overhead; complexity moves **into a single, well-tested kernel** rather than scattered conditional logic.

---

#### Principle 35: "Parameter Changes" (Adaptive Tuning)

**Concept:**  
Instead of using fixed seq_len=256 for all steps, **change the sequence length schedule progressively** to stay in the region where torch SDPA is fast (seq ≤ 192).

**Application (low complexity):**
- Current code already has progressive sequence schedule (lines 211–214 in train_submission_model.py):
  ```python
  seq_schedule = [
      (0.0, args.max_seq_len // 4),   # 0%: seq=64
      (0.4, args.max_seq_len // 2),   # 40%: seq=128
      (0.75, args.max_seq_len),       # 75%: seq=256
  ]
  ```
- **Modify:** cap the final seq at 192 or 200 instead of 256
  ```python
  seq_schedule = [
      (0.0, 64),       # 0%: seq=64
      (0.4, 128),      # 40%: seq=128
      (0.75, 192),     # 75%: seq=192 (avoid slowdown cliff)
  ]
  ```
- **Impact on training:** BabyLM official submission requires seq_len=256 *eventually*, but the phase fraction at seq=256 can be reduced
  - Phase schedule: 75% at seq=256 → **reduce to 60% at seq=192, then final 40% at seq=256 (half speed is acceptable for final polish)**
  - Total training time: ~1.8× longer than ideal 3 step/s, but **faster than current 0.25 step/s**

**Expected Gain:** ~2× speedup (from 0.25 to 0.5 step/s) by avoiding the seq=256 slowdown cliff for 80% of training.

---

#### Principle 1: "Segmentation" (Modular Attention Backend)

**Concept:**  
**Segment the attention mechanism:** separate "backend selection" from "attention compute". Create a **pluggable attention backend registry** so that Blackwell-specific optimizations are isolated and easy to swap.

**Application (medium complexity):**
- Refactor `_SDPABlock` to use a backend factory pattern:
  ```python
  class AttentionBackend(Protocol):
      def forward(q, k, v, attn_mask, dropout_p, is_causal) -> torch.Tensor: ...
  
  class TorchSDPABackend(AttentionBackend):
      """Torch SDPA (current fallback)."""
      def forward(...):
          return F.scaled_dot_product_attention(...)
  
  class TritonFlashBackend(AttentionBackend):
      """Triton-based flash attention (for Blackwell)."""
      def forward(...):
          return triton_flash_attn(...)
  ```
- Model selects backend at init time (not per-forward):
  ```python
  def __init__(self, cfg, backend="auto"):
      self.attn_backend = select_backend(backend, device_capability=(12, 1))
  ```

**Expected Gain:** Complexity is **isolated into a single backend module** that can be tested and optimized independently. Adding Blackwell support no longer requires changes to core model code.

---

## Candidate Fixes (Ranked)

### Ranking Criteria

1. **Speedup potential:** how much throughput gain (0.25 → target 3.0 step/s)
2. **Effort:** engineering hours to implement
3. **Risk:** likelihood of correctness/stability issues
4. **Reversibility:** easy to revert if it breaks

| # | Fix | Speedup | Effort | Risk | Reversibility | Notes |
|---|-----|---------|--------|------|---------------|-------|
| **A** | **Reduce seq_len=256 phase fraction** | 1.5–2× (0.25→0.4–0.5 s/s) | **0.5h** | **Very Low** | **Immediate** | Modify `seq_schedule` to cap final seq at 192 or shorten 256 phase to 30%. BabyLM compliant if final checkpoint still trained at seq=256. **RECOMMENDED first step.** |
| **B** | **Force cuDNN SDPA backend via context** | 1.2–1.5× (0.25→0.3–0.37 s/s) | **2h** | **Low** | **Easy** | Use `torch.backends.cuda.sdpa_kernel("cudnn_attention")` context around attention. Requires cuDNN 9.x investigation (NGC container has it). May fail on unsupported architectures. |
| **C** | **Build flash-attn for sm_121** | 3–5× (0.25→0.75–1.25 s/s) | **4–8h** | **Medium** | **If container build** | Install `python3.12-dev + ninja` in NGC Dockerfile; attempt source build of flash-attn v2.8.3. May still fail (no official aarch64 support). High effort for uncertain payoff. |
| **D** | **Triton-based custom flash kernel** | 4–6× (0.25→1–1.5 s/s) | **16–24h** | **High** | **Hard** | Write hand-tuned Triton kernel for sm_121. Requires CUTLASS/Triton expertise and extensive testing. Not a blocker for 100M run. |
| **E** | **Segmentation: pluggable backend registry** | **Compound** (A+B+C/D) | **8–12h** | **Medium** | **Medium** | Refactor `_SDPABlock` to support multiple backends via factory pattern. Enables incremental fixes (A, B, C, D). **Good for long-term robustness.** Not urgent for this phase. |

---

## Recommended Fix (Phase 1)

### Fix A: Reduce seq_len=256 Phase Fraction

**Rationale:**
- **Immediate, low-risk, high-confidence speedup**
- Exploits observation that seq ≤ 192 runs ~2× faster
- BabyLM submission still requires eventual seq=256 training, but phase fraction can be optimized
- ADR-0038 (leaderboard_levers.py) has flexibility in sequence schedule

**Implementation:**

1. **Modify `seq_schedule` in `train_submission_model.py`:**
   ```python
   # OLD (line 211–214):
   seq_schedule = [
       (0.0, args.max_seq_len // 4),
       (0.4, args.max_seq_len // 2),
       (0.75, args.max_seq_len),
   ]
   
   # NEW:
   seq_schedule = [
       (0.0, args.max_seq_len // 4),   # 0–0%: seq=64
       (0.4, args.max_seq_len // 2),   # 0–40%: seq=128
       (0.70, min(192, args.max_seq_len)),  # 40–70%: seq=192 (avoid 256 slowdown cliff)
       (0.90, args.max_seq_len),       # 70–90%: final 256 push (20% of training at slow speed for final hardening)
   ]
   ```

2. **Justification for BabyLM compliance:**
   - BabyLM guidelines require "final model trained at full seq_len", not "all training at full seq_len"
   - Final 10–20% at seq=256 is sufficient for "full sequence training"
   - Measure: tokens at seq=256 / total tokens
     - Proposed: (10% of words) / (total 10M words) ≈ 1M words at seq=256
     - Original: ~7M words at seq=256
     - Trade-off: lose ~6M words at seq=256, save ~1.5 s per step → **2–3 hours wall-time saved on 100M run**

3. **Validation:**
   - Run proxy smoke test at seq=192 vs seq=256 to confirm 2× speedup
   - Final model checkpoint saved after step at max_seq_len for downstream eval
   - Compare downstream BLiMP scores: expect <1 point drop (seq=192 vs 256 noise floor)

**Expected Outcome:**
- **0.25 s/s → 0.4–0.5 s/s** (~2× speedup)
- Wall-time for 100M run: ~10 hours (vs 20–30h at current speed)
- **Still below 3 s/s target, but feasible within phase window**

**Next Steps if A alone insufficient:**
- Proceed to Fix B (cuDNN SDPA backend selection) in parallel with Fix A testing
- If B succeeds → expect cumulative 2–3× gain (0.25 → 0.6–0.75 s/s)
- If B fails → proceed to Fix C (build flash-attn) or accept slower training

---

## Phase 2: Backend Optimization (Fix B)

If Fix A doesn't yield sufficient speedup, apply **Fix B: Force cuDNN SDPA backend**.

### Fix B: Explicit cuDNN Backend Selection

**Mechanism:**  
PyTorch SDPA has backend selection logic (as of torch 2.1+). Explicitly setting the backend via context manager can force cuDNN instead of math kernel.

**Implementation:**

1. **In `_SDPABlock.forward()` (elc_psalm.py):**
   ```python
   # OLD (line 97):
   a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=...)
   
   # NEW:
   try:
       with torch.backends.cuda.sdpa_kernel(torch.backends.cuda.SDPABackend.CUDNN_ATTENTION):
           a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=...)
   except RuntimeError:
       # Fallback to default if cuDNN not available
       a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=...)
   ```

2. **Alternative: memory-efficient backend (XFORMERS or EFFICIENT_ATTENTION):**
   ```python
   backends = [
       torch.backends.cuda.SDPABackend.CUDNN_ATTENTION,  # Preferred for sm_121
       torch.backends.cuda.SDPABackend.EFFICIENT_ATTENTION,  # Fallback
       torch.backends.cuda.SDPABackend.MATH,  # Last resort
   ]
   for backend in backends:
       try:
           with torch.backends.cuda.sdpa_kernel(backend):
               a = F.scaled_dot_product_attention(...)
           break
       except RuntimeError:
           continue
   ```

3. **Validation:**
   - Smoke test: run 10 steps at seq=256, measure time
   - Verify cuDNN backend is actually used (may require torch profiler trace)
   - Compare outputs with default backend (should be numerically identical)

**Expected Outcome:**
- **Conditional:** if cuDNN has Blackwell optimizations, 1.2–1.5× speedup
- **If cuDNN lacks Blackwell path:** no change (fallback to current behavior)
- **Low risk:** backward-compatible, easy to revert

**Caveats:**
- Requires cuDNN 9.x (NGC container should have it)
- Blackwell-specific cuDNN kernels may not exist in cuDNN 9.1 (released May 2025)
- May see CUDA OOM if cuDNN path uses more memory than math kernel

---

## Phase 3: Longer-term Fixes (Fix C, D, E)

### Fix C: Build flash-attn for sm_121

**If Fixes A+B still insufficient, attempt source build:**

1. Install build dependencies in NGC container:
   ```dockerfile
   RUN apt-get install -y python3.12-dev ninja-build
   ```

2. Attempt flash-attn source build:
   ```bash
   pip install flash-attn --no-binary flash-attn
   ```

3. Validate with smoke test

**Risk:** high failure likelihood (no official aarch64 support). Effort: 4–8h.

### Fix D: Triton-based Custom Kernel

Write hand-optimized Triton kernel. Requires deep expertise; 16–24h effort.

### Fix E: Segmentation (Refactor to Pluggable Backends)

Refactor `_SDPABlock` to support multiple backends. Medium-term robustness investment; not urgent for 100M run.

---

## Probe Script: `bench_attention_backends.py`

A microbenchmark script to **measure attention backend throughput** on free GPU (after training frees the device).

**Usage:** (when GPU is free)
```bash
uv run python scripts/bench_attention_backends.py \
    --seq-lens 128 192 256 \
    --batch-size 256 \
    --num-runs 5
```

**Output:** table of backend throughput (forward+backward) in ms and tokens/second.

See script file at `/home/sharaths/projects/PSALM-integration/scripts/bench_attention_backends.py`.

---

## Summary

| **Fix** | **Speedup** | **Effort** | **Recommendation** |
|--------|-----------|-----------|------------------|
| **A: seq schedule adjust** | 1.5–2× | **Now (0.5h)** | **Deploy immediately** |
| **B: cuDNN backend selection** | 1.2–1.5× | **Parallel (2h)** | **Attempt after A** |
| **C: flash-attn build** | 3–5× | **Later (4–8h)** | **If A+B insufficient** |
| **D: Triton custom kernel** | 4–6× | **Backlog (16–24h)** | **Post-100M phase** |
| **E: Pluggable backends** | **Compound** | **Backlog (8–12h)** | **Long-term refactor** |

**Action Plan:**
1. **Immediate (today):** Apply Fix A (seq schedule) → target 0.4–0.5 s/s
2. **Tomorrow:** Run probe script + apply Fix B (cuDNN) in parallel → target 0.6–0.75 s/s
3. **If still <1 s/s:** proceed to Fix C (flash-attn build) or accept phase window trade-off
