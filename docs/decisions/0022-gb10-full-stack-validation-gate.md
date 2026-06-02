# ADR-0022 — GB10 full-stack validation gate (blocking infra)

- Status: Accepted
- Date: 2026-06-02
- Supersedes scope of ADR-0007 (partial stack notes) with an explicit acceptance checklist
- Unit: U1 (Wave 1)

## Context

All training runs target a single **NVIDIA DGX Spark GB10** (Grace-Blackwell, aarch64,
sm_121, CUDA 13, ~128GB). ADR-0007 selected NGC PyTorch; CLAUDE.md flags Unsloth and
flash-attn on sm_121 as high risk. The repo lacks `infra/dgx_spark/Dockerfile.verified`
and documented proof that **Vidyut** (Rust wheel), **torch**, **flash-attn**, and
**Unsloth** build and run on aarch64/sm_121.

Human direction (2026-06): **full** stack validation is a **blocking gate** before
Wave-2 training charters and before competition GPU spend.

## Decision

1. **Gate artifact:** commit `infra/dgx_spark/Dockerfile.verified` plus
   `infra/dgx_spark/build-validation.sh` log output archived under
   `docs/infra/gb10-validation-2026-06.md` (created by U1 execution, not this ADR).

2. **Minimum acceptance (all required on GB10 hardware):**
   - NGC PyTorch import; CUDA device visible; matmul on GPU.
   - `flash-attn` import and one forward pass (or documented waiver ADR if upstream
     blocks sm_121 — waiver requires human sign-off).
   - Unsloth training hook smoke test (minimal LoRA step).
   - Vidyut wheel install + `vidyut_source` generator smoke (≥100 sentences).
   - `make gate` inside container.

3. **Wave-1 precondition:** U2–U7 GPU-touching worktrees must not start until U1
   reports `GB10_STACK_VERIFIED` in the foundation index checklist.

4. **Fallback policy:** if flash-attn fails, train with vanilla attention for proxy
   runs only after waiver ADR; competition submission requires flash-attn or
   equivalent kernel documented in model card.

## Consequences

- Delays training but prevents mid-battery stack failures.
- Dockerfile is the single source of truth for CI/gpu markers.
- Planning branch documents the gate; implementation is U1 on `data-engine` worktree.

## Alternatives considered

- **Defer validation to first training run:** rejected — wastes H1′ battery time.
- **x86 cloud development:** rejected for primary path — program is GB10-only per ADR-0005.
- **Drop Unsloth:** rejected — consolidation architecture assumes efficient SLM training.

## Links

- ADR-0007: `docs/decisions/0007-ngc-blackwell-aarch64-stack.md`
- U1 charter (future): `docs/contracts/reframe-2026-06-foundation.md`
