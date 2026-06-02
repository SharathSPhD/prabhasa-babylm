# ADR-0023 — flash-attn on GB10 sm_121: torch SDPA fallback for proxy training

- Status: Accepted (build-time decision per human direction 2026-06-02)
- Date: 2026-06-02
- Depends on: ADR-0022, ADR-0007
- Unit: U1 (`workstream/gb10`)

## Context

U1 validation on the live DGX Spark GB10 host (`aarch64`, NVIDIA GB10, compute
capability 12.1, driver 580.x, CUDA 13) confirmed **torch 2.12.0+cu130** with GPU
matmul, bf16, and **torch SDPA** forward+backward. **flash-attn 2.8.3** did not
install in the host `uv` venv: no precompiled `linux_aarch64` wheel for
`cu130`+`cp312`, and source build failed immediately without `python3.12-dev`
(`Python.h: No such file or directory`). Container builds (`Dockerfile.verified`)
install `python3-dev` + `ninja` so flash-attn may succeed there; that remains
optional follow-up (`RUN_DOCKER=1`).

## Decision

1. **Proxy / H1′ training (60M–150M batteries):** use **PyTorch scaled dot-product
   attention (SDPA)** as the default attention backend. Do not block Wave-2
   generator or data work on flash-attn.

2. **flash-attn policy:** treat as **build-time optional** in
   `infra/dgx_spark/Dockerfile.verified`. `smoke.py` reports `FALLBACK` when
   import fails; `--strict-flash-attn` only for explicit re-validation.

3. **Competition / publication path:** before any competition submission, either
   (a) flash-attn builds and passes smoke inside `Dockerfile.verified`, or
   (b) model card documents equivalent kernel (e.g. cuDNN SDPA / Transformer
   Engine) with measured throughput — human sign-off required per ADR-0022.

4. **Unsloth:** import verified with **pinned companion wheels** and
   `--no-deps` on `unsloth` so **torch is not downgraded** (full `pip install
   unsloth` replaced `torch 2.12+cu130` with `2.10` and broke CUDA). Container
   and host docs list required packages: `unsloth`, `unsloth-zoo`, `bitsandbytes`,
   `cut-cross-entropy`, `tyro`, `docstring-parser`, `msgspec`, `hf-transfer`.

## Consequences

- `GB10_STACK_VERIFIED` may be declared for torch + Vidyut + Unsloth import with
  flash-attn = FALLBACK.
- Integration should add an optional `gb10` extra or lockfile fragment; do not
  add flash-attn to default `ml` extra until aarch64 wheels or reliable container
  build is routine.
- Re-attempt flash-attn after `python3.12-dev` + `ninja-build` on host or inside
  NGC image build.

## Alternatives considered

- **Block all GPU training until flash-attn passes:** rejected — SDPA is sufficient
  for proxy gates; delays H1′ without empirical benefit on 60M runs.
- **Pin flash-attn in pyproject `ml` extra now:** rejected — breaks aarch64 CI and
  x86 dev machines; keep infra-pinned in `Dockerfile.verified` only.
