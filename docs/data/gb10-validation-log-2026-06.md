# GB10 validation log (U1, 2026-06-02)

Host: DGX Spark stand-in, `aarch64`, `workstream/gb10`.

## Host inventory

```
$ uname -m
aarch64

$ nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
name, driver_version, compute_cap
NVIDIA GB10, 580.159.03, 12.1
```

## torch (uv venv, `uv sync --extra ml`)

- `torch==2.12.0+cu130`, `cuda_available=True`
- matmul, bf16, SDPA fwd+bwd: PASS (`infra/dgx_spark/smoke.py`)

## flash-attn 2.8.3 build attempt

```
Precompiled wheel not found. Building from source...
fatal error: Python.h: No such file or directory
```

Log: `/tmp/gb10-flash-build/flash-attn-build.log`. Status: **FALLBACK** → ADR-0023.

## unsloth

- Full `pip install unsloth` downgraded torch to 2.10.0 (CUDA broken).
- Pinned `--no-deps` install: **PASS** import `FastLanguageModel`.

## vidyut

- `vidyut==0.4.0` wheel on aarch64.
- `VidyutGenerator().stream(100)`: PASS.

## pytest gate

```
215 passed, 1 skipped in 45.91s
```

## smoke.py exit code

0 (flash-attn FALLBACK non-fatal).
