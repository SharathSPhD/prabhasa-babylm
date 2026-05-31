# 7. NGC Blackwell / aarch64 / CUDA-13 software stack

Date: 2026-05-31
Status: Accepted

## Context

PSALM reuses pramana's tooling, which pinned `nvcr.io/nvidia/pytorch:24.09-py3`.
That image predates the GB10 Grace-Blackwell GPU (sm_121), arm64 wheels, and CUDA
13. Unsloth and flash-attention are the highest-risk dependencies on this stack.

## Decision

Upgrade the container base to a 25.x NGC PyTorch image with Blackwell + aarch64 +
CUDA 13 support (`ARG NGC_PYTORCH_TAG=25.04-py3`, verified against the host
driver before long runs). Use `uv` for the Python environment. De-risk Unsloth +
flash-attn builds on sm_121 in Phase 0/1 before committing compute; if a
dependency cannot build, record a fallback (e.g. native PyTorch SDPA, TRL without
Unsloth) in a superseding ADR.

## Consequences

Modern kernels and arm64 support, at the cost of an early integration risk that
must be retired in Phase 1. The exact NGC tag is a config knob, not hard-coded.
