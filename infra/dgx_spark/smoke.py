#!/usr/bin/env python3
"""GB10 DGX Spark stack smoke tests (ADR-0022).

Runs environment inventory and component checks 1–5 from the U1 validation gate.
Exits 0 when all hard requirements pass; exits 1 on hard failure. Flash-attn
failure is a documented FALLBACK (exit 0) unless --strict-flash-attn is set.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["PASS", "FALLBACK", "FAIL", "SKIP"]

HARD_FAIL = 1
OK = 0


@dataclass
class CheckResult:
    component: str
    status: Status
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)
    hard_failed: bool = False

    def add(self, component: str, status: Status, detail: str = "") -> None:
        self.results.append(CheckResult(component, status, detail))
        if status == "FAIL":
            self.hard_failed = True

    def print_summary(self) -> None:
        print("\n=== GB10 STACK SMOKE SUMMARY ===")
        for r in self.results:
            print(f"  {r.component:24} {r.status:8} {r.detail}")
        print("================================\n")


def _run_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=30,
        ).strip()
        return out or "nvidia-smi ok"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"


def check_env_inventory(report: Report) -> None:
    arch = platform.machine()
    smi = _run_nvidia_smi()
    report.add("host_arch", "PASS" if arch == "aarch64" else "FAIL", arch)
    report.add("nvidia_smi", "PASS" if "unavailable" not in smi else "FAIL", smi)
    report.add("python", "PASS", sys.version.split()[0])


def check_torch_gpu(report: Report) -> None:
    try:
        import torch
    except ImportError as exc:
        report.add("torch", "FAIL", str(exc))
        return

    cuda_build = torch.version.cuda or "none"
    cuda_ok = torch.cuda.is_available()
    detail = f"torch={torch.__version__} cuda_build={cuda_build} cuda_available={cuda_ok}"
    if not cuda_ok:
        report.add("torch", "FAIL", detail)
        return
    report.add("torch", "PASS", detail)

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    report.add("gpu_device", "PASS", f"{name} cc={cap[0]}.{cap[1]}")

    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    torch.cuda.synchronize()
    _ = a @ b
    torch.cuda.synchronize()
    report.add("torch_matmul", "PASS", "256x256 fp32")

    bf16 = torch.cuda.is_bf16_supported()
    if not bf16:
        report.add("torch_bf16", "FAIL", "bf16 not supported")
        return
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    _ = x @ x
    torch.cuda.synchronize()
    report.add("torch_bf16", "PASS", "matmul ok")

    q = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=True
    )
    out.sum().backward()
    torch.cuda.synchronize()
    report.add("torch_sdpa", "PASS", "fwd+bwd causal bf16")


def check_flash_attn(report: Report, *, strict: bool) -> None:
    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_func
        import torch

        B, H, S, D = 2, 4, 128, 64
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = flash_attn_func(q, k, v, causal=True)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        report.add("flash_attn", "PASS", f"fwd+bwd {ms:.1f}ms")
    except ImportError as exc:
        status: Status = "FAIL" if strict else "FALLBACK"
        report.add("flash_attn", status, f"import/build: {exc}")
        if strict:
            report.hard_failed = True
    except Exception as exc:
        status = "FAIL" if strict else "FALLBACK"
        report.add("flash_attn", status, f"runtime: {exc}")
        if strict:
            report.hard_failed = True


def bench_attention(report: Report) -> None:
    """Optional microbench when flash-attn is installed."""
    try:
        import torch
        from flash_attn import flash_attn_func
    except ImportError:
        return

    import torch

    B, H, S, D = 2, 4, 128, 64

    def sdpa_ms() -> float:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        out.sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    def fa_ms() -> float:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = flash_attn_func(q, k, v, causal=True)
        out.sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    report.add("bench_sdpa_ms", "PASS", f"{sdpa_ms():.1f}")
    report.add("bench_flash_attn_ms", "PASS", f"{fa_ms():.1f}")


def check_unsloth(report: Report) -> None:
    try:
        from unsloth import FastLanguageModel  # noqa: F401

        report.add("unsloth", "PASS", "FastLanguageModel import ok (no full LoRA step)")
    except ImportError as exc:
        report.add("unsloth", "FAIL", str(exc))
    except Exception as exc:
        report.add("unsloth", "FALLBACK", f"import warning: {exc}")


def check_vidyut(report: Report, *, min_sentences: int) -> None:
    try:
        import vidyut

        ver = getattr(vidyut, "__version__", "unknown")
    except ImportError as exc:
        report.add("vidyut", "FAIL", str(exc))
        return

    try:
        from psalm.infrastructure.generators.vidyut_source import VidyutGenerator

        items = list(VidyutGenerator().stream(min_sentences, seed=0))
        if len(items) < min_sentences:
            report.add("vidyut", "FAIL", f"only {len(items)} sentences")
            return
        sample = items[0].text[:48]
        deriv = len(items[0].derivation)
        report.add("vidyut", "PASS", f"v{ver} n={len(items)} sample={sample!r} deriv_steps={deriv}")
    except Exception as exc:
        report.add("vidyut", "FAIL", str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="GB10 stack smoke tests")
    parser.add_argument(
        "--strict-flash-attn",
        action="store_true",
        help="Treat missing flash-attn as hard failure",
    )
    parser.add_argument(
        "--vidyut-min",
        type=int,
        default=100,
        help="Minimum Vidyut sentences for smoke (default 100)",
    )
    parser.add_argument("--bench", action="store_true", help="Run flash vs SDPA microbench")
    args = parser.parse_args()

    report = Report()
    print("=== GB10 ENV INVENTORY ===")
    print(f"uname -m: {platform.machine()}")
    print(f"nvidia-smi: {_run_nvidia_smi()}")
    print(f"python: {sys.version.split()[0]}")

    check_env_inventory(report)
    check_torch_gpu(report)
    check_flash_attn(report, strict=args.strict_flash_attn)
    if args.bench:
        bench_attention(report)
    check_unsloth(report)
    check_vidyut(report, min_sentences=args.vidyut_min)

    report.print_summary()
    return HARD_FAIL if report.hard_failed else OK


if __name__ == "__main__":
    raise SystemExit(main())
