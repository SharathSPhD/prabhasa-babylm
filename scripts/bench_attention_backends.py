#!/usr/bin/env python3
"""Microbench attention backends on Blackwell GB10 sm_121.

This script measures forward+backward pass throughput of torch.nn.functional
.scaled_dot_product_attention across different SDPA backends (math, cudnn,
memory_efficient, flash if available) at various sequence lengths.

RUN THIS WHEN GPU IS FREE (after training run completes).

    uv run python scripts/bench_attention_backends.py \
        --seq-lens 128 192 256 \
        --batch-size 256 \
        --num-runs 5 \
        --output bench_results.json

Output: JSON with timings and throughput (tokens/second) for each backend x seq_len.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class BenchResult:
    """Single benchmark result: one backend, seq_len, batch_size combination."""

    backend: str
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int
    run_idx: int
    fwd_ms: float
    bwd_ms: float
    total_ms: float
    tokens_per_sec: float

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def gflops(self) -> float:
        """Rough FLOP count for attention: 2 * batch * seq^2 * head_dim * num_heads."""
        # Q @ K: batch * num_heads * seq * seq * head_dim
        qk_flops = 2 * self.batch_size * self.num_heads * (self.seq_len ** 2) * self.head_dim
        # Softmax @ V: batch * num_heads * seq * seq * head_dim
        sv_flops = 2 * self.batch_size * self.num_heads * (self.seq_len ** 2) * self.head_dim
        # Gradient wrt Q, K, V: 3x forward
        total = qk_flops + sv_flops + 3 * (qk_flops + sv_flops)
        return total / 1e9


def try_backend(
    backend_name: Literal["math", "cudnn_attention", "efficient_attention", "flash"],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, bool]:
    """Attempt to run SDPA with a specific backend.

    Returns (output, success).
    """
    try:
        if backend_name == "math":
            # Default (usually math kernel)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal
            )
        elif backend_name == "cudnn_attention":
            with torch.backends.cuda.sdpa_kernel(
                torch.backends.cuda.SDPABackend.CUDNN_ATTENTION
            ):
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=is_causal
                )
        elif backend_name == "efficient_attention":
            with torch.backends.cuda.sdpa_kernel(
                torch.backends.cuda.SDPABackend.EFFICIENT_ATTENTION
            ):
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=is_causal
                )
        elif backend_name == "flash":
            # Requires flash-attn installed
            from flash_attn import flash_attn_func

            # Flash expects (batch, seq, num_heads, head_dim) but we have (batch, num_heads, seq, head_dim)
            # Convert: (b, h, s, d) -> (b, s, h, d)
            b, h, s, d = q.shape
            q_rearr = q.transpose(1, 2).contiguous()
            k_rearr = k.transpose(1, 2).contiguous()
            v_rearr = v.transpose(1, 2).contiguous()
            out_rearr = flash_attn_func(
                q_rearr, k_rearr, v_rearr, causal=is_causal, dropout_p=0.0
            )
            # Convert back: (b, s, h, d) -> (b, h, s, d)
            out = out_rearr.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
        return out, True
    except (RuntimeError, ImportError, AttributeError) as e:
        # Backend not available on this device/pytorch version
        return None, False


def benchmark_backend(
    backend_name: str,
    seq_len: int,
    batch_size: int,
    num_heads: int = 12,
    head_dim: int = 64,
    is_causal: bool = True,
    num_runs: int = 5,
    device: str = "cuda",
) -> list[BenchResult] | None:
    """Run forward+backward benchmark for one backend x seq_len combination.

    Returns list of BenchResult for each run, or None if backend unavailable.
    """
    # Allocate tensors
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    # Warmup run
    try_backend(backend_name, q, k, v, is_causal)
    torch.cuda.synchronize()

    results = []
    for run_idx in range(num_runs):
        # Fresh tensors for each run
        q = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward
        out, success = try_backend(backend_name, q, k, v, is_causal)
        if not success:
            return None

        loss = out.sum()
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # Backward
        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t1

        total_ms = (t_fwd + t_bwd) * 1000
        tokens_per_sec = (batch_size * seq_len) / (total_ms / 1000)

        results.append(
            BenchResult(
                backend=backend_name,
                seq_len=seq_len,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                run_idx=run_idx,
                fwd_ms=t_fwd * 1000,
                bwd_ms=t_bwd * 1000,
                total_ms=total_ms,
                tokens_per_sec=tokens_per_sec,
            )
        )

    return results


def print_results_table(all_results: list[BenchResult]) -> None:
    """Print results as a readable table."""
    print("\n" + "=" * 120)
    print("ATTENTION BACKEND BENCHMARK RESULTS")
    print("=" * 120)
    print(
        f"{'Backend':<20} {'Seq Len':>8} {'Batch':>6} {'Fwd (ms)':>10} {'Bwd (ms)':>10} "
        f"{'Total (ms)':>12} {'Tokens/sec':>12}"
    )
    print("-" * 120)

    current_backend = None
    for r in sorted(all_results, key=lambda x: (x.backend, x.seq_len, x.run_idx)):
        if r.backend != current_backend:
            current_backend = r.backend
            print()

        print(
            f"{r.backend:<20} {r.seq_len:>8} {r.batch_size:>6} "
            f"{r.fwd_ms:>10.2f} {r.bwd_ms:>10.2f} {r.total_ms:>12.2f} {r.tokens_per_sec:>12.0f}"
        )

    print("=" * 120)


def print_summary_stats(all_results: list[BenchResult]) -> None:
    """Print summary statistics per backend and seq_len."""
    print("\nSUMMARY STATISTICS (mean ± std over runs)")
    print("-" * 100)

    backends = sorted(set(r.backend for r in all_results))
    seq_lens = sorted(set(r.seq_len for r in all_results))

    for seq_len in seq_lens:
        print(f"\nSeq Len = {seq_len}")
        print(f"  {'Backend':<20} {'Mean (ms)':>12} {'Std (ms)':>12} {'Tokens/sec':>12}")
        for backend in backends:
            rows = [r for r in all_results if r.seq_len == seq_len and r.backend == backend]
            if not rows:
                continue
            times_ms = [r.total_ms for r in rows]
            toks_sec = [r.tokens_per_sec for r in rows]
            mean_time = sum(times_ms) / len(times_ms)
            std_time = (sum((t - mean_time) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
            mean_tok = sum(toks_sec) / len(toks_sec)
            print(f"  {backend:<20} {mean_time:>12.2f} {std_time:>12.2f} {mean_tok:>12.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark attention backends on Blackwell sm_121"
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 192, 256],
        help="Sequence lengths to benchmark (default: 128 192 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for benchmark (default: 256)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per config (default: 5)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Number of attention heads (default: 12)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench_attention_results.json"),
        help="Output JSON file (default: bench_attention_results.json)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["math", "cudnn_attention", "efficient_attention", "flash"],
        help="Backends to test (default: all)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; cannot benchmark on GPU.")
        return 1

    device = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {device}")
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
    print(f"Torch Version: {torch.__version__}")

    torch.set_float32_matmul_precision("high")

    all_results: list[BenchResult] = []
    for backend in args.backends:
        for seq_len in args.seq_lens:
            print(f"\nBenchmarking {backend} at seq_len={seq_len}...")
            results = benchmark_backend(
                backend,
                seq_len=seq_len,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                num_runs=args.num_runs,
            )
            if results is None:
                print(f"  {backend} not available on this device/PyTorch version")
            else:
                all_results.extend(results)
                mean_time = sum(r.total_ms for r in results) / len(results)
                print(f"  Mean: {mean_time:.2f} ms")

    # Print results
    print_results_table(all_results)
    print_summary_stats(all_results)

    # Save to JSON
    output_dict = {
        "device": device,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "seq_lens": args.seq_lens,
        "num_runs": args.num_runs,
        "results": [asdict(r) for r in all_results],
    }
    args.output.write_text(json.dumps(output_dict, indent=2))
    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
