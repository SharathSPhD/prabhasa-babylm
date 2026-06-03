"""Device resolution for the GPU-only battery (ADR-0035).

The evidence-grade battery runs on CUDA; there is **no silent CUDA→CPU fallback**.

* ``require_cuda_device`` aborts hard when CUDA is unavailable and is called at the
  top of any evidence-grade entrypoint (the battery), so a GPU-less host fails fast
  with a non-zero exit instead of quietly producing CPU numbers.
* ``resolve_training_device`` permits an *explicitly requested* CPU device (proxy /
  CI / unit runs) but never downgrades a requested CUDA device to CPU.

``torch`` is imported lazily inside ``_cuda_available`` so this module — and its
logic tests — import without the heavy ML stack or a GPU present.
"""

from __future__ import annotations


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA is required but ``torch.cuda.is_available()`` is False."""


def _cuda_available() -> bool:
    """True iff a CUDA device is visible to torch. Patch-point for tests."""
    import torch

    return bool(torch.cuda.is_available())


def require_cuda_device() -> str:
    """Return ``"cuda"`` or raise :class:`CudaUnavailableError`.

    Call this at the start of any evidence-grade (battery) run so the process
    aborts non-zero on a CPU-only host rather than silently degrading.
    """
    if not _cuda_available():
        raise CudaUnavailableError(
            "GPU-only battery requires CUDA but torch.cuda.is_available() is False "
            "(ADR-0035: no CPU fallback). Run on the GB10 GPU host."
        )
    return "cuda"


def resolve_training_device(requested: str) -> str:
    """Resolve a requested device with no silent downgrade.

    * ``"cuda"`` / ``"cuda:N"`` -> returned only if CUDA is available, else raise.
    * ``"cpu"`` -> returned as-is (explicit, intentional proxy / CI run).
    * anything else -> ``ValueError``.
    """
    req = requested.strip().lower()
    if req.startswith("cuda"):
        if not _cuda_available():
            raise CudaUnavailableError(
                f"requested device {requested!r} but CUDA is unavailable; refusing to "
                "silently fall back to CPU (ADR-0035). Use device='cpu' explicitly for "
                "a proxy run, or run on the GB10 GPU host."
            )
        return requested
    if req == "cpu":
        return "cpu"
    raise ValueError(f"unsupported device {requested!r}; use 'cuda', 'cuda:N', or 'cpu'")
