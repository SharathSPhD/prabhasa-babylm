"""Tests for GPU-only device resolution (ADR-0035: no silent CUDA->CPU fallback).

These patch the ``_cuda_available`` probe so they run without torch or a GPU.
"""

from __future__ import annotations

import pytest

from psalm.infrastructure.ml import device as dev
from psalm.infrastructure.ml.device import (
    CudaUnavailableError,
    require_cuda_device,
    resolve_training_device,
)


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, "_cuda_available", lambda: False)


@pytest.fixture
def with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, "_cuda_available", lambda: True)


class TestRequireCuda:
    def test_returns_cuda_when_available(self, with_cuda: None) -> None:
        assert require_cuda_device() == "cuda"

    def test_aborts_when_unavailable(self, no_cuda: None) -> None:
        with pytest.raises(CudaUnavailableError):
            require_cuda_device()


class TestResolveTrainingDevice:
    def test_cuda_passes_through_when_available(self, with_cuda: None) -> None:
        assert resolve_training_device("cuda") == "cuda"
        assert resolve_training_device("cuda:1") == "cuda:1"

    def test_cuda_never_silently_downgrades(self, no_cuda: None) -> None:
        with pytest.raises(CudaUnavailableError):
            resolve_training_device("cuda")

    def test_explicit_cpu_is_allowed(self, no_cuda: None) -> None:
        # An explicit CPU request is an intentional proxy/CI run, not a fallback.
        assert resolve_training_device("cpu") == "cpu"

    def test_unknown_device_rejected(self, with_cuda: None) -> None:
        with pytest.raises(ValueError, match="unsupported device"):
            resolve_training_device("tpu")
