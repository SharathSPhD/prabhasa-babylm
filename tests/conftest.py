"""Shared pytest fixtures for PSALM tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def rng_seed() -> int:
    return 12345
