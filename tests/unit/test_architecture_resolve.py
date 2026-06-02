"""Architecture name resolution for competition YAML."""

from __future__ import annotations

import pytest

from psalm.config.architecture import is_elc_architecture, resolve_architecture
from psalm.domain.model.elc_config import STRICT_SMALL_VOCAB, STRICT_VOCAB, ElcPsalmConfig


def test_elc_architecture_names_resolve() -> None:
    s = resolve_architecture("elc_psalm_s", vocab_size=16_000)
    m = resolve_architecture("elc_psalm_m", vocab_size=24_000)
    assert isinstance(s, ElcPsalmConfig)
    assert isinstance(m, ElcPsalmConfig)
    assert s.vocab_size == 16_000
    assert m.vocab_size == 24_000
    assert s.total_params < m.total_params


def test_elc_defaults_match_joint_vocab() -> None:
    s = resolve_architecture("elc_psalm_s", vocab_size=STRICT_SMALL_VOCAB)
    m = resolve_architecture("elc_psalm_m", vocab_size=STRICT_VOCAB)
    assert s.vocab_size == STRICT_SMALL_VOCAB
    assert m.vocab_size == STRICT_VOCAB


def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        resolve_architecture("psalm-proxy-60m", vocab_size=32_000)


def test_is_elc_architecture() -> None:
    assert is_elc_architecture("elc_psalm_s")
    assert not is_elc_architecture("decoder-60m")
