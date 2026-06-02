"""Resolve YAML ``architecture`` names to domain model configs.

Decoder presets use ``preset_for``; competition ELC-PSALM backbones use
``elc_preset_for``. Training-loop construction from these configs is a follow-up.
"""

from __future__ import annotations

from psalm.domain.model.config import ModelConfig
from psalm.domain.model.elc_config import ElcPsalmConfig, elc_preset_for

ELC_ARCHITECTURE_NAMES: frozenset[str] = frozenset({"elc_psalm_s", "elc_psalm_m"})


def resolve_architecture(
    name: str,
    *,
    vocab_size: int,
    max_seq_len: int = 512,
) -> ElcPsalmConfig | ModelConfig:
    """Map a config ``architecture`` string to a validated architecture model."""
    if name == "elc_psalm_s":
        return elc_preset_for("S", vocab_size=vocab_size, max_seq_len=max_seq_len)
    if name == "elc_psalm_m":
        return elc_preset_for("M", vocab_size=vocab_size, max_seq_len=max_seq_len)
    raise ValueError(f"unknown architecture {name!r}; ELC names: {sorted(ELC_ARCHITECTURE_NAMES)}")


def is_elc_architecture(name: str) -> bool:
    return name in ELC_ARCHITECTURE_NAMES
