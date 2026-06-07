"""GeGLU FFN + RMSNorm wiring tests (LTG/ELC-BERT architecture levers).

Guards against the dead-flag regression: ffn_type/norm_type must reach the model
and change the actual modules, not just the config.
"""

from __future__ import annotations

import torch

from psalm.domain.model.elc_config import ElcPsalmConfig
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, _AttnMode, _GeGLU


def _cfg(**kw: object) -> ElcPsalmConfig:
    base: dict[str, object] = {
        "vocab_size": 512,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "max_seq_len": 32,
    }
    base.update(kw)
    return ElcPsalmConfig(**base)  # type: ignore[arg-type]


def test_default_is_layernorm_gelu() -> None:
    m = ElcPsalmEncoder(_cfg())
    assert type(m.blocks[0].ln1).__name__ == "LayerNorm"
    assert isinstance(m.blocks[0].mlp, torch.nn.Sequential)
    assert type(m.ln_f).__name__ == "LayerNorm"


def test_rmsnorm_applied() -> None:
    m = ElcPsalmEncoder(_cfg(norm_type="rmsnorm"))
    assert type(m.blocks[0].ln1).__name__ == "RMSNorm"
    assert type(m.blocks[0].ln2).__name__ == "RMSNorm"
    assert type(m.ln_f).__name__ == "RMSNorm"


def test_geglu_applied() -> None:
    m = ElcPsalmEncoder(_cfg(ffn_type="geglu"))
    assert isinstance(m.blocks[0].mlp, _GeGLU)


def test_geglu_rmsnorm_forward_runs() -> None:
    cfg = _cfg(ffn_type="geglu", norm_type="rmsnorm", pos_encoding="rope")
    m = ElcPsalmEncoder(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    h = m.encode(x, attn_mode=_AttnMode.BIDIRECTIONAL)
    logits = m.lm_head(h)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_geglu_param_count_comparable_to_gelu() -> None:
    """GeGLU hidden is scaled by ~2/3 so params stay comparable to plain GELU."""
    gelu = ElcPsalmEncoder(_cfg()).num_parameters()
    geglu = ElcPsalmEncoder(_cfg(ffn_type="geglu")).num_parameters()
    assert abs(geglu - gelu) / gelu < 0.10  # within 10%
