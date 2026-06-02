"""BabyLM eval model factories (mock baseline vs ELC-PSALM PLL adapter)."""

from __future__ import annotations

from pathlib import Path

from psalm.benchmarks.babylm_eval import MockUniformBaseline, PseudoLogLikelihoodModel
from psalm.domain.model.elc_config import ElcPsalmConfig


def _ascii_encode(text: str, *, vocab_size: int) -> list[int]:
    """Map chars to ``[1, vocab_size - 2]`` — reserve 0 pad and ``vocab_size - 1`` mask."""
    cap = max(vocab_size - 2, 1)
    return [(ord(c) % cap) + 1 for c in text]


def tiny_elc_config(*, vocab_size: int = 128) -> ElcPsalmConfig:
    """CPU-friendly ELC config for CLI smoke / wiring tests."""
    return ElcPsalmConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        mlm_probability=0.15,
    )


def build_babylm_smoke_model(
    *,
    mock: bool = False,
    use_elc: bool = True,
    checkpoint: Path | None = None,
    seed: int = 0,
    vocab_size: int = 128,
    device: str = "cpu",
) -> PseudoLogLikelihoodModel:
    """Select eval model: ELC PLL default; checkpoint when present; mock with ``--mock``."""
    if mock:
        return MockUniformBaseline(vocab_size=vocab_size, seed=seed)

    if checkpoint is not None and checkpoint.is_file():
        loaded = _load_elc_checkpoint(checkpoint, vocab_size=vocab_size, device=device)
        if loaded is not None:
            return loaded

    if use_elc:
        return _untrained_elc_evaluator(vocab_size=vocab_size, device=device, seed=seed)

    return MockUniformBaseline(vocab_size=vocab_size, seed=seed)


def _untrained_elc_evaluator(
    *, vocab_size: int, device: str = "cpu", seed: int = 0
) -> PseudoLogLikelihoodModel:
    import torch

    from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, ElcPsalmEvaluator

    torch.manual_seed(seed)
    cfg = tiny_elc_config(vocab_size=vocab_size)
    mask_id = cfg.vocab_size - 1
    model = ElcPsalmEncoder(cfg).to(device)
    model.eval()

    def encode(text: str, *, vs: int = vocab_size) -> list[int]:
        return _ascii_encode(text, vocab_size=vs)

    return ElcPsalmEvaluator(model, encode, mask_id=mask_id, device=device)


def _load_elc_checkpoint(
    path: Path, *, vocab_size: int, device: str = "cpu"
) -> PseudoLogLikelihoodModel | None:
    """Load ``.pt`` checkpoint; return None on failure (caller falls back)."""
    try:
        from psalm.infrastructure.ml.elc_psalm import ElcPsalmEvaluator
        from psalm.infrastructure.ml.elc_trainer import load_elc_checkpoint

        model, mask_id = load_elc_checkpoint(path, device=device)
        model.eval()
        vs = model.cfg.vocab_size

        def encode(text: str, *, vocab: int = vs) -> list[int]:
            return _ascii_encode(text, vocab_size=vocab)

        return ElcPsalmEvaluator(model, encode, mask_id=mask_id, device=device)
    except Exception:
        return None
