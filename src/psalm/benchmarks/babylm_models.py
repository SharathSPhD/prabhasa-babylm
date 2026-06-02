"""BabyLM eval model factories (mock baseline vs ELC-PSALM PLL adapter)."""

from __future__ import annotations

from pathlib import Path

from psalm.benchmarks.babylm_eval import MockUniformBaseline, PseudoLogLikelihoodModel
from psalm.domain.model.elc_config import ElcPsalmConfig


def _ascii_encode(text: str, *, vocab_size: int) -> list[int]:
    return [(ord(c) % max(vocab_size - 2, 1)) + 1 for c in text]


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
    use_elc: bool = False,
    checkpoint: Path | None = None,
    seed: int = 0,
    vocab_size: int = 128,
) -> PseudoLogLikelihoodModel:
    """Select eval model: mock default; ELC untrained when ``use_elc``; checkpoint if present."""
    if mock:
        return MockUniformBaseline(vocab_size=vocab_size, seed=seed)

    if checkpoint is not None and checkpoint.is_file():
        loaded = _load_elc_checkpoint(checkpoint, vocab_size=vocab_size)
        if loaded is not None:
            return loaded

    if use_elc:
        return _untrained_elc_evaluator(vocab_size=vocab_size)

    return MockUniformBaseline(vocab_size=vocab_size, seed=seed)


def _untrained_elc_evaluator(*, vocab_size: int) -> PseudoLogLikelihoodModel:
    from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, ElcPsalmEvaluator

    cfg = tiny_elc_config(vocab_size=vocab_size)
    mask_id = cfg.vocab_size - 1
    model = ElcPsalmEncoder(cfg)
    model.eval()

    def encode(text: str, *, vs: int = vocab_size) -> list[int]:
        return _ascii_encode(text, vocab_size=vs)

    return ElcPsalmEvaluator(model, encode, mask_id=mask_id, device="cpu")


def _load_elc_checkpoint(path: Path, *, vocab_size: int) -> PseudoLogLikelihoodModel | None:
    """Load ``state_dict`` from checkpoint; return None on failure (caller falls back)."""
    try:
        torch = __import__("torch")
        from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, ElcPsalmEvaluator

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "cfg" in payload:
            cfg = ElcPsalmConfig.model_validate(payload["cfg"])
        else:
            cfg = tiny_elc_config(vocab_size=vocab_size)
        model = ElcPsalmEncoder(cfg)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
        model.eval()
        mask_id = (
            int(payload.get("mask_id", cfg.vocab_size - 1))
            if isinstance(payload, dict)
            else cfg.vocab_size - 1
        )
        vs = cfg.vocab_size

        def encode(text: str, *, vocab: int = vs) -> list[int]:
            return _ascii_encode(text, vocab_size=vocab)

        return ElcPsalmEvaluator(model, encode, mask_id=mask_id, device="cpu")
    except Exception:
        return None
