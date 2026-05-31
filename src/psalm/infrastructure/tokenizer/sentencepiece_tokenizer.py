"""SentencePiece adapter implementing the tokenizer ports.

Trains a sandhi-aware SentencePiece model from a text stream. SentencePiece is an
optional dependency (the ``data`` extra); the adapter imports it lazily and
raises a clear error if it is unavailable, so the core package and its tests run
without it.

The trainer passes ``hard_vocab_limit=False`` so that small corpora (e.g. the
60M proxy tier or unit tests) do not fail when the requested ``vocab_size`` is
larger than the number of inducible pieces; the realised vocabulary is reported
by :pyattr:`SentencePieceTokenizer.vocab_size`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from psalm.application.data.tokenizer import TokenizerSpec, TrainedTokenizer


class SentencePieceUnavailableError(RuntimeError):
    """Raised when the optional ``sentencepiece`` dependency is not installed."""


def _import_spm() -> object:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SentencePieceUnavailableError(
            "sentencepiece is not installed. Install the data extra: "
            "`uv pip install 'psalm[data]'`."
        ) from exc
    return spm


class SentencePieceTokenizer(TrainedTokenizer):
    """A trained SentencePiece model wrapped behind the :class:`TrainedTokenizer` port."""

    def __init__(self, model_path: str | Path) -> None:
        spm = _import_spm()
        self._model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor()  # type: ignore[attr-defined]
        self._sp.Load(str(self._model_path))

    @property
    def vocab_size(self) -> int:
        return int(self._sp.GetPieceSize())

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._sp.EncodeAsIds(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._sp.DecodeIds(list(ids)))


class SentencePieceTrainer:
    """Trains a SentencePiece tokenizer from a text stream (``TokenizerTrainer`` port)."""

    def train(
        self, texts: Iterable[str], spec: TokenizerSpec, out_dir: Path
    ) -> SentencePieceTokenizer:
        spm = _import_spm()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = out_dir / "corpus.txt"
        n_lines = 0
        with corpus_path.open("w", encoding="utf-8") as handle:
            for text in texts:
                line = text.replace("\n", " ").strip()
                if line:
                    handle.write(line + "\n")
                    n_lines += 1
        if n_lines == 0:
            raise ValueError("cannot train a tokenizer on an empty corpus")

        prefix = out_dir / "spm"
        kwargs: dict[str, object] = {
            "input": str(corpus_path),
            "model_prefix": str(prefix),
            "vocab_size": spec.vocab_size,
            "model_type": spec.model_type,
            "character_coverage": spec.character_coverage,
            "hard_vocab_limit": False,
        }
        symbols = spec.effective_symbols()
        if symbols:
            kwargs["user_defined_symbols"] = list(symbols)
        spm.SentencePieceTrainer.Train(**kwargs)  # type: ignore[attr-defined]
        return SentencePieceTokenizer(Path(f"{prefix}.model"))
