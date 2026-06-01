"""Real Sanskrit corpus sources via panini-data-toolkit.

The toolkit's ``corpus`` module exposes a unified ``iter_sentences(key, limit)``
over four license-checked Sanskrit sources — DCS (Apache-2.0), IndicCorp v2
(CC0-style), Itihāsa, and Sāmayik — reading from a local data directory
(``PANINI_DATA_DIR``). This adapter wraps that as a PSALM :class:`CorpusSource`
so real-text continuation streams (NL pretraining and tokenizer training) share
the same ``AnnotatedSentence`` contract as the synthetic generators.

Real-corpus sentences carry no gold kāraka parse (that supervision is unique to
the Saṃsādhanī generator); ``karaka_parse`` is therefore empty here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from psalm.application.data.ports import AnnotatedSentence

# License per toolkit source key (see panini-data-toolkit corpus docs).
_SOURCE_LICENSES: dict[str, str] = {
    "dcs": "Apache-2.0",
    "indiccorp_v2_sa": "CC0-1.0",
    "itihasa": "CC-BY-4.0",
    "saamayik": "CC-BY-4.0",
}


class CorpusNotProvisionedError(RuntimeError):
    """Raised when the requested toolkit corpus is not available on disk."""


class PaniniToolkitCorpusSource:
    """A PSALM ``CorpusSource`` backed by panini-data-toolkit's corpus loader."""

    def __init__(
        self,
        key: str,
        *,
        data_dir: str | None = None,
        language: str = "sa",
        limit: int | None = None,
    ) -> None:
        if key not in _SOURCE_LICENSES:
            raise ValueError(
                f"unknown corpus key {key!r}; expected one of {sorted(_SOURCE_LICENSES)}"
            )
        self.key = key
        self.name = f"panini-toolkit:{key}"
        self.license = _SOURCE_LICENSES[key]
        self.language = language
        self.limit = limit
        self._data_dir = data_dir

    def _activate_data_dir(self) -> None:
        if self._data_dir is not None:
            os.environ["PANINI_DATA_DIR"] = self._data_dir

    def is_available(self) -> bool:
        self._activate_data_dir()
        try:
            from panini_data_toolkit import corpus
        except ImportError:
            return False
        try:
            return bool(corpus.is_available(self.key))
        except Exception:  # noqa: BLE001
            return False

    def stream(self) -> Iterator[AnnotatedSentence]:
        self._activate_data_dir()
        try:
            from panini_data_toolkit import corpus
        except ImportError as exc:  # pragma: no cover - import guard
            raise CorpusNotProvisionedError(
                f"panini-data-toolkit is not installed; cannot stream corpus {self.key!r}."
            ) from exc
        if not corpus.is_available(self.key):
            raise CorpusNotProvisionedError(
                f"corpus {self.key!r} is not provisioned under "
                f"{corpus.get_data_dir()!r}. Run the toolkit's "
                "scripts/download_corpus.sh or set PANINI_DATA_DIR."
            )
        for text in corpus.iter_sentences(self.key, limit=self.limit):
            text = text.strip()
            if not text:
                continue
            yield AnnotatedSentence(
                text=text,
                language=self.language,
                karaka_parse=(),
                derivation=(),
                meta={"source": self.name, "license": self.license},
            )
