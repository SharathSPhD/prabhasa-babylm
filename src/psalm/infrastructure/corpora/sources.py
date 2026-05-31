"""Corpus-source adapters for license-clean assembly.

Implements the :class:`~psalm.application.data.ports.CorpusSource` port for two
concrete backends:

* :class:`LocalTextCorpusSource` — one sentence per line from a provisioned local
  file (used for DCS/GRETIL exports staged on the DGX Spark).
* :class:`HFDatasetCorpusSource` — a Hugging Face dataset split, with ``datasets``
  imported lazily.

Both refuse to fabricate data: if the backing data is not provisioned (missing
file, or ``datasets`` not installed / not downloaded) they raise
:class:`SourceNotProvisionedError`. A registry of the four named PSALM sources
records their license and provisioning notes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.licenses import CorpusLicense, license_for


class SourceNotProvisionedError(RuntimeError):
    """Raised when a corpus source's backing data has not been provisioned."""


@dataclass(frozen=True)
class LocalTextCorpusSource:
    """A corpus source reading one sentence per line from a local file."""

    name: str
    license: CorpusLicense
    path: Path
    language: str = "sa"

    def stream(self) -> Iterator[AnnotatedSentence]:
        if not self.path.exists():
            raise SourceNotProvisionedError(
                f"{self.name}: local corpus file not found at {self.path}. "
                "Provision the export before streaming."
            )
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line:
                yield AnnotatedSentence(text=line, language=self.language)


@dataclass(frozen=True)
class HFDatasetCorpusSource:
    """A corpus source backed by a Hugging Face dataset split."""

    name: str
    license: CorpusLicense
    repo_id: str
    split: str = "train"
    text_column: str = "text"
    language: str = "sa"

    def stream(self) -> Iterator[AnnotatedSentence]:
        try:
            import datasets
        except ImportError as exc:
            raise SourceNotProvisionedError(
                f"{self.name}: the `datasets` library is not installed "
                "(install the data extra) so the HF source cannot be streamed."
            ) from exc
        dataset = datasets.load_dataset(self.repo_id, split=self.split)  # pragma: no cover
        for row in dataset:  # pragma: no cover - requires network/data
            text = str(row[self.text_column]).strip()
            if text:
                yield AnnotatedSentence(text=text, language=self.language)


@dataclass(frozen=True)
class KnownSource:
    """Static description of a named PSALM corpus source."""

    key: str
    spdx: str
    url: str
    note: str

    @property
    def license(self) -> CorpusLicense:
        return license_for(self.spdx)


#: Named Phase-1 sources. Licenses verified on the HF Hub where noted (ADR-0010);
#: unverified sources default to non-redistributable until classified.
KNOWN_SOURCES: dict[str, KnownSource] = {
    "vidyut-prakriya": KnownSource(
        "vidyut-prakriya",
        "MIT",
        "https://huggingface.co/datasets/preetammukherjee/sanskrit_morph_prakriya",
        "Open Pāṇinian derivation engine output (verb forms); MIT, license-clean. "
        "Primary generator per ADR-0010; sentence-level generation runs Vidyut's API.",
    ),
    "dcs": KnownSource(
        "dcs",
        "apache-2.0",
        "https://huggingface.co/datasets/sampathlonka/DCS_Sanskrit_Morphology_v1",
        "Digital Corpus of Sanskrit; 721K morphologically-analysed sentences; "
        "Apache-2.0, license-clean. Primary real corpus per ADR-0010.",
    ),
    "gretil": KnownSource(
        "gretil",
        "unknown",
        "https://huggingface.co/datasets/paws/sanskrit-verses-gretil",
        "449K GRETIL verses; no explicit license tag on the HF mirror — gated until "
        "per-text terms are confirmed.",
    ),
    "babylm": KnownSource(
        "babylm",
        "research-only",
        "https://babylm.github.io/",
        "BabyLM English corpus is a mix of sources; treat as research-only, do not republish raw text.",
    ),
    "hf-sanskrit": KnownSource(
        "hf-sanskrit",
        "unknown",
        "https://huggingface.co/datasets",
        "Per-dataset license must be confirmed before assembly; defaults to forbidden until classified.",
    ),
}
