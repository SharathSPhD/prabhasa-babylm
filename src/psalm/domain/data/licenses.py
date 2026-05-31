"""License modelling and corpus-source manifests for license-clean assembly.

Phase 1 assembles corpora from heterogeneous sources (DCS, GRETIL, BabyLM, HF
Sanskrit). The closure contract's integrity layer requires that every published
artifact be license-clean and attributed. This module models licenses and source
manifests and provides the checks the data pipeline uses before publishing.

Pure module; no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Redistribution(StrEnum):
    """How a source may be redistributed (drives publish gating)."""

    ALLOWED = "allowed"  # public domain / CC0: no conditions
    ATTRIBUTION = "attribution"  # CC-BY: redistribute with credit
    SHARE_ALIKE = "share_alike"  # CC-BY-SA: credit + same license
    RESEARCH_ONLY = "research_only"  # may use to train, must not redistribute raw text
    FORBIDDEN = "forbidden"  # may not redistribute


@dataclass(frozen=True)
class CorpusLicense:
    """A license with its redistribution semantics."""

    spdx: str
    redistribution: Redistribution
    url: str = ""

    @property
    def is_redistributable(self) -> bool:
        """True if the raw text may be republished (possibly with conditions)."""
        return self.redistribution in (
            Redistribution.ALLOWED,
            Redistribution.ATTRIBUTION,
            Redistribution.SHARE_ALIKE,
        )

    @property
    def requires_attribution(self) -> bool:
        return self.redistribution in (
            Redistribution.ATTRIBUTION,
            Redistribution.SHARE_ALIKE,
        )


#: Known licenses used by PSALM sources. Conservative by design.
KNOWN_LICENSES: dict[str, CorpusLicense] = {
    "CC0-1.0": CorpusLicense(
        "CC0-1.0", Redistribution.ALLOWED, "https://creativecommons.org/publicdomain/zero/1.0/"
    ),
    "PD": CorpusLicense("PD", Redistribution.ALLOWED),
    "MIT": CorpusLicense("MIT", Redistribution.ATTRIBUTION, "https://opensource.org/license/mit"),
    "apache-2.0": CorpusLicense(
        "apache-2.0", Redistribution.ATTRIBUTION, "https://www.apache.org/licenses/LICENSE-2.0"
    ),
    "CC-BY-4.0": CorpusLicense(
        "CC-BY-4.0", Redistribution.ATTRIBUTION, "https://creativecommons.org/licenses/by/4.0/"
    ),
    "CC-BY-SA-4.0": CorpusLicense(
        "CC-BY-SA-4.0",
        Redistribution.SHARE_ALIKE,
        "https://creativecommons.org/licenses/by-sa/4.0/",
    ),
    "CC-BY-SA-3.0": CorpusLicense(
        "CC-BY-SA-3.0",
        Redistribution.SHARE_ALIKE,
        "https://creativecommons.org/licenses/by-sa/3.0/",
    ),
    "research-only": CorpusLicense("research-only", Redistribution.RESEARCH_ONLY),
    "unknown": CorpusLicense("unknown", Redistribution.FORBIDDEN),
}


def license_for(spdx: str) -> CorpusLicense:
    """Look up a known license, defaulting to the FORBIDDEN ``unknown`` license.

    Defaulting to forbidden is deliberate: an unrecognised license must block
    publication until a human classifies it.
    """
    return KNOWN_LICENSES.get(spdx, KNOWN_LICENSES["unknown"])


@dataclass(frozen=True)
class SourceManifest:
    """Provenance + license record for one corpus source."""

    name: str
    license: CorpusLicense
    url: str = ""
    n_docs: int = 0
    n_tokens: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_docs < 0 or self.n_tokens < 0:
            raise ValueError("n_docs and n_tokens must be >= 0")

    @property
    def is_clean(self) -> bool:
        """True if this source may be redistributed in a published dataset."""
        return self.license.is_redistributable


def assembly_is_clean(manifests: list[SourceManifest]) -> bool:
    """True if every source in the assembly is redistributable."""
    return all(m.is_clean for m in manifests)


def summarize_assembly(manifests: list[SourceManifest]) -> dict[str, object]:
    """Aggregate manifests into a publish-readiness summary."""
    return {
        "n_sources": len(manifests),
        "total_docs": sum(m.n_docs for m in manifests),
        "total_tokens": sum(m.n_tokens for m in manifests),
        "all_clean": assembly_is_clean(manifests),
        "blocked": [m.name for m in manifests if not m.is_clean],
        "licenses": sorted({m.license.spdx for m in manifests}),
    }
