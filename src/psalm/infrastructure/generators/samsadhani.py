"""Saṃsādhanī Pāṇinian generator adapter.

The Pāṇinian synthetic-Sanskrit stream is produced by the Saṃsādhanī
computational-Sanskrit toolset (Department of Sanskrit Studies, University of
Hyderabad). Saṃsādhanī is an **external dependency**: it is not pip-installable
and is provisioned separately on the DGX Spark (see ``configure``). This adapter
implements the ``SentenceGenerator`` port and yields, for each generated
sentence, the surface form plus the gold kāraka parse and derivation that the
generator produces for free.

Until the external tool is provisioned, ``stream`` raises a clear, actionable
error rather than silently returning fake data — consistent with the program's
integrity rules (no fabricated results).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from psalm.application.data.ports import AnnotatedSentence


class SamsadhaniNotConfiguredError(RuntimeError):
    """Raised when the external Saṃsādhanī toolset has not been provisioned."""


class SamsadhaniGenerator:
    """Adapter over the external Saṃsādhanī generator.

    Parameters
    ----------
    install_root:
        Path to the provisioned Saṃsādhanī installation on the host. When unset
        or missing, the adapter is "not configured" and refuses to fabricate data.
    """

    def __init__(self, install_root: str | Path | None = None) -> None:
        self.install_root = Path(install_root) if install_root else None

    @property
    def is_configured(self) -> bool:
        return self.install_root is not None and self.install_root.exists()

    def _require_configured(self) -> Path:
        if not self.is_configured:
            raise SamsadhaniNotConfiguredError(
                "Saṃsādhanī is not provisioned. It is an external tool from the "
                "University of Hyderabad (https://sanskrit.uohyd.ac.in/scl/), not "
                "a pip package. Provision it on the DGX Spark, then pass its path "
                "via SamsadhaniGenerator(install_root=...). See "
                "docs/data/samsadhani-setup.md."
            )
        assert self.install_root is not None
        return self.install_root

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        """Stream ``n`` annotated Pāṇinian sentences.

        The concrete subprocess/API binding to the provisioned toolset is wired
        in once Saṃsādhanī is available on the host (Phase 1 de-risking task);
        the parsing of its output into ``AnnotatedSentence`` (surface + kāraka +
        derivation) is the adapter's responsibility and is unit-tested against
        recorded fixtures.
        """
        self._require_configured()
        raise NotImplementedError(
            "Saṃsādhanī binding is provisioned in Phase 1 on the DGX Spark; "
            "this method is wired to the host installation there."
        )
