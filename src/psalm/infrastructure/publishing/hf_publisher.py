"""Hugging Face dataset publisher implementing the publishing port.

Writes the rendered dataset card to ``README.md`` and uploads the local dataset
directory to the Hub under the configured namespace. ``huggingface_hub`` is
imported lazily and a token is required; without one the adapter raises
:class:`HFNotConfiguredError` rather than attempting an anonymous upload.

``dry_run=True`` renders and writes the card locally without any network call,
so the card can be previewed and tested offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from psalm.application.data.publishing import DatasetCard


class HFNotConfiguredError(RuntimeError):
    """Raised when no HF token is available or ``huggingface_hub`` is missing."""


@dataclass
class HfDatasetPublisher:
    """Publishes a dataset directory + card to the Hugging Face Hub."""

    token: str | None = None

    def publish(
        self,
        repo_id: str,
        local_dir: Path,
        card: DatasetCard,
        *,
        private: bool = False,
        dry_run: bool = False,
    ) -> str:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        readme = local_dir / "README.md"
        readme.write_text(card.to_markdown(), encoding="utf-8")
        if dry_run:
            return str(readme)

        token = self.token or os.environ.get("HF_TOKEN")
        if not token:
            raise HFNotConfiguredError(
                "No HF token: set HF_TOKEN or pass token=... to publish to the Hub."
            )
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise HFNotConfiguredError(
                "huggingface_hub is not installed; install the data extra."
            ) from exc
        api = HfApi(token=token)  # pragma: no cover - network
        api.create_repo(  # pragma: no cover - network
            repo_id, repo_type="dataset", private=private, exist_ok=True
        )
        api.upload_folder(  # pragma: no cover - network
            repo_id=repo_id, repo_type="dataset", folder_path=str(local_dir)
        )
        return f"https://huggingface.co/datasets/{repo_id}"  # pragma: no cover - network
