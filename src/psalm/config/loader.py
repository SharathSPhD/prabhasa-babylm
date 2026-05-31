"""Load and hash resolved run configurations.

The config hash recorded with every run is the reproducibility anchor: two runs
with the same hash were configured identically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from psalm.config.settings import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file into a validated :class:`ExperimentConfig`."""
    raw = Path(path).read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(raw) or {}
    return ExperimentConfig.model_validate(data)


def config_hash(config: ExperimentConfig) -> str:
    """Stable short hash of a resolved config, for the experiment ledger."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
