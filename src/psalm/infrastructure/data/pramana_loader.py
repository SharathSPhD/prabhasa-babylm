"""Infrastructure layer: load pramana files and convert to NLI format.

This module handles I/O and integrates the pure domain converter with file access.
It provides a single entry point for loading pramana JSONL files and converting
all examples to NLI format for use with PSALM's H2 fine-tuning pipeline.

Following hexagonal architecture:
- domain/pramana_converter.py: Pure transformation logic (no I/O)
- infrastructure/data/pramana_loader.py: I/O and file handling (this module)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from psalm.domain.data.pramana_converter import (
    PramanaConversionError,
    convert_pramana_example,
)

logger = logging.getLogger(__name__)


def convert_pramana_jsonl(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load and convert pramana JSONL file to NLI format.

    Args:
        jsonl_path: Path to pramana JSONL file (e.g., data/training/stage_0.jsonl)

    Returns:
        List of NLI-format dicts with keys: premise, hypothesis, label

    Behavior:
    - Reads JSONL line by line (memory-efficient for large files)
    - Skips malformed JSON lines with warning
    - Skips examples that fail conversion (logs reason)
    - Returns successfully converted examples
    """
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"pramana JSONL file not found: {jsonl_path}")

    nli_examples: list[dict[str, Any]] = []
    line_num = 0
    skipped = 0

    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                pramana_ex = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: malformed JSON - {e}")
                skipped += 1
                continue

            # Convert to NLI
            try:
                nli_ex = convert_pramana_example(pramana_ex)
                nli_examples.append(nli_ex)
            except PramanaConversionError as e:
                logger.warning(f"Line {line_num}: conversion failed - {e}")
                skipped += 1
                continue

    if skipped > 0:
        logger.info(f"Loaded {jsonl_path}: {len(nli_examples)} converted, {skipped} skipped")
    else:
        logger.info(f"Loaded {jsonl_path}: {len(nli_examples)} examples")

    return nli_examples


def load_pramana_datasets(
    stage_0_path: str | Path | None = None,
    stage_1_path: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load pramana stage_0 and/or stage_1 datasets.

    Args:
        stage_0_path: Path to stage_0.jsonl (optional)
        stage_1_path: Path to stage_1.jsonl (optional)

    Returns:
        Dict with keys 'stage_0', 'stage_1', or 'combined' (if both loaded)
        Each value is a list of NLI-format dicts

    Example:
        datasets = load_pramana_datasets(
            Path("pramana/data/training/stage_0.jsonl"),
            Path("pramana/data/training/stage_1.jsonl"),
        )
        combined = datasets["combined"]  # 75 examples
    """
    datasets: dict[str, list[dict[str, Any]]] = {}

    if stage_0_path:
        datasets["stage_0"] = convert_pramana_jsonl(stage_0_path)

    if stage_1_path:
        datasets["stage_1"] = convert_pramana_jsonl(stage_1_path)

    if stage_0_path and stage_1_path:
        datasets["combined"] = datasets["stage_0"] + datasets["stage_1"]

    return datasets


__all__ = [
    "convert_pramana_jsonl",
    "load_pramana_datasets",
]
