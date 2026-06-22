"""Tests for BLiMP minimal pair loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from psalm.infrastructure.acd.blimp_minimal_pairs import (
    MinimalPair,
    NpiLicensingLoader,
    batch_tokenize_pairs,
)


@pytest.fixture
def temp_blimp_dir() -> Path:
    """Create a temporary BLiMP-like directory with NPI licensing data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a simple NPI licensing file (BLiMP-2026 format)
        npi_file = tmp_path / "npi_licensing.jsonl"
        pairs = [
            {
                "sentence_good": "No one knows anything .",
                "sentence_bad": "The one knows something .",
            },
            {"sentence_good": "Nobody ate any cake .", "sentence_bad": "Everyone ate some cake ."},
        ]
        with open(npi_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        yield tmp_path


def test_minimal_pair_dataclass() -> None:
    """Test MinimalPair creation."""
    pair = MinimalPair(
        paradigm="npi_licensing",
        clean_sentence="No one knows anything.",
        corrupt_sentence="The one knows something.",
        clean_label=1,
        corrupt_label=0,
    )

    assert pair.paradigm == "npi_licensing"
    assert pair.clean_label == 1
    assert pair.corrupt_label == 0


def test_npi_loader_init(temp_blimp_dir: Path) -> None:
    """Test NPI loader initialization."""
    loader = NpiLicensingLoader(temp_blimp_dir)

    assert loader.paradigm == "npi_licensing"
    assert loader.paradigm_file.exists()


def test_npi_loader_load_pairs(temp_blimp_dir: Path) -> None:
    """Test loading minimal pairs."""
    loader = NpiLicensingLoader(temp_blimp_dir)
    pairs = loader.load_pairs()

    assert len(pairs) == 2
    assert pairs[0].clean_sentence == "No one knows anything ."
    assert pairs[0].corrupt_sentence == "The one knows something ."
    assert pairs[1].clean_sentence == "Nobody ate any cake ."


def test_npi_loader_iter(temp_blimp_dir: Path) -> None:
    """Test iterating over pairs."""
    loader = NpiLicensingLoader(temp_blimp_dir)

    pairs = list(loader.load_pairs_iter())

    assert len(pairs) == 2


def test_batch_tokenize_pairs(temp_blimp_dir: Path) -> None:
    """Test tokenization of minimal pairs."""
    loader = NpiLicensingLoader(temp_blimp_dir)
    pairs = loader.load_pairs()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    clean_ids, corrupt_ids, clean_labels, corrupt_labels = batch_tokenize_pairs(
        pairs,
        tokenizer,
        max_length=64,
    )

    assert clean_ids.shape == (len(pairs), 64)
    assert corrupt_ids.shape == (len(pairs), 64)
    assert clean_labels.shape == (len(pairs), 64)
    assert corrupt_labels.shape == (len(pairs), 64)

    # Check that clean and corrupt are different
    assert not torch.equal(clean_ids, corrupt_ids)

    # Check that labels contain -100 (ignore) and real token IDs
    assert (clean_labels == -100).any()
    assert (clean_labels >= 0).any()


def test_batch_tokenize_pairs_mask_positioning(temp_blimp_dir: Path) -> None:
    """Test that [MASK] is positioned correctly."""
    loader = NpiLicensingLoader(temp_blimp_dir)
    pairs = loader.load_pairs()[:1]  # Single pair

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    clean_ids, _, _, _ = batch_tokenize_pairs(
        pairs,
        tokenizer,
        max_length=64,
    )

    # Find mask position
    mask_positions = (clean_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Should have at least one mask
    assert len(mask_positions) > 0
