"""BLiMP minimal pair loader for weak paradigms (NPI licensing, filler-gap, islands).

Assembles clean/corrupt pairs for circuit-targeting analysis. Focuses on paradigms
where the validated v0.2 recipe underperforms (NPI licensing, filler-gap, islands).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer


@dataclass
class MinimalPair:
    """Single clean/corrupt pair for attribution analysis."""

    paradigm: str
    clean_sentence: str
    corrupt_sentence: str
    clean_label: int  # 1 for good, 0 for bad
    corrupt_label: int


class NpiLicensingLoader:
    """Load NPI licensing minimal pairs from BLiMP data.

    NPI licensing is flagged as a weak paradigm (F5-F8: ~2pp under baseline).
    Captures the "any" vs "the" distinction under negation scope.
    """

    def __init__(
        self, blimp_root: str | Path, paradigm: str = "npi_licensing"
    ) -> None:
        """Initialize loader.

        Args:
            blimp_root: Path to BLiMP filtered data directory.
            paradigm: Paradigm name (e.g., 'npi_licensing').
        """
        self.blimp_root = Path(blimp_root)
        self.paradigm = paradigm
        self.paradigm_file = self.blimp_root / f"{paradigm}.jsonl"

        if not self.paradigm_file.exists():
            raise FileNotFoundError(f"Paradigm file not found: {self.paradigm_file}")

    def load_pairs(self) -> list[MinimalPair]:
        """Load all pairs for the paradigm.

        Returns:
            List of MinimalPair objects.
        """
        pairs = []
        with open(self.paradigm_file, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                # BLiMP-2026 format: "sentence_good" and "sentence_bad" in each line
                sentence_good = data.get("sentence_good")
                sentence_bad = data.get("sentence_bad")

                if sentence_good and sentence_bad:
                    pairs.append(
                        MinimalPair(
                            paradigm=self.paradigm,
                            clean_sentence=sentence_good,
                            corrupt_sentence=sentence_bad,
                            clean_label=1,
                            corrupt_label=0,
                        )
                    )

        return pairs

    def load_pairs_iter(self) -> Iterator[MinimalPair]:
        """Iterate over pairs without loading all at once."""
        with open(self.paradigm_file, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                sentence_good = data.get("sentence_good")
                sentence_bad = data.get("sentence_bad")

                if sentence_good and sentence_bad:
                    yield MinimalPair(
                        paradigm=self.paradigm,
                        clean_sentence=sentence_good,
                        corrupt_sentence=sentence_bad,
                        clean_label=1,
                        corrupt_label=0,
                    )


def batch_tokenize_pairs(
    pairs: list[MinimalPair],
    tokenizer: AutoTokenizer,
    mask_token: str = "[MASK]",
    max_length: int = 192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize clean and corrupt sentences with [MASK] at the position of interest.

    For now, we [MASK] the final token in each sentence (simplified heuristic).
    A more sophisticated approach would identify the key word (NPI, filler, etc.)
    and mask that specifically.

    Args:
        pairs: List of MinimalPair objects.
        tokenizer: HF tokenizer.
        mask_token: Token to use for masking (default "[MASK]").
        max_length: Maximum sequence length.

    Returns:
        Tuple of (clean_input_ids, corrupt_input_ids, clean_labels, corrupt_labels).
        - clean/corrupt_input_ids: shape (len(pairs), max_length)
        - clean/corrupt_labels: shape (len(pairs), max_length); -100 for non-mask positions
    """
    clean_inputs, corrupt_inputs = [], []
    clean_labels, corrupt_labels = [], []

    for pair in pairs:
        # Tokenize clean sentence
        clean_tokens = tokenizer.encode(pair.clean_sentence, add_special_tokens=False)  # type: ignore[attr-defined]
        # Mask the last token (simplified; ideally target-word-specific)
        if len(clean_tokens) > 0:
            # Create a version with the last token masked
            clean_masked = clean_tokens[:-1] + [tokenizer.mask_token_id]  # type: ignore[attr-defined]
            # Label: the true final token
            clean_label_seq = [-100] * (len(clean_masked) - 1) + [clean_tokens[-1]]
        else:
            clean_masked = [tokenizer.mask_token_id]  # type: ignore[attr-defined]
            clean_label_seq = [tokenizer.mask_token_id]  # type: ignore[attr-defined]

        # Tokenize corrupt sentence
        corrupt_tokens = tokenizer.encode(pair.corrupt_sentence, add_special_tokens=False)  # type: ignore[attr-defined]
        if len(corrupt_tokens) > 0:
            corrupt_masked = corrupt_tokens[:-1] + [tokenizer.mask_token_id]  # type: ignore[attr-defined]
            corrupt_label_seq = [-100] * (len(corrupt_masked) - 1) + [corrupt_tokens[-1]]
        else:
            corrupt_masked = [tokenizer.mask_token_id]  # type: ignore[attr-defined]
            corrupt_label_seq = [tokenizer.mask_token_id]  # type: ignore[attr-defined]

        # Pad to max_length
        pad_len_clean = max(0, max_length - len(clean_masked))
        clean_padded = (clean_masked + [tokenizer.pad_token_id] * pad_len_clean)[  # type: ignore[attr-defined]
            :max_length
        ]

        pad_len_corrupt = max(0, max_length - len(corrupt_masked))
        corrupt_padded = (
            corrupt_masked + [tokenizer.pad_token_id] * pad_len_corrupt  # type: ignore[attr-defined]
        )[:max_length]

        pad_len_clean_label = max(0, max_length - len(clean_label_seq))
        clean_label_padded = (clean_label_seq + [-100] * pad_len_clean_label)[
            :max_length
        ]

        pad_len_corrupt_label = max(0, max_length - len(corrupt_label_seq))
        corrupt_label_padded = (
            corrupt_label_seq + [-100] * pad_len_corrupt_label
        )[:max_length]

        clean_inputs.append(clean_padded)
        corrupt_inputs.append(corrupt_padded)
        clean_labels.append(clean_label_padded)
        corrupt_labels.append(corrupt_label_padded)

    return (
        torch.tensor(clean_inputs, dtype=torch.long),
        torch.tensor(corrupt_inputs, dtype=torch.long),
        torch.tensor(clean_labels, dtype=torch.long),
        torch.tensor(corrupt_labels, dtype=torch.long),
    )
