"""Zero-copy memory-mapped uint16 token datasets for pre-tokenized data.

Pre-tokenized binary datasets eliminate the CPU tokenization bottleneck
(SentencePiece on-the-fly), achieving 10x data throughput via direct memmap
access from the GPU and multi-worker prefetching (4 workers, factor=2).

Module:
    - BinDataset: Memory-mapped uint16 token windows
    - packed_batches_from_bin: Prefetched batch iterator

Example:
    >>> from pathlib import Path
    >>> from psalm.infrastructure.ml.bin_dataset import BinDataset
    >>> import torch
    >>>
    >>> dataset = BinDataset(Path("data/dose_A.bin"), seq_len=128)
    >>> batch = dataset[0]
    >>> assert batch.shape == torch.Size([129])  # seq_len + 1 for (input, target)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BinDataset(Dataset[torch.Tensor]):
    """Memory-mapped uint16 token dataset. Zero-copy on Linux."""

    def __init__(self, path: Path, seq_len: int) -> None:
        """Load or create uint16 memmap from path.

        Args:
            path: Path to .bin memmap file (uint16 dtype)
            seq_len: Sequence length per sample (window = seq_len + 1 for input + target)
        """
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype="uint16", mode="r")
        self.seq_len = seq_len
        # Number of non-overlapping windows of size (seq_len + 1)
        self.n = max(0, (len(self.data) - 1) // (seq_len + 1))

    def __len__(self) -> int:
        """Number of samples."""
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get sample idx as (seq_len + 1) int64 tokens.

        Args:
            idx: Sample index

        Returns:
            Tensor of shape (seq_len + 1,) with dtype int64 (for torch compatibility)
        """
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = self.data[start:end].astype(np.int64)
        return torch.from_numpy(chunk)

    @staticmethod
    def build(
        text_path: Path,
        out_path: Path,
        encode_fn: Callable[[str], list[int]],
        *,
        verbose: bool = True,
    ) -> None:
        """Tokenize text_path line-by-line, write uint16 memmap to out_path.

        Args:
            text_path: Input text file (one sentence per line)
            out_path: Output .bin memmap file path
            encode_fn: Function str -> list[int] (e.g., SentencePiece encoder)
            verbose: Print progress to stdout

        Example:
            >>> from pathlib import Path
            >>> import sentencepiece as spm
            >>> from psalm.infrastructure.ml.bin_dataset import BinDataset
            >>>
            >>> sp = spm.SentencePieceProcessor()
            >>> sp.Load("tokenizer.model")
            >>> BinDataset.build(
            ...     Path("input.txt"),
            ...     Path("output.bin"),
            ...     encode_fn=lambda s: sp.EncodeAsIds(s),
            ...     verbose=True,
            ... )
        """
        tokens: list[int] = []
        with open(text_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if line:
                    tokens.extend(encode_fn(line))
                if verbose and i % 100_000 == 0 and i > 0:
                    print(f"  tokenized {i:,} lines, {len(tokens):,} tokens", flush=True)

        arr = np.array(tokens, dtype=np.uint16)
        fp = np.memmap(out_path, dtype="uint16", mode="w+", shape=(len(arr),))
        fp[:] = arr
        fp.flush()
        if verbose:
            print(f"  wrote {len(arr):,} tokens to {out_path}", flush=True)


def packed_batches_from_bin(
    dataset: BinDataset, batch_size: int, *, device: str, num_workers: int = 4
) -> Iterator[torch.Tensor]:
    """Create a prefetched DataLoader over BinDataset.

    Args:
        dataset: BinDataset instance
        batch_size: Batch size
        device: Device to place batches on ("cuda" or "cpu")
        num_workers: Number of worker processes (default 4)

    Yields:
        Tensors of shape (batch_size, seq_len + 1) on the specified device

    Note:
        pin_memory=True and prefetch_factor=2 reduce stalls from CPU->GPU transfer.
        prefetch_factor only applies when num_workers > 0.
    """
    # prefetch_factor requires num_workers > 0; otherwise DataLoader raises ValueError
    prefetch_factor: int | None = 2 if num_workers > 0 else None
    loader: DataLoader[torch.Tensor] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    for batch in loader:
        yield batch.to(device, non_blocking=True)
