"""Pack serialized corpus lines into fixed-length next-token-prediction batches.

A tokenizer turns each line into ids; lines are concatenated with an EOS
separator into one long stream, then chunked into ``seq_len + 1`` windows so the
model predicts token *t+1* from tokens *<= t*. This is the standard packed-LM
input and keeps every token a training target (no padding waste).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch


class TokenPacker:
    """Encodes and packs text lines into causal LM windows."""

    def __init__(self, encode, *, eos_id: int, seq_len: int) -> None:
        self._encode = encode
        self.eos_id = eos_id
        self.seq_len = seq_len

    def _id_stream(self, lines: Iterable[str]) -> Iterator[int]:
        for line in lines:
            yield from self._encode(line)
            yield self.eos_id

    def windows(self, lines: Iterable[str]) -> Iterator[list[int]]:
        """Yield ``seq_len + 1``-length id windows (input + shifted target)."""
        buf: list[int] = []
        need = self.seq_len + 1
        for tid in self._id_stream(lines):
            buf.append(tid)
            if len(buf) == need:
                yield buf
                buf = []

    def packed_windows(self, lines: Iterable[str]) -> Iterator[list[int]]:
        """Yield ``seq_len``-length id windows (no next-token shift) for MLM encoders."""
        buf: list[int] = []
        for tid in self._id_stream(lines):
            buf.append(tid)
            if len(buf) == self.seq_len:
                yield buf
                buf = []

    def packed_batches(
        self, lines: Iterable[str], *, batch_size: int, device: str
    ) -> Iterator[torch.Tensor]:
        """Yield packed Long tensors of shape ``(batch_size, seq_len)``."""
        batch: list[list[int]] = []
        for window in self.packed_windows(lines):
            batch.append(window)
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long, device=device)
                batch = []
        if batch:
            yield torch.tensor(batch, dtype=torch.long, device=device)

    def batches(
        self, lines: Iterable[str], *, batch_size: int, device: str
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (input, target) Long tensors of shape (batch_size, seq_len)."""
        batch: list[list[int]] = []
        for window in self.windows(lines):
            batch.append(window)
            if len(batch) == batch_size:
                yield self._to_tensors(batch, device)
                batch = []
        if batch:
            yield self._to_tensors(batch, device)

    @staticmethod
    def _to_tensors(batch: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.tensor(batch, dtype=torch.long, device=device)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


class RoleStreamPacker:
    """Windows a flat role-label array into ``(batch, seq_len)`` tensors in lockstep
    with ``TokenPacker.packed_batches`` (RQ-B / SPEC 0003).

    The role array must be built with ``--with-eos-role`` so its per-line length equals
    TokenPacker's ``encode(line)+eos`` stream; then continuous modular ``seq_len`` windowing
    yields role windows positionally aligned with the token windows. Memory-light (only one
    batch materialised at a time) and exact (modular wrap mirrors the infinite token stream).
    """

    def __init__(self, roles: np.ndarray | list[int], *, seq_len: int) -> None:
        self._roles = np.asarray(roles, dtype=np.int64).ravel()
        if self._roles.size == 0:
            raise ValueError("RoleStreamPacker: empty role array")
        self.seq_len = seq_len

    def packed_batches(
        self, n_steps: int, batch_size: int, *, device: str
    ) -> Iterator[torch.Tensor]:
        n = self._roles.size
        step_len = batch_size * self.seq_len
        pos = 0
        for _ in range(n_steps):
            idx = np.arange(pos, pos + step_len) % n
            batch = self._roles[idx].reshape(batch_size, self.seq_len)
            pos = (pos + step_len) % n
            yield torch.tensor(batch, dtype=torch.long, device=device)
