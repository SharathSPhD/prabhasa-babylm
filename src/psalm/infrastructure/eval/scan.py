"""SCAN compositional-generalization benchmark loader.

SCAN (Lake & Baroni, 2018) maps a natural-language command to an action
sequence, e.g. ``jump twice -> I_JUMP I_JUMP``. The systematic-generalization
splits (notably ``length`` and ``add_prim``) are the standard probe for whether
a model composes rather than memorises — the core of the H1 compositional claim.

Lines are ``IN: <command> OUT: <action seq>``. This loader downloads a split
once into a cache directory and parses it into ``(source, target)`` pairs for
``H1Runner.EvalSets.compositional``. No fabrication: if the file is absent and
the network is unavailable, it raises rather than inventing data.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

_BASE = "https://raw.githubusercontent.com/brendenlake/SCAN/master"

# split -> (train_relpath, test_relpath)
SPLITS: dict[str, tuple[str, str]] = {
    "simple": (
        "simple_split/tasks_train_simple.txt",
        "simple_split/tasks_test_simple.txt",
    ),
    "length": (
        "length_split/tasks_train_length.txt",
        "length_split/tasks_test_length.txt",
    ),
    "addprim_jump": (
        "add_prim_split/tasks_train_addprim_jump.txt",
        "add_prim_split/tasks_test_addprim_jump.txt",
    ),
}


class ScanUnavailableError(RuntimeError):
    """Raised when a SCAN split is neither cached nor downloadable."""


def _cache_path(cache_dir: Path, relpath: str) -> Path:
    return cache_dir / relpath.replace("/", "__")


def _ensure_file(relpath: str, cache_dir: Path, *, allow_download: bool) -> Path:
    dest = _cache_path(cache_dir, relpath)
    if dest.exists():
        return dest
    if not allow_download:
        raise ScanUnavailableError(
            f"SCAN file {relpath!r} not cached at {dest!r} and download disabled."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE}/{relpath}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed host
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise ScanUnavailableError(f"failed to download SCAN from {url}: {exc}") from exc
    dest.write_bytes(data)
    return dest


def _parse(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "IN:" not in line or "OUT:" not in line:
            continue
        _, rest = line.split("IN:", 1)
        cmd, act = rest.split("OUT:", 1)
        pairs.append((cmd.strip(), act.strip()))
    return pairs


def load_scan(
    split: str = "length",
    *,
    which: str = "test",
    cache_dir: str | Path = "data/cache/scan",
    limit: int | None = None,
    allow_download: bool = True,
) -> list[tuple[str, str]]:
    """Load ``(command, action)`` pairs for a SCAN split.

    Parameters
    ----------
    split: one of :data:`SPLITS` (``simple``, ``length``, ``addprim_jump``).
    which: ``"train"`` or ``"test"``.
    limit: optional cap on the number of pairs returned.
    allow_download: if False, only use a pre-cached file.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown SCAN split {split!r}; expected {sorted(SPLITS)}")
    if which not in ("train", "test"):
        raise ValueError("which must be 'train' or 'test'")
    train_rel, test_rel = SPLITS[split]
    relpath = train_rel if which == "train" else test_rel
    path = _ensure_file(relpath, Path(cache_dir), allow_download=allow_download)
    pairs = _parse(path)
    return pairs[:limit] if limit is not None else pairs
