"""COGS compositional-generalization benchmark loader (Kim & Linzen, 2020).

COGS maps an English sentence to a logical form (a conjunction of predicate-role
terms), e.g. ``A rose was helped by a dog .`` ->
``rose ( x _ 1 ) AND help . theme ( x _ 3 , x _ 1 ) AND help . agent ( ... )``.
The training set is in-distribution; the **gen** set holds 21 systematic-
generalization categories of 1000 examples each.

PSALM adopts COGS as the primary H1 compositional benchmark (ADR-0014) because it
spans a difficulty spectrum — a *learnable* lexical-generalization tier (baseline
well off the 0% floor) and a hard *structural-generalization* tier (recursion and
PP-role reanalysis, where vanilla transformers floor) — and because it is
*mechanism-aligned*: kāraka analysis is a theory of argument roles, and most COGS
categories test exactly argument-role transfer (subject↔object, active↔passive,
dative alternation) to novel lexical items and structures. If the Pāṇinian prior
helps anywhere, COGS argument-role generalization is where the mechanism predicts
it should.

Lines are loaded as ``(sentence, logical_form)`` pairs for
``H1Runner.EvalSets.compositional``. No fabrication: absent a cache and network,
this raises rather than inventing data.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

_BASE = "https://raw.githubusercontent.com/najoungkim/COGS/master/data"

# which -> remote filename
_FILES: dict[str, str] = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
    "gen": "gen.tsv",
}

#: The three structural-generalization categories — recursion depth and PP-role
#: reanalysis — where small from-scratch models floor. The hard, mechanism-
#: aligned tier of the H1 test.
STRUCTURAL_CATEGORIES: frozenset[str] = frozenset(
    {"cp_recursion", "pp_recursion", "obj_pp_to_subj_pp"}
)

#: The remaining 18 gen categories are lexical generalization (novel item in a
#: known structural slot / argument-role transfer) — the learnable, off-floor
#: tier that gives the baseline room to score above 0.
_ALL_GEN_CATEGORIES: frozenset[str] = frozenset(
    {
        "obj_to_subj_common", "obj_to_subj_proper", "subj_to_obj_common",
        "subj_to_obj_proper", "prim_to_subj_common", "prim_to_subj_proper",
        "prim_to_obj_common", "prim_to_obj_proper", "prim_to_inf_arg",
        "active_to_passive", "passive_to_active", "do_dative_to_pp_dative",
        "pp_dative_to_do_dative", "unacc_to_transitive",
        "obj_omitted_transitive_to_transitive",
        "only_seen_as_transitive_subj_as_unacc_subj",
        "only_seen_as_unacc_subj_as_unerg_subj",
        "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj",
    }
)
LEXICAL_CATEGORIES: frozenset[str] = _ALL_GEN_CATEGORIES - STRUCTURAL_CATEGORIES


class CogsUnavailableError(RuntimeError):
    """Raised when COGS data is neither cached nor downloadable."""


def _cache_path(cache_dir: Path, fname: str) -> Path:
    return cache_dir / fname


def _ensure_file(fname: str, cache_dir: Path, *, allow_download: bool) -> Path:
    dest = _cache_path(cache_dir, fname)
    if dest.exists():
        return dest
    if not allow_download:
        raise CogsUnavailableError(f"COGS file {fname!r} not cached at {dest!r} and download disabled.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_BASE}/{fname}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed host
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise CogsUnavailableError(f"failed to download COGS from {url}: {exc}") from exc
    dest.write_bytes(data)
    return dest


def _parse(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        sentence, lf, category = parts[0].strip(), parts[1].strip(), parts[2].strip()
        rows.append((sentence, lf, category))
    return rows


def load_cogs(
    which: str = "gen",
    *,
    tier: str | None = None,
    categories: frozenset[str] | set[str] | None = None,
    cache_dir: str | Path = "data/cache/cogs",
    limit: int | None = None,
    allow_download: bool = True,
) -> list[tuple[str, str]]:
    """Load ``(sentence, logical_form)`` pairs for a COGS split.

    Parameters
    ----------
    which: ``train`` / ``dev`` / ``test`` / ``gen``.
    tier: for ``gen`` only — ``"lexical"`` or ``"structural"`` to restrict to that
        difficulty tier (ignored for non-gen splits).
    categories: explicit category filter (overrides ``tier``) for ``gen``.
    limit: optional cap (applied after filtering).
    """
    if which not in _FILES:
        raise ValueError(f"unknown COGS split {which!r}; expected {sorted(_FILES)}")
    path = _ensure_file(_FILES[which], Path(cache_dir), allow_download=allow_download)
    rows = _parse(path)

    keep: frozenset[str] | set[str] | None = categories
    if keep is None and tier is not None:
        if tier == "lexical":
            keep = LEXICAL_CATEGORIES
        elif tier == "structural":
            keep = STRUCTURAL_CATEGORIES
        else:
            raise ValueError(f"unknown tier {tier!r}; expected 'lexical' or 'structural'")

    pairs = [
        (sentence, lf) for sentence, lf, category in rows if keep is None or category in keep
    ]
    return pairs[:limit] if limit is not None else pairs
