"""Train/eval leakage (disjointness) check — fails closed (ADR-0035).

A held-out evaluation is only valid if no eval string appears in training. This
module computes the exact string-set overlap and refuses to proceed when any is
found, so a contaminated split aborts the battery rather than reporting inflated
numbers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class LeakageError(RuntimeError):
    """Raised when train/eval string sets overlap (fails closed)."""


@dataclass(frozen=True)
class LeakageReport:
    """Outcome of a train/eval disjointness check."""

    n_train: int
    n_eval: int
    n_overlap: int
    examples: tuple[str, ...]

    @property
    def disjoint(self) -> bool:
        return self.n_overlap == 0


def _normalize(text: str) -> str:
    """Whitespace-collapsed exact key (so trivial spacing differences still count)."""
    return " ".join(text.split())


def check_disjoint(
    train: Iterable[str],
    eval_set: Iterable[str],
    *,
    max_examples: int = 5,
) -> LeakageReport:
    """Return a :class:`LeakageReport` of the train∩eval string overlap.

    Does not raise — use :func:`assert_disjoint` to fail closed. Comparison is on
    whitespace-normalized exact strings.
    """
    train_keys = {_normalize(t) for t in train}
    eval_keys = [_normalize(e) for e in eval_set]
    eval_unique = set(eval_keys)
    overlap = sorted(train_keys & eval_unique)
    return LeakageReport(
        n_train=len(train_keys),
        n_eval=len(eval_unique),
        n_overlap=len(overlap),
        examples=tuple(overlap[:max_examples]),
    )


def assert_disjoint(train: Iterable[str], eval_set: Iterable[str]) -> LeakageReport:
    """Return the report if train and eval are disjoint, else raise ``LeakageError``."""
    report = check_disjoint(train, eval_set)
    if not report.disjoint:
        raise LeakageError(
            f"train/eval leakage: {report.n_overlap} eval string(s) appear in training "
            f"(e.g. {report.examples!r}); refusing to report a contaminated eval."
        )
    return report
