"""Information-parity preflight audit (ADR-0035 D6).

This is the hard gate that BLOCKS GPU training until the structural prior is shown
to carry real signal and the harness is mock/CPU-free. It returns PASS/FAIL across
six checks:

1. ``H(structure | kāraka) ≫ 0`` — the Paribhāṣā graph is not a relabelling of the
   kāraka parse (carries information beyond the roles).
2. ``VISAYATA`` is 100% transitive — every transitive sentence renders its
   object-relation edge (lossless render).
3. template count ``≫ 2`` — the structural target is not near-degenerate.
4. acceptance ``≥`` target — the realized corpus is faithfully comprehensible.
5. venue off-saturation — arm-A early accuracy in ``[0.55, 0.75]`` (headroom to
   detect a real effect; a saturated venue cannot).
6. zero mock / CPU paths — no ``MockUniformBaseline`` / ``RunMode.MOCK`` / silent
   CUDA→CPU fallback remains in the battery source.

The audit is a pure function over already-measured inputs (depends_on: []), so it
is fully unit-testable without a GPU. The metric inputs (1–4) come from the
paribhāṣā workstream's parity report; (5) from a smoke venue measurement; (6) from
:func:`scan_mock_cpu_paths`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Thresholds (ADR-0035 D6). "≫" is operationalized with explicit margins.
MIN_H_STRUCTURE_GIVEN_KARAKA = 1.0  # bits; >> 0 with a 1-bit margin
MIN_TEMPLATE_COUNT = 10  # >> 2
REQUIRED_VISAYATA_TRANSITIVE = 1.0  # exactly 100%
VENUE_SATURATION_LOW = 0.55
VENUE_SATURATION_HIGH = 0.75

# Source tokens that must NOT appear in the battery (no-mock / no-CPU-fallback).
_FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("MockUniformBaseline", r"\bMockUniformBaseline\b"),
    ("RunMode.MOCK", r"\bRunMode\.MOCK\b"),
    ("silent-cpu-fallback", r"falling back to CPU"),
)


@dataclass(frozen=True)
class ParityCheck:
    """One audit check with its measured value and threshold."""

    id: str
    passed: bool
    value: float
    threshold: str
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class InfoParityInputs:
    """Already-measured inputs to the audit (no GPU needed to evaluate)."""

    h_structure_given_karaka: float
    visayata_transitive_fraction: float
    template_count: int
    acceptance_fraction: float
    acceptance_target: float
    venue_early_accuracy: float
    mock_cpu_path_count: int


@dataclass
class InfoParityReport:
    """Aggregate PASS/FAIL across the six ADR-0035 D6 checks."""

    checks: list[ParityCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }


def scan_mock_cpu_paths(paths: Iterable[Path]) -> list[str]:
    """Return offending ``path:lineno: token`` strings for any forbidden pattern.

    Scans the given source files for ``MockUniformBaseline`` / ``RunMode.MOCK`` /
    silent CUDA→CPU fallback. An empty list means the battery is mock/CPU-clean.
    """
    hits: list[str] = []
    compiled = [(name, re.compile(rx)) for name, rx in _FORBIDDEN_PATTERNS]
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, rx in compiled:
                if rx.search(line):
                    hits.append(f"{path}:{lineno}: {name}")
    return hits


def audit_information_parity(inputs: InfoParityInputs) -> InfoParityReport:
    """Evaluate the six checks and return a PASS/FAIL report."""
    checks = [
        ParityCheck(
            id="C1_H_structure_given_karaka",
            passed=inputs.h_structure_given_karaka >= MIN_H_STRUCTURE_GIVEN_KARAKA,
            value=inputs.h_structure_given_karaka,
            threshold=f">= {MIN_H_STRUCTURE_GIVEN_KARAKA} bits",
            detail="Paribhāṣā graph carries info beyond the kāraka roles (not a relabel).",
        ),
        ParityCheck(
            id="C2_visayata_transitive",
            passed=inputs.visayata_transitive_fraction >= REQUIRED_VISAYATA_TRANSITIVE,
            value=inputs.visayata_transitive_fraction,
            threshold=f"== {REQUIRED_VISAYATA_TRANSITIVE}",
            detail="Every transitive sentence renders its VISAYATA edge (lossless).",
        ),
        ParityCheck(
            id="C3_template_count",
            passed=inputs.template_count >= MIN_TEMPLATE_COUNT,
            value=float(inputs.template_count),
            threshold=f">= {MIN_TEMPLATE_COUNT}",
            detail="Structural target is not near-degenerate (templates >> 2).",
        ),
        ParityCheck(
            id="C4_acceptance_ge_target",
            passed=inputs.acceptance_fraction >= inputs.acceptance_target,
            value=inputs.acceptance_fraction,
            threshold=f">= {inputs.acceptance_target}",
            detail="Realized corpus is faithfully comprehensible at/above target.",
        ),
        ParityCheck(
            id="C5_venue_off_saturation",
            passed=VENUE_SATURATION_LOW <= inputs.venue_early_accuracy <= VENUE_SATURATION_HIGH,
            value=inputs.venue_early_accuracy,
            threshold=f"in [{VENUE_SATURATION_LOW}, {VENUE_SATURATION_HIGH}]",
            detail="Arm-A early accuracy has headroom to reveal a real effect.",
        ),
        ParityCheck(
            id="C6_zero_mock_cpu_paths",
            passed=inputs.mock_cpu_path_count == 0,
            value=float(inputs.mock_cpu_path_count),
            threshold="== 0",
            detail="No mock baseline / RunMode.MOCK / silent CPU fallback in battery.",
        ),
    ]
    return InfoParityReport(checks=checks)
